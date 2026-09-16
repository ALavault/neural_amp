"""Measurements for DIAGNOSIS_seeds.md. Read-only: records, logs, checkpoints.

Writes diagnosis/seeds/measurements.json and prints one table per hypothesis.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import lightning as pl  # noqa: E402
import product_nablafx_bench as bench  # noqa: E402
import torch  # noqa: E402

ESR = "metric/test/esr"
SEGMENTS = json.loads(
    (ROOT / "paper/icassp2027/data/test_segments.json").read_text(encoding="utf-8")
)
SAMPLE_RATE = 48_000


def condition(record: dict) -> str:
    model = "ssm" if record["model"] == "ssm-wavenet" else "s4"
    return f"{model}-{'changes' if record.get('polarity_guard') else 'released'}"


def load_records() -> list[dict]:
    records = []
    for path in sorted((ROOT / "demo/nablafx_bench").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["condition"] = condition(record)
        records.append(record)
    return records


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        for position, index in enumerate(order):
            ranks[index] = float(position)
        return ranks

    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def processor_args(record: dict) -> SimpleNamespace:
    ssm = record["ssm"] or {}
    return SimpleNamespace(
        model=record["model"],
        num_blocks=ssm.get("num_blocks", 8),
        channels=ssm.get("channels", 16),
        state_dim=ssm.get("state_dim", 4),
        output_act=ssm.get("output_act", "tanh"),
        discretization=ssm.get("discretization", "free"),
    )


def split_statistics(record: dict) -> dict:
    """Rebuild the seed's train/val split: same RNG order as the bench."""
    pl.seed_everything(record["seed"], workers=True)
    torch.set_float32_matmul_precision("high")
    bench.build_processor(processor_args(record))
    data = bench.data_module("trainval")
    data.setup("fit")
    pooled = torch.stack([y for _, y in data.trainval_dataset])
    level = pooled.pow(2).mean(-1).sqrt().flatten()
    val_indices = list(data.val_dataset.indices)
    train_indices = list(data.train_dataset.indices)
    quiet_threshold = float(level.quantile(0.1))
    return {
        "val_segments": len(val_indices),
        "val_rms_min": float(level[val_indices].min()),
        "val_rms_median": float(level[val_indices].median()),
        "train_rms_min": float(level[train_indices].min()),
        "train_quiet_count": int((level[train_indices] < quiet_threshold).sum()),
        "train_energy": float(pooled[train_indices].pow(2).sum()),
    }


def pole_statistics(record: dict) -> dict:
    """Time constants of the learned poles, tau = -1 / ln|p|, in ms at 48 kHz."""
    state = torch.load(
        ROOT / f"demo/runs/nablafx_{record['run_id']}/checkpoints/last.ckpt",
        map_location="cpu",
        weights_only=False,
    )["state_dict"]
    taus = []
    prefix = "model.processor.blocks."
    blocks = {key for key in state if key.startswith(prefix)}
    for block in sorted({key.split(".")[3] for key in blocks}):
        log_a = state[f"{prefix}{block}.ssm.log_A_real"]
        log_dt = state[f"{prefix}{block}.ssm.log_dt"].unsqueeze(-1)
        magnitude = torch.exp(-torch.exp(log_a) * torch.exp(log_dt))
        taus += (-1.0 / torch.log(magnitude) / SAMPLE_RATE * 1e3).flatten().tolist()
    taus_sorted = sorted(taus)
    return {
        "poles": len(taus),
        "tau_ms_median": statistics.median(taus_sorted),
        "tau_ms_max": taus_sorted[-1],
        "poles_over_1ms": sum(t > 1 for t in taus),
        "poles_over_10ms": sum(t > 10 for t in taus),
    }


def energy_weighted_esr(run_id: str) -> float:
    """Total error energy over total target energy, instead of the segment mean."""
    esr = SEGMENTS["runs"][run_id]["esr"]
    energy = [rms**2 for rms in SEGMENTS["segment_rms"]]
    return sum(e * w for e, w in zip(esr, energy, strict=True)) / sum(energy)


def main() -> None:
    records = load_records()
    conditions = sorted({r["condition"] for r in records})
    out: dict = {"runs": {}, "conditions": {}}

    print("== H1: stopping point ==")
    for name in conditions:
        group = sorted(
            (r for r in records if r["condition"] == name), key=lambda r: r["seed"]
        )
        steps = [r["global_step"] for r in group]
        esr = [r["test_last"][ESR] for r in group]
        print(
            f"{name:14s} steps {steps} ESR {[round(e, 4) for e in esr]}"
            f" spearman(step, ESR) {spearman(steps, esr):+.2f}"
        )
    pooled = spearman(
        [r["global_step"] for r in records], [r["test_last"][ESR] for r in records]
    )
    print(f"pooled over the {len(records)} runs: {pooled:+.2f}")

    print("\n== H2: train/val split ==")
    for record in records:
        if record["condition"] in ("ssm-changes", "s4-changes"):
            stats = split_statistics(record)
            out["runs"].setdefault(record["run_id"], {})["split"] = stats
            quiet = SEGMENTS["runs"][record["run_id"]]["quiet"]
            print(
                f"{record['run_id']:28s} val min rms {stats['val_rms_min']:.4f}"
                f" train min rms {stats['train_rms_min']:.4f}"
                f" train quiet {stats['train_quiet_count']:2d}"
                f" | test ESR {record['test_last'][ESR]:.4f} quiet half {quiet:.4f}"
            )

    print("\n== H3: learned pole time constants (SSM only) ==")
    for record in records:
        if record["model"] != "ssm-wavenet":
            continue
        stats = pole_statistics(record)
        out["runs"].setdefault(record["run_id"], {})["poles"] = stats
        print(
            f"{record['run_id']:28s} median {stats['tau_ms_median']:6.2f} ms"
            f" max {stats['tau_ms_max']:8.1f} ms"
            f" >1ms {stats['poles_over_1ms']:3d} >10ms {stats['poles_over_10ms']:3d}"
            f" | test ESR {record['test_last'][ESR]:.4f}"
        )

    print("\n== H4: per-segment mean against energy-weighted ESR ==")
    for name in conditions:
        group = sorted(
            (r for r in records if r["condition"] == name), key=lambda r: r["seed"]
        )
        mean_esr = [r["test_last"][ESR] for r in group]
        weighted = [energy_weighted_esr(r["run_id"]) for r in group]
        cv = lambda v: 100 * statistics.stdev(v) / statistics.fmean(v)  # noqa: E731
        out["conditions"][name] = {
            "segment_mean_esr": mean_esr,
            "energy_weighted_esr": weighted,
            "cv_segment_mean": cv(mean_esr),
            "cv_energy_weighted": cv(weighted),
        }
        print(
            f"{name:14s} segment mean {[round(v, 4) for v in mean_esr]}"
            f" CV {cv(mean_esr):3.0f}%"
            f" | energy weighted {[round(v, 4) for v in weighted]}"
            f" CV {cv(weighted):3.0f}%"
        )

    (Path(__file__).parent / "measurements.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
