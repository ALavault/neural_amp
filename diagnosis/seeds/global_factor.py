"""Post-hoc check of DIAGNOSIS_seeds.md section 5: localized or global seed effect?

Read-only. Section 5 reads the fact that 77-81 % of the best-worst gap sits on
the six quietest test segments as the quiet regime being left to the seed. If
instead the seed scales the error of every segment by one factor, the same share
follows from the quiet segments' higher ESR alone. On log ESR (runs x segments)
a global factor is a run main effect; a quiet-specific one is a run x segment
interaction. Runs repeated at the same seed (run_id + "_repeat") give the spread
the seed does not set; their per-segment ESR is computed on CPU like
scripts/product_segment_analysis.py does.

Writes diagnosis/seeds/global_factor.json.
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

import product_nablafx_bench as bench  # noqa: E402, I001
import torch  # noqa: E402

SEGMENTS = json.loads(
    (ROOT / "paper/icassp2027/data/test_segments.json").read_text(encoding="utf-8")
)
RMS = SEGMENTS["segment_rms"]
ORDER = sorted(range(len(RMS)), key=lambda i: RMS[i])
QUIET, LOUD = ORDER[: len(RMS) // 2], ORDER[len(RMS) // 2 :]


def condition(record: dict) -> str:
    model = "ssm" if record["model"] == "ssm-wavenet" else "s4"
    return f"{model}-{'changes' if record.get('polarity_guard') else 'released'}"


def cv(values: list[float]) -> float:
    return 100 * statistics.stdev(values) / statistics.fmean(values)


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(v) for v in values))


def segment_esr(record: dict) -> list[float]:
    """Per-segment test ESR of the last checkpoint, on CPU."""
    if record["run_id"] in SEGMENTS["runs"]:
        return SEGMENTS["runs"][record["run_id"]]["esr"]
    ssm = record["ssm"] or {}
    processor = bench.build_processor(
        SimpleNamespace(
            model=record["model"],
            num_blocks=ssm.get("num_blocks", 8),
            channels=ssm.get("channels", 16),
            state_dim=ssm.get("state_dim", 4),
            output_act=ssm.get("output_act", "tanh"),
            discretization=ssm.get("discretization", "free"),
        )
    )
    state = torch.load(
        ROOT / f"demo/runs/nablafx_{record['run_id']}/checkpoints/last.ckpt",
        map_location="cpu",
        weights_only=False,
    )["state_dict"]
    processor.load_state_dict(
        {
            key.removeprefix("model.processor."): value
            for key, value in state.items()
            if key.startswith("model.processor.")
        }
    )
    processor.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(INPUTS), 8):
            processor.reset_states()
            predictions.append(processor(INPUTS[start : start + 8]))
    prediction = torch.cat(predictions)
    return (
        ((TARGETS - prediction).pow(2).sum(-1) / TARGETS.pow(2).sum(-1))
        .flatten()
        .tolist()
    )


def main() -> None:
    global INPUTS, TARGETS
    torch.set_num_threads(16)
    data = bench.data_module("test")
    data.setup("test")
    INPUTS = torch.stack([x for x, _ in data.test_dataset])
    TARGETS = torch.stack([y for _, y in data.test_dataset])

    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((ROOT / "demo/nablafx_bench").glob("*.json"))
    ]
    groups: dict[str, list[dict]] = {}
    for record in records:
        record["segment_esr"] = segment_esr(record)
        groups.setdefault(condition(record), []).append(record)
    out: dict = {}

    print("== Seed spread (CV over the seeds' first runs) of summaries of the same test")
    print(
        f"{'condition':14s} {'mean ESR':>8s} {'energy ESR':>10s} {'mean sqrt':>9s}"
        f" {'L1':>4s} {'MR-STFT':>7s}"
    )
    for name, group in sorted(groups.items()):
        first = [r for r in group if not r["run_id"].endswith("_repeat")]
        energy = [rms**2 for rms in RMS]
        summaries = {
            "mean_esr": [statistics.fmean(r["segment_esr"]) for r in first],
            "energy_weighted_esr": [
                sum(e * w for e, w in zip(r["segment_esr"], energy, strict=True))
                / sum(energy)
                for r in first
            ],
            "mean_sqrt_esr": [
                statistics.fmean(math.sqrt(e) for e in r["segment_esr"]) for r in first
            ],
            "l1": [r["test_last"]["metric/test/l1"] for r in first],
            "mrstft": [r["test_last"]["metric/test/mrstft"] for r in first],
        }
        out[name] = {"cv": {key: cv(values) for key, values in summaries.items()}}
        c = out[name]["cv"]
        print(
            f"{name:14s} {c['mean_esr']:7.0f}% {c['energy_weighted_esr']:9.0f}%"
            f" {c['mean_sqrt_esr']:8.0f}% {c['l1']:3.0f}% {c['mrstft']:6.0f}%"
        )

    print("\n== Worst over best seed, geometric mean of the per-segment ratios")
    for name, group in sorted(groups.items()):
        first = sorted(
            (r for r in group if not r["run_id"].endswith("_repeat")),
            key=lambda r: statistics.fmean(r["segment_esr"]),
        )
        ratio = [
            w / b
            for w, b in zip(
                first[-1]["segment_esr"], first[0]["segment_esr"], strict=True
            )
        ]
        quiet = geometric_mean([ratio[i] for i in QUIET])
        loud = geometric_mean([ratio[i] for i in LOUD])
        # With one factor r on every segment, the quiet half's share of the
        # absolute gap is just its share of the best run's ESR.
        best = first[0]["segment_esr"]
        share_if_global = sum(best[i] for i in QUIET) / sum(best)
        gap = [w - b for w, b in zip(first[-1]["segment_esr"], best, strict=True)]
        share = sum(gap[i] for i in QUIET) / sum(gap)
        out[name]["worst_over_best"] = {
            "quiet": quiet,
            "loud": loud,
            "quiet_share_of_gap": share,
            "quiet_share_if_one_factor": share_if_global,
        }
        print(
            f"{name:14s} quiet half x{quiet:.2f}, loud half x{loud:.2f}"
            f" | quiet share of the gap {100 * share:.0f}%,"
            f" {100 * share_if_global:.0f}% if one factor scaled every segment"
        )

    print("\n== Log ESR, seeds' first runs x segments: run effect against interaction")
    for name, group in sorted(groups.items()):
        rows = [
            [math.log(e) for e in r["segment_esr"]]
            for r in group
            if not r["run_id"].endswith("_repeat")
        ]
        n, m = len(rows), len(rows[0])
        grand = statistics.fmean(v for row in rows for v in row)
        run_mean = [statistics.fmean(row) for row in rows]
        seg_mean = [statistics.fmean(rows[k][i] for k in range(n)) for i in range(m)]
        ms_run = m * sum((v - grand) ** 2 for v in run_mean) / (n - 1)
        ms_inter = sum(
            (rows[k][i] - run_mean[k] - seg_mean[i] + grand) ** 2
            for k in range(n)
            for i in range(m)
        ) / ((n - 1) * (m - 1))
        out[name]["anova_log_esr"] = {
            "ms_run": ms_run,
            "ms_run_x_segment": ms_inter,
            "df": [n - 1, (n - 1) * (m - 1)],
            "run_sd": math.sqrt(max(ms_run - ms_inter, 0) / m),
            "interaction_sd": math.sqrt(ms_inter),
        }
        a = out[name]["anova_log_esr"]
        print(
            f"{name:14s} F({n - 1},{(n - 1) * (m - 1)}) = {ms_run / ms_inter:5.1f}"
            f" | sd of the run effect {a['run_sd']:.2f},"
            f" of the run x segment interaction {a['interaction_sd']:.2f} (log ESR)"
        )

    print("\n== Same seed, second run (GPU nondeterminism only)")
    by_id = {r["run_id"]: r for r in records}
    for record in records:
        if not record["run_id"].endswith("_repeat"):
            continue
        original = by_id[record["run_id"].removesuffix("_repeat")]
        diff = [
            math.log(b) - math.log(a)
            for a, b in zip(original["segment_esr"], record["segment_esr"], strict=True)
        ]
        out.setdefault("repeats", {})[record["run_id"]] = {
            "mean_esr": [
                statistics.fmean(original["segment_esr"]),
                statistics.fmean(record["segment_esr"]),
            ],
            "log_ratio_mean": statistics.fmean(diff),
            "log_ratio_sd_over_segments": statistics.stdev(diff),
            "per_segment_ratio_quiet_to_loud": [math.exp(diff[i]) for i in ORDER],
        }
        rep = out["repeats"][record["run_id"]]
        print(
            f"{record['run_id']:30s} mean ESR {rep['mean_esr'][0]:.4f} ->"
            f" {rep['mean_esr'][1]:.4f} | log ratio: mean {rep['log_ratio_mean']:+.2f},"
            f" sd over segments {rep['log_ratio_sd_over_segments']:.2f}"
            f" | per segment, quiet to loud:"
            f" {' '.join(f'{v:.2f}' for v in rep['per_segment_ratio_quiet_to_loud'])}"
        )

    (Path(__file__).parent / "global_factor.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
