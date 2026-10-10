#!/usr/bin/env python3
"""Readings of demo/greybox_deep/predictions.md (deeper grey box, D-0014). Read-only,
GPU (fp32, no TF32), last checkpoints.

VS100R (arms C, GREY, L, F1, F3; seeds 42-44): P of demo/train_length (every
non-low-signal take, di_test at 48 kHz, mean of ln ESR per gain-contour pair, the
unseen pair excluded); the delay-compensated co-reading of coreading_delay.py (rule
unchanged); d_i = s(arm) - s(reference) per seed for each comparison; history slopes
(src/fssr_nam/product/history.py) of each model against the device on the 8 test takes.
JVM410H OD1 speaker output (arms A, GREY, L, F3; seed 42, partitions 925, 736, 688, 744,
369): the bench's test ESR of each record, d_s = ln ESR(arm) - ln ESR(reference) per
partition; history slopes against the device on the 9 test settings.
--arms limits the VS100R arms (dry run on existing runs: --arms C GREY --no-jvm).
Writes demo/greybox_deep/evaluation.json (or evaluation_dry.json with --arms).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import soundfile as sf
import torch
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "demo/train_length"))
import coreading_delay as cd  # noqa: E402
import eval_train_length as tl  # noqa: E402
from nablafx.core import BlackBoxModel  # noqa: E402

from fssr_nam.product import vs100r  # noqa: E402
from fssr_nam.product.history import HISTORY_S, history_deviation  # noqa: E402

ev, bench = tl.ev, tl.bench
SEEDS = (42, 43, 44)
VS100R = {
    "C": "vs100r48fast_film_s{}",
    "GREY": "vs100r48grey_s{}",
    "L": "vs100r48gbL_s{}",
    "F1": "vs100r48gbF1_s{}",
    "F3": "vs100r48gbF3_s{}",
}
VS100R_COMPARISONS = (
    ("F3", "GREY", True),
    ("F3", "C", True),
    ("L", "GREY", False),
    ("F1", "L", False),
    ("F3", "F1", False),
    ("L", "C", False),
)
PARTITIONS = (925, 736, 688, 744, 369)
JVM_A = {
    925: "param_ssm_wavenet_marshall",
    **{p: f"g2_film_joint_s{p}" for p in PARTITIONS[1:]},
}
JVM = {
    "A": JVM_A,
    "GREY": "jvm_grey_s{}",
    "L": "jvm_gbL_s{}",
    "F3": "jvm_gbF3_s{}",
}
JVM_COMPARISONS = (
    ("F3", "GREY", True),
    ("F3", "A", True),
    ("L", "GREY", False),
    ("GREY", "A", False),
    ("L", "A", False),
)
JVM_DATA = (
    ROOT / "datasets/raw/external/tone_twist_param/extracted"
    "/Marshall-JVM410H-ChOD1-trimmed"
)


def exists(run_id: str) -> bool:
    return (bench.RESULTS_DIR / f"{run_id}.json").exists()


def checks(run_id: str) -> dict:
    """Steps of the record and the queue's check line, if any (predictions.md)."""
    record = json.loads((bench.RESULTS_DIR / f"{run_id}.json").read_text())
    lines = [
        json.loads(line)
        for line in (ROOT / "demo/RUNS.jsonl").read_text().splitlines()
        if f'"{run_id}"' in line
    ]
    found = [r for r in lines if r.get("status") == "check" and r["run_id"] == run_id]
    return {
        "global_step": record["global_step"],
        "check_ok": found[-1]["ok"] if found else None,
    }


def load(run_id: str, device: str) -> BlackBoxModel:
    """Model of a bench record at its last checkpoint, on cd.DEVICE."""
    record = json.loads((bench.RESULTS_DIR / f"{run_id}.json").read_text())
    options = (record["ssm"] or {}) | (record.get("greybox") or {})
    args = argparse.Namespace(model=record["model"], device=device, **options)
    processor = bench.build_processor(args)
    path = bench.RUNS_DIR / f"nablafx_{run_id}/checkpoints/last.ckpt"
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    prefix = "model.processor."
    processor.load_state_dict(
        {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    )
    return BlackBoxModel(processor).eval().to(cd.DEVICE)


def parameters(model: BlackBoxModel) -> dict:
    """Total, knob head and processing; head 0 for models without a `head` (SSM-WaveNet
    conditions through FiLM inside its blocks, not separated here)."""
    head = getattr(model.processor, "head", None)
    total = sum(p.numel() for p in model.processor.parameters())
    in_head = sum(p.numel() for p in head.parameters()) if head is not None else 0
    return {"total": total, "head": in_head, "processing": total - in_head}


def p_reading(ln_esr: dict, takes: list[dict]) -> float:
    pairs: dict[tuple, list[float]] = {}
    for t in takes:
        if tuple(t["gain_contour"]) != tl.UNSEEN:
            pairs.setdefault(tuple(t["gain_contour"]), []).append(ln_esr[t["order"]])
    return statistics.mean(statistics.mean(v) for v in pairs.values())


def compare(values: dict, arm: str, reference: str, primary: bool) -> dict:
    d_i = {k: values[arm][k] - values[reference][k] for k in values[reference]}
    out = {"d_i": d_i, "d": statistics.mean(d_i.values())}
    out["ratio"] = math.exp(out["d"])
    out["label"] = tl.verdict(list(d_i.values()))
    if primary:
        out["acceptable"] = tl.acceptable(list(d_i.values()))
    return out


def memory(deviations: list[dict]) -> dict:
    return {
        str(t): {
            "median_delta": statistics.median(d[t]["delta"] for d in deviations),
            "median_gamma_model": statistics.median(
                d[t]["gamma_eval"] for d in deviations
            ),
            "median_gamma_device": statistics.median(
                d[t]["gamma_ref"] for d in deviations
            ),
        }
        for t in HISTORY_S
    }


@torch.no_grad()
def vs100r_readings(arms: list[str]) -> dict:
    takes = [t for t in ev.takes() if not t["low_signal"]]
    data = {t["order"]: ev.audio(t) for t in takes}
    x48 = {k: tuple(resample_poly(s, 1, 4) for s in v) for k, v in data.items()}
    validation = {
        t["order"]: tuple(
            resample_poly(s, 1, 4) for s in vs100r.audio(t, name="di_validation")
        )
        for t in takes
    }
    tests = [t for t in takes if t["split"] == "test"]
    runs, raw, aligned = {}, {}, {}
    for arm in arms:
        for seed in SEEDS:
            run_id = VS100R[arm].format(seed)
            if not exists(run_id):
                print(f"{run_id}: absent", flush=True)
                continue
            model = load(run_id, "vs100r")
            pred = {
                t["order"]: cd.run(model, x48[t["order"]][0], cd.controls(t))
                for t in takes
            }
            ln_esr = {
                t["order"]: math.log(ev.esr(pred[t["order"]], x48[t["order"]][1]))
                for t in takes
            }
            tau, _ = cd.common_delay(model, takes, validation)
            entry = {
                "arm": arm,
                "seed": seed,
                "s_pairs": p_reading(ln_esr, takes),
                "ln_esr_by_take": ln_esr,
                "tau": tau,
                "grid_edge": tau in (cd.DELAYS[0], cd.DELAYS[-1]),
                "s_pairs_aligned": cd.score(model, takes, data, tau)["s_pairs"],
                "parameters": parameters(model),
                **checks(run_id),
                "history": [
                    history_deviation(
                        x48[t["order"]][0], x48[t["order"]][1], pred[t["order"]], 48_000
                    )
                    for t in tests
                ],
            }
            runs[run_id] = entry
            raw.setdefault(arm, {})[seed] = entry["s_pairs"]
            aligned.setdefault(arm, {})[seed] = entry["s_pairs_aligned"]
            print(
                f"{run_id}: s {entry['s_pairs']:.3f}, aligned"
                f" {entry['s_pairs_aligned']:.3f} (tau {tau:+.2f})",
                flush=True,
            )
    complete = {a for a in arms if len(raw.get(a, {})) == len(SEEDS)}
    readings = {}
    for arm, reference, primary in VS100R_COMPARISONS:
        if {arm, reference} <= complete:
            readings[f"{arm}_vs_{reference}"] = {
                "raw": compare(raw, arm, reference, primary),
                "aligned": compare(aligned, arm, reference, False),
            }
    history = {
        arm: memory([h for r in runs.values() if r["arm"] == arm for h in r["history"]])
        for arm in complete
    }
    return {"runs": runs, "readings": readings, "history": history}


@torch.no_grad()
def jvm_readings() -> dict:
    x, _ = sf.read(JVM_DATA / "DRY/test/test.input.wav", dtype="float64")
    settings = sorted((JVM_DATA / "SpeakerOut/test").iterdir())
    targets = {
        s.name: sf.read(next(s.glob("*.wav")), dtype="float64")[0] for s in settings
    }
    runs, ln_esr = {}, {}
    for arm, pattern in JVM.items():
        for p in PARTITIONS:
            run_id = pattern[p] if isinstance(pattern, dict) else pattern.format(p)
            if not exists(run_id):
                print(f"{run_id}: absent", flush=True)
                continue
            record = json.loads((bench.RESULTS_DIR / f"{run_id}.json").read_text())
            model = load(run_id, "marshall")
            deviations = []
            for name, y in targets.items():
                knobs = torch.tensor([float(v[1:]) / 100 for v in name.split("_")])
                pred = cd.run(model, x, knobs)
                deviations.append(history_deviation(x, y, pred, 44_100))
            esr = record["test_last"]["metric/test/esr"]
            runs[run_id] = {
                "arm": arm,
                "partition": p,
                "test_esr": esr,
                "finite": math.isfinite(esr),
                "parameters": parameters(model),
                **checks(run_id),
                "history": memory(deviations),
            }
            ln_esr.setdefault(arm, {})[p] = math.log(esr)
            print(f"{run_id}: test ESR {esr:.4f}", flush=True)
    complete = {a for a, v in ln_esr.items() if len(v) == len(PARTITIONS)}
    readings = {
        f"{arm}_vs_{reference}": compare(ln_esr, arm, reference, primary)
        for arm, reference, primary in JVM_COMPARISONS
        if {arm, reference} <= complete
    }
    return {"runs": runs, "readings": readings}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=list(VS100R))
    parser.add_argument("--no-jvm", action="store_true")
    args = parser.parse_args()
    result = {"vs100r": vs100r_readings(args.arms)}
    if not args.no_jvm:
        result["jvm"] = jvm_readings()
    name = "evaluation.json" if args.arms == list(VS100R) else "evaluation_dry.json"
    (HERE / name).write_text(json.dumps(result, indent=1, default=str) + "\n")
    for part in result.values():
        print(json.dumps(part["readings"], indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
