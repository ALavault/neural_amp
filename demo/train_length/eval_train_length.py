#!/usr/bin/env python3
"""Evaluation of demo/train_length/predictions.md. Read-only, CPU.

Reference arm C (all the training segments) against arms L10 and L20 (10 and 20 evenly
spaced 3 s segments of each take), paired by seed. Takes: every non-low-signal VS100R
take, test and test_seen; section di_test (never trained on in any arm), scored region
and decimation of product_vs100r_eval.py; inference with the default FFT length.
Two checkpoints per run: last (primary) and best (secondary: the one the bench kept,
saved every 100 steps and ranked on the latest validation loss at saving time).

P (primary): per run, mean of ln ESR over the takes of each gain-contour pair, then mean
over the pairs, the unseen pair (2.5, 7.5) excluded; d_i = s(arm) - s(C) per seed;
verdict and acceptance bound (d <= 0.10 and every d_i <= 0.25).
Descriptive: take 16 alone; the 8 test takes as in eval_fast.py.
Readings only when the nine runs exist. Writes eval_train_length.json.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import torch
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import product_nablafx_bench as bench  # noqa: E402
import product_vs100r_eval as ev  # noqa: E402

ARMS = {
    "C": "vs100r48fast_film_s{}",
    "L10": "vs100r48len10_film_s{}",
    "L20": "vs100r48len20_film_s{}",
}
SEEDS = (42, 43, 44)
UNSEEN = (2.5, 7.5)


def load(run_id: str, checkpoint: str):
    """ev.load reads last.ckpt; for best, the record's best checkpoint instead."""
    model = ev.load(run_id)
    if checkpoint == "best":
        name = json.loads((bench.RESULTS_DIR / f"{run_id}.json").read_text())[
            "best_checkpoint"
        ]
        path = bench.RUNS_DIR / f"nablafx_{run_id}/checkpoints/{name}"
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        prefix = "model.processor."
        model.processor.load_state_dict(
            {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        )
    return model.eval()


def verdict(values: list[float]) -> str:
    d = statistics.mean(values)
    if d <= -0.1 and all(v < 0 for v in values):
        return "meilleur"
    if d >= 0.1 and all(v > 0 for v in values):
        return "moins bon"
    if abs(d) <= 0.05 and all(abs(v) <= 0.15 for v in values):
        return "équivalent"
    return "non tranché"


def score(model, takes: list[dict], data: dict) -> dict:
    """ln ESR per take, then the primary and descriptive aggregates."""
    ln_esr = {}
    for take in takes:
        loop, fx = data[take["order"]]
        controls = torch.tensor([v / 10 for v in take["gain_contour"]])
        y = ev.run(model, resample_poly(loop, 1, 4), controls)
        ln_esr[take["order"]] = math.log(ev.esr(y, resample_poly(fx, 1, 4)))
    pairs: dict[tuple, list[float]] = {}
    for take in takes:
        if tuple(take["gain_contour"]) != UNSEEN:
            pairs.setdefault(tuple(take["gain_contour"]), []).append(
                ln_esr[take["order"]]
            )
    return {
        "ln_esr_by_take": ln_esr,
        "s_pairs": statistics.mean(statistics.mean(v) for v in pairs.values()),
        "n_pairs": len(pairs),
        "s_test8": statistics.mean(
            ln_esr[t["order"]] for t in takes if t["split"] == "test"
        ),
        "take16": ln_esr.get(16),
    }


def acceptable(values: list[float]) -> bool:
    """predictions.md acceptance bound, independent of the verdict label."""
    return statistics.mean(values) <= 0.10 and all(v <= 0.25 for v in values)


def readings(runs: dict) -> dict:
    """Paired differences arm - C per seed, for each checkpoint and aggregate; verdict
    and acceptance for the primary s_pairs only, the others descriptive."""
    out = {}
    for arm in ("L10", "L20"):
        for checkpoint in ("last", "best"):
            row = {}
            for key in ("s_pairs", "s_test8", "take16"):
                d_i = {
                    seed: runs[ARMS[arm].format(seed)][checkpoint][key]
                    - runs[ARMS["C"].format(seed)][checkpoint][key]
                    for seed in SEEDS
                }
                row[key] = {"d_i": d_i, "d": statistics.mean(d_i.values())}
                if key == "s_pairs":
                    row[key]["verdict"] = verdict(list(d_i.values()))
                    row[key]["acceptable"] = acceptable(list(d_i.values()))
            out[f"{arm}_{checkpoint}"] = row
    return out


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(16)
    takes = [t for t in ev.takes() if not t["low_signal"]]
    data = {t["order"]: ev.audio(t) for t in takes}
    result = {"takes": [t["order"] for t in takes], "runs": {}}
    for arm, pattern in ARMS.items():
        for seed in SEEDS:
            run_id = pattern.format(seed)
            if not (bench.RESULTS_DIR / f"{run_id}.json").exists():
                print(f"{run_id}: absent", flush=True)
                continue
            entry = {"arm": arm, "seed": seed}
            for checkpoint in ("last", "best"):
                entry[checkpoint] = score(load(run_id, checkpoint), takes, data)
                print(f"{run_id} {checkpoint}: s {entry[checkpoint]['s_pairs']:.3f}")
            result["runs"][run_id] = entry
    if len(result["runs"]) == len(ARMS) * len(SEEDS):
        result["readings"] = readings(result["runs"])
        print(json.dumps(result["readings"], indent=1, ensure_ascii=False))
    (HERE / "eval_train_length.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
