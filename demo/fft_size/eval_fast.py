#!/usr/bin/env python3
"""Evaluation of predictions_fast_notf32.md. Read-only, CPU.

Arm A (current setting: FFT pow2, TF32) against arm C (FFT fast, no TF32), paired by
seed. 8 unseen VS100R test takes at 48 kHz, scored region and decimation of
product_vs100r_eval, as in eval_192k.py; inference with the default FFT length.
M1: s = mean over takes of ln ESR per run, d_i = s(C) - s(A) per seed, d = mean.
M2 (descriptive): |e| at the 16 kHz third-octave, mean over seeds, C minus A.
Readings only when the six runs exist. Writes eval_fast.json.
"""

from __future__ import annotations

import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import product_nablafx_bench as bench  # noqa: E402
import product_vs100r_eval as ev  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "eval_192k", ROOT / "demo/capture/valvestate/eval_192k.py"
)
e192 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e192)

PAIRS = {
    42: ("vs100r_gc_film_joint_s925_r2", "vs100r48fast_film_s42"),
    43: ("vs100r48_film_s43", "vs100r48fast_film_s43"),
    44: ("vs100r48_film_s44", "vs100r48fast_film_s44"),
}


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(16)
    takes = [t for t in ev.takes() if t["split"] == "test" and not t["low_signal"]]
    data = {t["order"]: ev.audio(t) for t in takes}
    real = np.concatenate(
        [ev.scored(resample_poly(fx, 1, 4)) for _, fx in data.values()]
    )
    real_levels = e192.levels(real)
    result = {"runs": {}}
    for seed, pair in PAIRS.items():
        for arm, run_id in zip(("A", "C"), pair, strict=True):
            if not (bench.RESULTS_DIR / f"{run_id}.json").exists():
                print(f"{run_id}: absent", flush=True)
                continue
            model = ev.load(run_id)
            esrs, outs = {}, []
            for take in takes:
                loop, fx = data[take["order"]]
                controls = torch.tensor([v / 10 for v in take["gain_contour"]])
                y = ev.run(model, resample_poly(loop, 1, 4), controls)
                esrs[take["order"]] = ev.esr(y, resample_poly(fx, 1, 4))
                outs.append(ev.scored(y))
            lv = e192.levels(np.concatenate(outs))
            result["runs"][run_id] = {
                "seed": seed,
                "arm": arm,
                "esr_by_take": esrs,
                "s": statistics.mean(math.log(v) for v in esrs.values()),
                "abs_e_16k": abs(lv[e192.TOP] - real_levels[e192.TOP]),
            }
            print(f"{run_id}: s {result['runs'][run_id]['s']:.3f}", flush=True)
    runs = result["runs"]
    if all(r in runs for pair in PAIRS.values() for r in pair):
        d_i = {s: runs[c]["s"] - runs[a]["s"] for s, (a, c) in PAIRS.items()}
        d = statistics.mean(d_i.values())
        values = list(d_i.values())
        if d <= -0.1 and all(v < 0 for v in values):
            m1 = "C meilleur"
        elif d >= 0.1 and all(v > 0 for v in values):
            m1 = "C moins bon"
        elif abs(d) <= 0.05 and all(abs(v) <= 0.15 for v in values):
            m1 = "Équivalent"
        else:
            m1 = "non tranché"
        top = {
            arm: statistics.mean(runs[pair[i]]["abs_e_16k"] for pair in PAIRS.values())
            for i, arm in enumerate(("A", "C"))
        }
        result["readings"] = {
            "d_i": d_i,
            "d": d,
            "M1": m1,
            "M2_abs_e16k_C_minus_A": top["C"] - top["A"],
        }
        print(json.dumps(result["readings"], indent=1, ensure_ascii=False))
    (HERE / "eval_fast.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
