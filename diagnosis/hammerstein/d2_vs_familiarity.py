#!/usr/bin/env python3
"""Is the models' 2nd-harmonic excess about familiar settings or about the gain?
(NOTE.md, predictions written before running.) Every non-low-signal take, its three
sweeps, the three A48 models: D2 level of model minus device (dB, median over
200 Hz - 8 kHz), with the take's gain, contour and number of training takes at its
(gain, contour) pair. Read-only, CPU. Writes d2_vs_familiarity.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vs100r_hammerstein as vh  # noqa: E402

from fssr_nam.product.hammerstein import calibration, identify  # noqa: E402

RUNS = vh.RUNS["A48"]


def main() -> None:
    torch.set_num_threads(16)
    models = {r: vh.ev.load(r) for r in RUNS}
    takes = [t for t in vh.ev.takes() if not t["low_signal"]]
    band = None
    rows = []
    for take in takes:
        controls = [v / 10 for v in take["gain_contour"]]
        for level in vh.LEVELS:
            loop, fx = vh.section(take, f"sweep_{level}dBFS")
            m = calibration(loop, vh.decimate, vh.RATE, vh.SWEEP, 2, vh.NFFT)
            dev = identify(
                vh.decimate(fx), loop, vh.decimate, vh.RATE, vh.SWEEP, 2, vh.NFFT, m
            )
            if band is None:
                band = (dev.freqs >= 200) & (dev.freqs <= 8000)
            d2_dev = 20 * np.log10(np.abs(dev.d[1][band]) + 1e-30)
            d1_dev = 20 * np.log10(np.abs(dev.d[0][band]) + 1e-30)
            for run, model in models.items():
                y = vh.model_output(model, "A48", loop, controls)
                fit = identify(y, loop, vh.decimate, vh.RATE, vh.SWEEP, 2, vh.NFFT, m)
                d2 = 20 * np.log10(np.abs(fit.d[1][band]) + 1e-30)
                rows.append(
                    {
                        "take": take["order"],
                        "split": take["split"],
                        "gain": take["gain_contour"][0],
                        "contour": take["gain_contour"][1],
                        "pair_count": take["takes_same_pair_in_training"],
                        "level": level,
                        "run": run,
                        "excess_db": float(np.median(d2 - d2_dev)),
                        "device_d2_rel_db": float(np.median(d2_dev - d1_dev)),
                    }
                )
        print(take["order"], take["gain_contour"], flush=True)
    out = {"rows": rows, "tests": {}}
    for level in vh.LEVELS:
        r = [x for x in rows if x["level"] == level]
        rho, p = spearmanr([x["gain"] for x in r], [x["excess_db"] for x in r])
        g5 = [x for x in r if x["gain"] == 5.0]
        fam = [x["excess_db"] for x in g5 if (x["gain"], x["contour"]) == (5.0, 5.0)]
        rare = [x["excess_db"] for x in g5 if (x["gain"], x["contour"]) != (5.0, 5.0)]
        out["tests"][str(level)] = {
            "spearman_gain_excess": rho,
            "p": p,
            "gain5_pair55_median": float(np.median(fam)),
            "n55": len(fam),
            "gain5_other_median": float(np.median(rare)),
            "n_other": len(rare),
            "by_gain": {
                str(g): float(np.median([x["excess_db"] for x in r if x["gain"] == g]))
                for g in sorted({x["gain"] for x in r})
            },
        }
        print(level, json.dumps(out["tests"][str(level)]), flush=True)
    (HERE / "d2_vs_familiarity.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
