#!/usr/bin/env python3
"""Hypothesis born during the measures (DIAGNOSIS_vs100r_192k.md §6): at gain 2.5, does
the contour knob change the real amp's output, and does take 16 sound like contour 7.5?
Every take plays the same di_test signal, so real outputs compare directly (ESR at 48
kHz on the scored region, after an oracle delay of +-2 samples). Read-only, CPU, no model.
Writes real_contour.json."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import product_vs100r_eval as ev  # noqa: E402


def esr_aligned(a: np.ndarray, b: np.ndarray) -> float:
    """min over integer lags -2..2 of ESR(a vs reference b), on the scored region."""
    best = np.inf
    for lag in range(-2, 3):
        x = np.roll(a, lag)
        x, y = ev.scored(x), ev.scored(b)
        best = min(best, float(((y - x) ** 2).sum() / (y**2).sum()))
    return best


def main() -> None:
    takes = [t for t in ev.takes() if not t["low_signal"]]
    real = {}
    for t in takes:
        real[t["order"]] = (
            tuple(t["gain_contour"]),
            resample_poly(ev.audio(t)[1], 1, 4),
        )
    n = min(len(v[1]) for v in real.values())
    pairs = {}
    for (i, (pi, ri)), (j, (pj, rj)) in itertools.combinations(real.items(), 2):
        pairs[f"{i}-{j}"] = {
            "pair_i": pi,
            "pair_j": pj,
            "esr": esr_aligned(ri[:n], rj[:n]),
        }
    (HERE / "real_contour.json").write_text(json.dumps(pairs, indent=1) + "\n")
    rows = sorted(pairs.items(), key=lambda kv: kv[1]["esr"])
    same = [v["esr"] for v in pairs.values() if v["pair_i"] == v["pair_j"]]
    print(
        "même couple (répétabilité) : médiane %.4f, max %.4f, n %d"
        % (np.median(same), max(same), len(same))
    )
    for key, v in rows:
        if "16" in key.split("-"):
            print(key, v["pair_i"], v["pair_j"], "%.4f" % v["esr"])


if __name__ == "__main__":
    main()
