#!/usr/bin/env python3
"""Is the VS100R model early against the device? (Product lane, CPU. POST HOC.)

Written on 2026-09-27 after the evaluation of vs100r_gc_film_joint_s925_r2, not
declared in predictions_model.md. The `delta` arm of P5 fitted the device about five
times better than the 48 kHz model, and delaying the 48 kHz output by about one sample
did the same on four takes. Here, on every OD2 take (test section, as
product_vs100r_eval.py): the model output at 48 kHz, delayed by d samples (fractional,
FFT phase), scored against the device for d from -2 to +3 by 0.1. Per take: best d and
its ESR; over the 35 takes that are not low-signal: the common d that minimises the
median ESR.

Writes demo/capture/valvestate/timing_shift.json.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import product_vs100r_eval as ev  # noqa: E402

RUN = "vs100r_gc_film_joint_s925_r2"
SHIFTS = np.round(np.arange(-2.0, 3.0001, 0.1), 1)


def delayed(x: np.ndarray, d: float) -> np.ndarray:
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x))
    return np.fft.irfft(spectrum * np.exp(-2j * np.pi * freqs * d), len(x))


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(16)
    model = ev.load(RUN)
    rows = []
    for take in ev.takes():
        loop, fx = ev.audio(take)
        x48, y48 = resample_poly(loop, 1, 4), resample_poly(fx, 1, 4)
        controls = torch.tensor([v / 10 for v in take["gain_contour"]])
        m = ev.run(model, x48, controls)
        curve = [ev.esr(delayed(m, d), y48) for d in SHIFTS]
        best = int(np.argmin(curve))
        rows.append(
            {
                "order": take["order"],
                "split": take["split"],
                "gain_contour": take["gain_contour"],
                "low_signal": take["low_signal"],
                "esr_0": curve[int(np.argmin(np.abs(SHIFTS)))],
                "best_shift": float(SHIFTS[best]),
                "esr_best": curve[best],
                "curve": curve,
            }
        )
        print(
            f"prise {take['order']:>2} {take['gain_contour']}:"
            f" ESR {rows[-1]['esr_0']:.4f} -> {curve[best]:.4f}"
            f" a {SHIFTS[best]:+.1f} ech.",
            flush=True,
        )
    kept = [r for r in rows if not r["low_signal"]]
    medians = [
        statistics.median(r["curve"][i] for r in kept) for i in range(len(SHIFTS))
    ]
    common = int(np.argmin(medians))
    summary = {
        "median_esr_0": medians[int(np.argmin(np.abs(SHIFTS)))],
        "common_shift": float(SHIFTS[common]),
        "median_esr_common_shift": medians[common],
        "median_best_shift": statistics.median(r["best_shift"] for r in kept),
        "median_esr_own_best": statistics.median(r["esr_best"] for r in kept),
        "test_median_esr_common_shift": statistics.median(
            r["curve"][common] for r in kept if r["split"] == "test"
        ),
    }
    print(json.dumps(summary, indent=1))
    out = Path(__file__).with_name("timing_shift.json")
    out.write_text(
        json.dumps(
            {"shifts": SHIFTS.tolist(), "summary": summary, "rows": rows}, indent=1
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
