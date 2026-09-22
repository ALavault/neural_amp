#!/usr/bin/env python3
"""Where does the gap between the best and worst seed live? (Read-only, CPU, no GPU.)

Pilot C separates seed 42 (test ESR 0.036) from seed 49 (0.178) by a factor 5 in ESR,
where the butterfly children were separated by 1.4. Nobody has listened to that.

Before building a listening page, this says WHICH material carries the gap. The first
butterfly page took its excerpts by position, the listener heard nothing, and
measurement afterwards showed those segments were among the least divergent of the
twelve. The lesson is applied here in advance: the excerpts will be chosen on the
divergence, which makes the test a best case - if the gap is inaudible there, it is
inaudible anywhere on this device.

Two quantities per test segment, both after a least-squares gain per model so that a
level difference cannot pose as a timbre difference:

- the model-to-model gap against the model-to-device error of the better seed. Above 1,
the two models differ from each other more than the good one differs from the device;
- the best 1.5 s window inside the segment, by gap-to-signal ratio in 20 ms frames. The
window stays strictly inside the segment because the protocol resets the model state
at each segment boundary, and a window crossing one would carry that transient.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import product_nablafx_bench as bench  # noqa: E402
from product_fork_pilot_analysis import outputs  # noqa: E402

SAMPLE_RATE = 48_000
WINDOW_SECONDS = 1.5
FRAME = int(0.020 * SAMPLE_RATE)
EDGE = 0.3  # keep this far from each segment boundary, in seconds
GOOD, BAD = "pilotC_decide_seed42", "pilotC_decide_seed49"


def gain(signal: np.ndarray, target: np.ndarray) -> float:
    """Least-squares scalar, so a level offset is not counted as a difference."""
    return float(signal @ target / (signal @ signal))


def best_window(gap: np.ndarray, signal: np.ndarray) -> tuple[float, float]:
    """Start (s) of the best 1.5 s window by gap-to-signal ratio, and that ratio."""
    usable = len(signal) // FRAME * FRAME
    frames = usable // FRAME
    energy_gap = (gap[:usable].reshape(frames, FRAME) ** 2).mean(1)
    energy_sig = (signal[:usable].reshape(frames, FRAME) ** 2).mean(1)
    span = int(WINDOW_SECONDS * SAMPLE_RATE) // FRAME
    lo, hi = int(EDGE * SAMPLE_RATE) // FRAME, frames - int(EDGE * SAMPLE_RATE) // FRAME
    best, at = -np.inf, lo
    for start in range(lo, hi - span):
        ratio = energy_gap[start : start + span].sum() / energy_sig[
            start : start + span
        ].sum()
        if ratio > best:
            best, at = ratio, start
    return at * FRAME / SAMPLE_RATE, float(10.0 * np.log10(best))


def main() -> None:
    torch.set_num_threads(8)
    data = bench.data_module("test")
    data.setup("test")
    inputs = torch.stack([x for x, _ in data.test_dataset])
    targets = torch.stack([y for _, y in data.test_dataset]).numpy()[:, 0]
    good = outputs(GOOD, inputs).numpy()[:, 0]
    bad = outputs(BAD, inputs).numpy()[:, 0]

    rows = []
    for segment in range(len(targets)):
        target = targets[segment].astype(np.float64)
        a = good[segment].astype(np.float64) * gain(good[segment], target)
        b = bad[segment].astype(np.float64) * gain(bad[segment], target)
        err_good = np.linalg.norm(a - target)
        err_bad = np.linalg.norm(b - target)
        gap = np.linalg.norm(a - b)
        start, ratio_db = best_window(a - b, target)
        rows.append(
            {
                "segment": segment,
                "esr_good": float(((a - target) ** 2).sum() / (target**2).sum()),
                "esr_bad": float(((b - target) ** 2).sum() / (target**2).sum()),
                "gap_over_err_good": float(gap / err_good),
                "err_bad_over_err_good": float(err_bad / err_good),
                "best_second_at_s": start,
                "gap_to_signal_db": ratio_db,
            }
        )

    rows.sort(key=lambda r: -r["gap_over_err_good"])
    print(f"{'seg':>4} {'ESR 42':>8} {'ESR 49':>8} {'ecart/err42':>12}"
          f" {'err49/err42':>12} {'meilleure s':>12} {'ecart/signal':>13}")
    for r in rows:
        print(
            f"{r['segment']:>4} {r['esr_good']:8.4f} {r['esr_bad']:8.4f}"
            f" {r['gap_over_err_good']:12.2f} {r['err_bad_over_err_good']:12.2f}"
            f" {r['best_second_at_s']:12.2f} {r['gap_to_signal_db']:12.1f} dB"
        )
    print(
        "\nLes trois segments les plus divergents :"
        f" {', '.join(str(r['segment']) for r in rows[:3])}"
    )
    (Path(__file__).parent / "seed_gap_where.json").write_text(
        json.dumps(rows, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
