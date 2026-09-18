#!/usr/bin/env python3
"""Where in time does the gap between two children sit? (Read-only, CPU.)

where_audible.py found no predicate in four aggregate features of a segment, and measured
that the gap sits 16 to 21 dB under the signal in the same half-octaves on the most
divergent segment - the masking configuration that most likely explains why nothing was
heard. Aggregates cannot see a short moment, so this looks frame by frame: 20 ms frames,
the gap-to-signal ratio in each, and the frames sorted by the slope of the signal envelope
into attack, sustain and decay.

What would be actionable: a one-second window whose ratio is well above the segment
average. That window would be the excerpt to listen to, in place of the 4.4 s one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from product_fork_pilot_analysis import outputs  # noqa: E402

import product_nablafx_bench as bench  # noqa: E402

SAMPLE_RATE = 48_000
FRAME = int(0.020 * SAMPLE_RATE)
CONTROL = "butterfly_ssm_seed42_f100_decide_k0"
CHILD = "butterfly_ssm_seed42_f100_decide_k1"


def envelope(signal: np.ndarray) -> np.ndarray:
    usable = signal[: len(signal) // FRAME * FRAME].reshape(-1, FRAME)
    return np.sqrt((usable**2).mean(1))


def main() -> None:
    torch.set_num_threads(6)
    data = bench.data_module("test")
    data.setup("test")
    x = torch.stack([a for a, _ in data.test_dataset])
    control = outputs(CONTROL, x).numpy()[:, 0]
    child = outputs(CHILD, x).numpy()[:, 0]

    out = []
    for i in range(len(control)):
        gain = float(child[i] @ control[i] / (child[i] @ child[i]))
        gap = envelope(control[i] - gain * child[i])
        signal = envelope(control[i])
        # Frames too quiet to matter perceptually are left out of the classification.
        loud = signal > signal.max() / 100
        ratio = 20 * np.log10(np.maximum(gap, 1e-12) / np.maximum(signal, 1e-12))
        slope = np.diff(20 * np.log10(np.maximum(signal, 1e-12)), prepend=np.nan)
        kind = np.where(slope > 3, "attaque", np.where(slope < -3, "chute", "tenue"))
        # The best one-second window, which is what an excerpt would be built around.
        window = 50
        means = np.convolve(np.where(loud, ratio, -120), np.ones(window) / window, "valid")
        best = int(np.argmax(means))
        out.append(
            {
                "segment": i,
                "ratio_median_db": float(np.median(ratio[loud])),
                "ratio_best_second_db": float(means[best]),
                "best_second_at_s": best * 0.020,
                "by_kind_db": {
                    k: float(np.median(ratio[loud & (kind == k)]))
                    for k in ("attaque", "tenue", "chute")
                    if (loud & (kind == k)).any()
                },
            }
        )

    print("segment  mediane dB  meilleure seconde dB  a t =    attaque  tenue  chute")
    for r in out:
        kinds = r["by_kind_db"]
        print(
            f"   {r['segment']:2d}     {r['ratio_median_db']:7.1f}     {r['ratio_best_second_db']:10.1f}"
            f"        {r['best_second_at_s']:5.2f}s"
            + "".join(f"  {kinds.get(k, float('nan')):6.1f}" for k in ("attaque", "tenue", "chute"))
        )
    gainable = max(r["ratio_best_second_db"] - r["ratio_median_db"] for r in out)
    print(f"\nmeilleur gain d'une seconde choisie sur la mediane de son segment: {gainable:+.1f} dB")

    (Path(__file__).parent / "when_audible.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
