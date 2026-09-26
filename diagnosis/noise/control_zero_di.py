#!/usr/bin/env python3
"""Control of predictions_quiet.md: DI set to zero on quiet frames. (Read-only, CPU.)

Declared in diagnosis/noise/predictions_quiet.md (commit c274a78) before this ran. Same
test set, windows and band as quiet_frames.py; only the input changes. If the quiet
output is the model amplifying the quiet DI (quantisation residue), zeroing it removes
the output; if it comes from state tails, biases or arithmetic, it stays.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from quiet_frames import (
    BAND,
    FRAME,
    RATE,
    RUNS,
    SKIP,
    WIN,
    frame_masks,
    processor,
    test_set,
    weights,
    windows,
)


def outputs(model, x: np.ndarray) -> np.ndarray:
    outs = []
    for start in range(0, len(x), 8):
        model.reset_states()
        outs.append(model(torch.from_numpy(x[start : start + 8, None]))[:, 0].numpy())
    return np.concatenate(outs)


def band_power(m: np.ndarray, spots: list[tuple[int, int]]) -> float:
    taper = np.hanning(WIN)
    spectra = np.fft.rfft(np.stack([m[s, t : t + WIN] for s, t in spots]) * taper)
    freqs = np.fft.rfftfreq(WIN, 1 / RATE)
    band = (freqs >= BAND[0]) & (freqs < BAND[1])
    return float((np.abs(spectra[:, band]) ** 2).mean(0).sum())


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(8)
    x, _ = test_set()
    floor, quiet, _ = frame_masks(x)
    spots = windows(quiet, x.shape[1] - SKIP)
    kept = x[:, SKIP:]
    count = kept.shape[1] // FRAME
    power = (kept[:, : count * FRAME].reshape(len(x), count, FRAME) ** 2).mean(-1)
    silent = np.repeat(power < 10 * floor, FRAME, axis=1)
    zeroed = x.copy()
    zeroed[:, SKIP : SKIP + count * FRAME][silent] = 0.0
    print(f"{len(spots)} fenetres ; {silent.mean():.1%} des echantillons mis a zero")
    model = processor()
    rows = {}
    for run in RUNS:
        model.load_state_dict(weights(run))
        model.eval()
        p = band_power(outputs(model, x), spots)
        p0 = band_power(outputs(model, zeroed), spots)
        rows[run] = {"db_zero_vs_measured": 10 * np.log10(p0 / p)}
        print(f"{run:22s} P0/P {rows[run]['db_zero_vs_measured']:+.1f} dB", flush=True)
    median = statistics.median(r["db_zero_vs_measured"] for r in rows.values())
    verdict = (
        "amplification du DI calme"
        if median <= -10
        else "etats, biais ou calcul"
        if median >= -3
        else "cause mixte"
    )
    print(f"\nmediane P0/P {median:+.1f} dB -> {verdict}")
    (Path(__file__).parent / "control_zero_di.json").write_text(
        json.dumps({"rows": rows, "median_db": median, "verdict": verdict}, indent=1)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())
