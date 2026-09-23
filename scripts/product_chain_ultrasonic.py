#!/usr/bin/env python3
"""Test D: does a first stage's ultrasonic content change a second stage in-band?

demo/chain_ultrasonic/predictions.md. Everything runs at 192 kHz on the M1 synthetic
systems. The first stage's output goes on either full-band, or through the frozen
192 -> 48 kHz decimation chain and back up (what a 48 kHz capture or model can pass on).
The second stage's two outputs are brought down to 48 kHz by the same chain and compared
in dB, on segment interiors. Controls: identity second stage, and a 5 kHz "cabinet"
low-pass after the second stage.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import numpy as np
from scipy.signal import butter, firwin, lfilter, resample_poly

from fssr_nam.data.excitations import EXCITATIONS
from fssr_nam.data.systems import SYSTEMS
from fssr_nam.dsp.multirate import derive_reference_rates

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo/chain_ultrasonic/results.json"
RATE = 192_000
SECONDS = 2.0
TRIM = int(0.1 * 48_000)
FIRST = ("tanh", "asymmetric_waveshaper", "memory_clipper", "hammerstein")
SECOND = ("identity", "tanh", "asymmetric_waveshaper", "wiener", "slow_sag")
INPUTS = ("procedural_plucks", "multisine", "two_tone", "colored_noise")


def band_limit(signal: np.ndarray, cutoff: float) -> np.ndarray:
    """Zero-phase linear FIR low-pass at 192 kHz (a DI has nothing above cutoff)."""
    kernel = firwin(2047, cutoff, fs=RATE, window=("kaiser", 8.6))
    return np.convolve(signal, kernel, mode="same")


def to_48k(signal: np.ndarray) -> np.ndarray:
    return np.asarray(derive_reference_rates(signal)[48_000], dtype=np.float64)


def stage(name: str, signal: np.ndarray) -> np.ndarray:
    if name == "identity":
        return signal
    return np.asarray(SYSTEMS[name](signal, RATE), dtype=np.float64)


def difference_db(test: np.ndarray, reference: np.ndarray) -> float:
    test, reference = test[TRIM:-TRIM], reference[TRIM:-TRIM]
    return float(10 * np.log10(np.sum((test - reference) ** 2) / np.sum(reference**2)))


def main() -> None:
    cab_b, cab_a = butter(4, 5_000, fs=48_000)
    results: dict = {}
    for excitation in INPUTS:
        x = band_limit(
            np.asarray(EXCITATIONS[excitation](RATE, SECONDS, 0), dtype=np.float64),
            20_000,
        )
        rows = results.setdefault(excitation, {})
        for first in FIRST:
            full = stage(first, x)
            lim = resample_poly(to_48k(full), 4, 1)[: full.size]
            above = 10 * np.log10(
                np.sum((full - lim)[4 * TRIM : -4 * TRIM] ** 2)
                / np.sum(full[4 * TRIM : -4 * TRIM] ** 2)
            )
            for second in SECOND:
                z_full, z_lim = to_48k(stage(second, full)), to_48k(stage(second, lim))
                rows[f"{first}->{second}"] = {
                    "db": difference_db(z_lim, z_full),
                    "db_cab": difference_db(
                        lfilter(cab_b, cab_a, z_lim), lfilter(cab_b, cab_a, z_full)
                    ),
                    "stage1_above_24k_db": float(above),
                }
                print(excitation, first, second, rows[f"{first}->{second}"], flush=True)
    summary = {}
    for excitation, rows in results.items():
        main_pairs = [v for k, v in rows.items() if not k.endswith("identity")]
        summary[excitation] = {
            "median_db": median(v["db"] for v in main_pairs),
            "max_db": max(v["db"] for v in main_pairs),
            "median_db_cab": median(v["db_cab"] for v in main_pairs),
            "identity_max_db": max(
                v["db"] for k, v in rows.items() if k.endswith("identity")
            ),
        }
    OUT.write_text(
        json.dumps({"summary": summary, "pairs": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
