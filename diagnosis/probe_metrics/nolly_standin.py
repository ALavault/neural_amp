#!/usr/bin/env python3
"""Synthetic stand-in for the Nolly renders, to dry-run nolly_reading.py before any
render exists (NOLLY_READING.md). Not a model of Nolly: a fixed DSP chain whose five
knobs move tone filters around a saturating stage with a slow sag, plus a known long
FIR as the cab. Writes 32-bit float WAV files named as in demo/capture/v2/settings.csv,
plus loopback_software.wav (the source itself) and center_stereo.wav (L == R).

    nolly_standin.py OUT --orders 1,20,39,...
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, oaconvolve, sosfilt

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / "demo/capture/v2"
RATE = 192_000
DB_PER_STEP = 2.4  # tone filters: dB per knob step away from 5


def biquad(kind: str, f0: float, gain_db: float, q: float = 0.707):
    """RBJ cookbook peaking and shelving filters."""
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / RATE
    alpha = np.sin(w) / (2 * q)
    c = np.cos(w)
    if kind == "peak":
        b = [1 + alpha * a, -2 * c, 1 - alpha * a]
        den = [1 + alpha / a, -2 * c, 1 - alpha / a]
    else:
        s = 2 * np.sqrt(a) * alpha
        sign = 1 if kind == "high" else -1
        b = [
            a * ((a + 1) + sign * (a - 1) * c + s),
            -2 * sign * a * ((a - 1) + sign * (a + 1) * c),
            a * ((a + 1) + sign * (a - 1) * c - s),
        ]
        den = [
            (a + 1) - sign * (a - 1) * c + s,
            2 * sign * ((a - 1) - sign * (a + 1) * c),
            (a + 1) - sign * (a - 1) * c - s,
        ]
    return np.array(b) / den[0], np.array(den) / den[0]


def cab_ir() -> np.ndarray:
    """Known long linear filter: decaying noise (20 ms), 150 ms, band 80 Hz-5 kHz."""
    n = int(0.15 * RATE)
    rng = np.random.default_rng(7)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / (0.02 * RATE))
    ir = sosfilt(butter(4, [80, 5_000], "bandpass", fs=RATE, output="sos"), ir)
    return ir / np.sqrt(np.sum(ir**2))


def render(x: np.ndarray, knobs: dict, cab: bool) -> np.ndarray:
    v = x
    for kind, f0, knob in (("low", 120, "B"), ("peak", 700, "M"), ("high", 3_000, "T")):
        v = lfilter(*biquad(kind, f0, DB_PER_STEP * (knobs[knob] - 5)), v)
    drive = 10 ** ((6 * (knobs["G"] - 5) + 26) / 20)
    envelope = np.sqrt(lfilter([1 - 0.99995], [1, -0.99995], v**2))  # ~100 ms
    y = np.tanh(drive * v / (1 + 2 * envelope)) * 0.5
    y = lfilter(*biquad("high", 5_000, DB_PER_STEP * (knobs["P"] - 5)), y)
    return oaconvolve(y, cab_ir())[: len(y)] if cab else y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--orders", required=True)
    args = parser.parse_args()
    layout = json.loads((CAPTURE / "layout.json").read_text())
    x = sf.read(ROOT / layout["file"], dtype="float64")[0]
    args.out.mkdir(parents=True, exist_ok=True)
    sf.write(args.out / "loopback_software.wav", x.astype(np.float32), RATE, "FLOAT")
    with (CAPTURE / "settings.csv").open() as f:
        rows = {int(r["order"]): r for r in csv.DictReader(f)}
    centre = render(x, dict.fromkeys("GBMTP", 5.0), cab=False).astype(np.float32)
    sf.write(
        args.out / "center_stereo.wav", np.stack([centre, centre], 1), RATE, "FLOAT"
    )
    for order in (int(o) for o in args.orders.split(",")):
        row = rows[order]
        knobs = {k: float(row[k]) for k in "GBMTP"}
        y = render(x, knobs, cab=row["cab"] == "on").astype(np.float32)
        sf.write(args.out / row["file"], y, RATE, "FLOAT")
        print(row["file"], flush=True)


if __name__ == "__main__":
    main()
