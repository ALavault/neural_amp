#!/usr/bin/env python3
"""Delay between the two Zen Go inputs on one shared signal (INTER_INPUT_TEST.md).

Product lane. Written before the takes. For each take, a stereo WAV (channel 0 = input
1, channel 1 = input 2) or two mono WAVs recorded together: the delay of input 2
relative to input 1 (positive: input 2 lags), per octave band from 500 Hz to 8 kHz, as
minus the coherence-gated, |cross-spectrum|-weighted slope of the unwrapped
cross-spectrum phase (the method of group_delay.py), at the file's own rate. Reported
in samples at that rate, in samples at 48 kHz and in microseconds, with the median over
the bands, the number of valid bands and their spread (standard deviation, at 48 kHz).
With --pair, both files must come from the same take (same capture, same clock).

Usage: inter_input_delay.py <take.wav> [...]
       inter_input_delay.py --pair <in1.wav> <in2.wav>
Writes demo/capture/valvestate/inter_input_delay.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import csd, welch

CENTRES = (500, 1_000, 2_000, 4_000, 8_000)


def band_delays(a: np.ndarray, b: np.ndarray, rate: int) -> dict[str, float | None]:
    """Delay of b relative to a, in samples at `rate`, per octave band."""
    nperseg = 4096 * max(1, rate // 48_000)
    freqs, s_ab = csd(a, b, rate, nperseg=nperseg)
    _, s_aa = welch(a, rate, nperseg=nperseg)
    _, s_bb = welch(b, rate, nperseg=nperseg)
    coherence = np.abs(s_ab) ** 2 / (s_aa * s_bb + 1e-30)
    out = {}
    for centre in CENTRES:
        band = (freqs >= centre / 2**0.5) & (freqs < centre * 2**0.5)
        keep = band & (coherence >= 0.5)
        if keep.sum() < 4:
            out[str(centre)] = None
            continue
        omega = 2 * np.pi * freqs[keep] / rate
        # scipy's csd is conj(A) * B: its phase is that of B / A, -omega * delay.
        phase = np.unwrap(np.angle(s_ab[keep]))
        slope = np.polyfit(omega, phase, 1, w=np.abs(s_ab[keep]))[0]
        out[str(centre)] = float(-slope)
    return out


def measure(name: str, in1: np.ndarray, in2: np.ndarray, rate: int) -> dict:
    bands = band_delays(in1, in2, rate)
    values = [v for v in bands.values() if v is not None]
    median = statistics.median(values) if values else None
    scale = 48_000 / rate
    return {
        "take": name,
        "rate": rate,
        "bands_samples": bands,
        "median_samples": median,
        "median_samples_48k": None if median is None else median * scale,
        "valid_bands": len(values),
        "spread_samples_48k": statistics.stdev(values) * scale
        if len(values) > 1
        else None,
        "median_us": None if median is None else median / rate * 1e6,
        "peak_dbfs": [
            float(20 * np.log10(np.max(np.abs(c)) + 1e-12)) for c in (in1, in2)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("takes", nargs="*")
    parser.add_argument("--pair", nargs=2, action="append", default=[])
    args = parser.parse_args()
    rows = []
    for path in args.takes:
        audio, rate = sf.read(path, always_2d=True)
        rows.append(measure(Path(path).name, audio[:, 0], audio[:, 1], rate))
    for first, second in args.pair:
        a, rate = sf.read(first)
        b, rate_b = sf.read(second)
        if rate != rate_b or len(a) != len(b):
            raise SystemExit(f"{first} and {second} differ in rate or length")
        rows.append(measure(f"{Path(first).name}+{Path(second).name}", a, b, rate))
    for r in rows:
        m = r["median_samples_48k"]
        print(
            f"{r['take']}: entrée 2 - entrée 1 = "
            + ("--" if m is None else f"{m:+.2f} éch. à 48 kHz")
            + ("" if m is None else f" ({r['median_us']:+.1f} µs)")
            + f", {r['valid_bands']}/5 bandes"
            + (
                ""
                if r["spread_samples_48k"] is None
                else f", écart-type {r['spread_samples_48k']:.2f}"
            )
            + f", crêtes {r['peak_dbfs'][0]:.1f} / {r['peak_dbfs'][1]:.1f} dBFS"
        )
    Path(__file__).with_name("inter_input_delay.json").write_text(
        json.dumps(rows, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
