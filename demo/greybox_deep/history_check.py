#!/usr/bin/env python3
"""Validation of the history-conditioned gain (HISTORY_MEASURE.md), then the devices'
own history slopes. Read-only, CPU.

Synthetic chains on the real DI loops (di_test of the 8 VS100R test takes, 48 kHz):
S0 high-pass 100 Hz -> tanh(10 x) -> low-pass 5 kHz (first-order filters); S1a/S1b
drive divided by 1 + k e (e: envelope of |x|, 100 ms; median 3 / 6 dB); S2 S0 then a
150 ms cabinet FIR; S3 tanh(10 x + b e) - tanh(b e), median of b e = 0.3. Devices:
VS100R FX send (same takes), JVM410H OD1 speaker output (9 test settings, 44.1 kHz),
each beside S0 on the same input. Writes demo/greybox_deep/history_check.json.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, oaconvolve, resample_poly, sosfilt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "demo/train_length"))
import eval_train_length as tl  # noqa: E402

from fssr_nam.product.history import HISTORY_S, history_slopes  # noqa: E402

JVM = (
    ROOT / "datasets/raw/external/tone_twist_param/extracted"
    "/Marshall-JVM410H-ChOD1-trimmed"
)
DRIVE = 10.0


def envelope(x: np.ndarray, fs: float, tau: float = 0.1) -> np.ndarray:
    a = math.exp(-1 / (tau * fs))
    return lfilter([1 - a], [1, -a], np.abs(x))


def loud_median(e: np.ndarray) -> float:
    """Median of the envelope where it is within 40 dB of its maximum."""
    return float(np.median(e[e >= 0.01 * e.max()]))


def cab(fs: float) -> np.ndarray:
    n = int(0.15 * fs)
    rng = np.random.default_rng(7)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / (0.02 * fs))
    ir = sosfilt(butter(4, [80, 5_000], "bandpass", fs=fs, output="sos"), ir)
    return ir / np.sqrt(np.sum(ir**2))


def chains(x: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    high = butter(1, 100, "high", fs=fs)
    low = butter(1, 5_000, "low", fs=fs)
    u = lfilter(*high, x)
    e = envelope(x, fs)
    m = loud_median(e)
    out = {"S0": lfilter(*low, np.tanh(DRIVE * u))}
    for name, db in (("S1a", 3.0), ("S1b", 6.0)):
        k = (10 ** (db / 20) - 1) / m
        out[name] = lfilter(*low, np.tanh(DRIVE * u / (1 + k * e)))
    out["S2"] = oaconvolve(out["S0"], cab(fs))[: len(x)]
    b = 0.3 / m
    out["S3"] = lfilter(*low, np.tanh(DRIVE * u + b * e) - np.tanh(b * e))
    return out


def gammas(x: np.ndarray, y: np.ndarray, fs: float) -> dict:
    slopes = history_slopes(x, y, fs)
    return {
        str(t): {k: round(v, 5) for k, v in slopes[t].items()} for t in HISTORY_S
    } | {"frames": slopes["frames"]}


def summary(cases: list[dict]) -> dict:
    out = {}
    for t in HISTORY_S:
        values = [c[str(t)]["gamma"] for c in cases]
        out[str(t)] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    return out


def main() -> None:
    takes = [t for t in tl.ev.takes() if not t["low_signal"] and t["split"] == "test"]
    systems: dict[str, list] = {}
    for t in takes:
        loop, fx = (resample_poly(s, 1, 4) for s in tl.ev.audio(t))
        for name, y in chains(loop, 48_000).items():
            systems.setdefault(name, []).append(gammas(loop, y, 48_000))
        systems.setdefault("VS100R", []).append(gammas(loop, fx, 48_000))
        print(t["order"], flush=True)
    x, fs = sf.read(JVM / "DRY/test/test.input.wav", dtype="float64")
    systems["S0_jvm_input"] = [gammas(x, chains(x, fs)["S0"], fs)]
    for folder in sorted((JVM / "SpeakerOut/test").iterdir()):
        y = sf.read(next(folder.glob("*.wav")), dtype="float64")[0]
        systems.setdefault("JVM410H", []).append(gammas(x, y, fs))
    result = {
        "takes": [t["order"] for t in takes],
        "summary": {k: summary(v) for k, v in systems.items()},
        "cases": systems,
    }
    (HERE / "history_check.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["summary"], indent=1), flush=True)


if __name__ == "__main__":
    main()
