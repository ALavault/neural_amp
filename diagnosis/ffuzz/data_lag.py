#!/usr/bin/env python3
"""Input-to-target lag of the F-Fuzz audio itself, no model (DIAGNOSIS_ffuzz.md, H1).

For each of the seven dry files (train and test) and the settings F000 and F100, the
lag maximising the cross-correlation between the dry input and the rendered target,
on the whole file; positive = the target comes later. Written to data_lag.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
PARAM = ROOT / "datasets/raw/external/tone_twist_param/extracted"
DRY = PARAM / "DIGITAL-PARAMETRIC/DRY-with-markers"
WET = PARAM / "MultidrivePedalPro-F-Fuzz"
MAX_LAG = 400


def lag(x: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    n = 1 << (2 * len(x) - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(y, n) * np.conj(np.fft.rfft(x, n)), n)
    window = np.concatenate([corr[-MAX_LAG:], corr[: MAX_LAG + 1]])
    best = int(np.argmax(np.abs(window))) - MAX_LAG
    peak = window[best + MAX_LAG] / (np.linalg.norm(x) * np.linalg.norm(y))
    return best, float(peak)


out = {}
for split in ("trainval", "test"):
    for dry in sorted((DRY / split).glob("*.input.wav")):
        name = dry.name.split(".")[0]
        x, _ = sf.read(dry, dtype="float64")
        for setting in ("F000_V100", "F100_V100"):
            y, _ = sf.read(
                WET / split / setting / f"{setting}.{name}.target.wav", dtype="float64"
            )
            best, peak = lag(x, y)
            out[f"{split}/{name}/{setting}"] = {"lag": best, "normalised_peak": peak}
            print(
                f"{split:8s} {name:14s} {setting}  lag {best:+d}  peak {peak:+.3f}",
                flush=True,
            )
(Path(__file__).with_name("data_lag.json")).write_text(json.dumps(out, indent=1) + "\n")
