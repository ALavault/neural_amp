#!/usr/bin/env python3
"""Reamp source for the mic'd VS100R session. Product lane.

The source of scripts/product_valvestate_session.py, every section unchanged and at the
same sample position (checked against the existing file), then, before the closing
markers, three groups of sections for offline tests (not training by default):

- polarity: a 10 s DI guitar excerpt never used by the first capture (idmt-gtr4-ib,
  190-200 s, from the file whose 60-75 s make di_test), then the same excerpt inverted:
  the device's even (asymmetric) part is (y(x) + y(-x)) / 2;
- levels: the same excerpt at -12, -6 and +6 dB;
- bursts: decaying 110 Hz and 440 Hz tones ("plucks", 2 ms attack) at five envelope
  amplitudes, then pairs of
  a loud 110 Hz burst and a quiet 1 kHz probe after a gap of 5 ms to 2 s, with the probe
  alone as reference (recovery after a loud event: bias shift, blocking).

Writes datasets/raw/internal/valvestate_vs100r_mic/reamp_valvestate_mic_192k.wav (not
committed: ToneTwist DI material, CC BY-NC) and demo/capture/valvestate_mic/layout.json.
The processing scripts of the first capture (product_valvestate_passes.py, _qc.py,
_dataset.py, product_vs100r_eval.py) read its layout and paths: takes recorded with this
source need them pointed at this layout (closing markers at 301.2 s, not 212.0 s).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
import product_valvestate_session as session

ROOT = session.ROOT
RATE = session.RATE
AUDIO_DIR = ROOT / "datasets/raw/internal/valvestate_vs100r_mic"
SHEET_DIR = ROOT / "demo/capture/valvestate_mic"
EXCERPT = ("idmt-gtr4-ib", 190, 10)  # file, start (s), length (s)
LEVELS_DB = (-12, -6, 6)
PLUCK_AMPLITUDES_DBFS = (-30, -20, -12, -6, -3)
GAPS_MS = (5, 20, 50, 100, 200, 500, 1000, 2000)


def excerpt() -> np.ndarray:
    name, start, seconds = EXCERPT
    x, rate = sf.read(session.DRY / "test" / f"{name}.input.wav", dtype="float64")
    segment = x[start * rate : (start + seconds) * rate]
    return session.fade(resample_poly(segment, RATE // rate, 1))


def tone(hz: float, seconds: float, peak_dbfs: float) -> np.ndarray:
    """Steady sine with 2 ms raised-cosine ramps."""
    t = np.arange(int(seconds * RATE)) / RATE
    x = 10 ** (peak_dbfs / 20) * np.sin(2 * np.pi * hz * t)
    n = int(0.002 * RATE)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


def pluck(hz: float, amplitude_dbfs: float) -> np.ndarray:
    """2 ms linear attack, exponential decay (150 ms time constant), 0.6 s, then a 2 ms
    raised-cosine release; amplitude is the envelope's peak."""
    t = np.arange(int(0.6 * RATE)) / RATE
    envelope = np.minimum(t / 0.002, 1.0) * np.exp(-np.maximum(t - 0.002, 0) / 0.15)
    n = int(0.002 * RATE)
    envelope[-n:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(n) / n)
    return 10 ** (amplitude_dbfs / 20) * envelope * np.sin(2 * np.pi * hz * t)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * RATE))


def extra_sections() -> list[tuple[str, np.ndarray]]:
    x = excerpt()
    gap = silence(session.GAP_S)
    sections = [("gap", silence(1.0))]
    sections += [("polarity_plus", x), ("gap", gap), ("polarity_minus", -x)]
    for level in LEVELS_DB:
        sections += [("gap", gap), (f"level_{level:+d}dB", x * 10 ** (level / 20))]
    plucks = []
    for hz in (110, 440):
        for peak in PLUCK_AMPLITUDES_DBFS:
            plucks += [pluck(hz, peak), silence(0.4)]
    sections += [("gap", gap), ("bursts_plucks", np.concatenate(plucks))]
    probe = tone(1000, 0.1, -30)
    recovery = [silence(2.0), probe]
    for ms in GAPS_MS:
        recovery += [silence(2.0), tone(110, 0.3, -3), silence(ms / 1000), probe]
    sections += [("gap", gap), ("bursts_recovery", np.concatenate(recovery))]
    return sections


def build() -> tuple[np.ndarray, dict]:
    base, layout = session.reamp()
    sections = {s["section"]: s for s in layout["sections"]}
    head = base[: sections["markers_end"]["start"]].astype(np.float64)
    parts = [head]
    out = [
        s for s in layout["sections"] if s["section"] not in ("markers_end", "pad_end")
    ]
    position = len(head)
    tail = [
        *extra_sections(),
        ("markers_end", session.markers()),
        ("pad_end", np.zeros(session.PAD_END_S * RATE)),
    ]
    for name, x in tail:
        if name != "gap":
            out.append({"section": name, "start": position, "samples": len(x)})
        parts.append(x)
        position += len(x)
    signal = np.concatenate(parts)
    assert np.max(np.abs(signal)) < 1.0
    return signal.astype(np.float32), {"sample_rate": RATE, "sections": out}


def check_against_first_capture(signal: np.ndarray, layout: dict) -> float:
    """Every section of the first source, read back from its 24-bit file, must be at
    the same position with the same samples (to 24-bit quantisation)."""
    old_layout = json.loads((session.SHEET_DIR / "layout.json").read_text())
    old = sf.read(
        session.AUDIO_DIR / f"reamp_valvestate_{RATE // 1000}k.wav", dtype="float64"
    )[0]
    new = {s["section"]: s for s in layout["sections"]}
    worst = 0.0
    for s in old_layout["sections"]:
        if s["section"] in ("markers_end", "pad_end"):
            continue
        assert new[s["section"]]["start"] == s["start"], s["section"]
        assert new[s["section"]]["samples"] == s["samples"], s["section"]
        a, n = s["start"], s["samples"]
        worst = max(worst, float(np.abs(signal[a : a + n] - old[a : a + n]).max()))
    assert worst <= 2.0**-23, worst
    return worst


def main() -> None:
    signal, layout = build()
    worst = check_against_first_capture(signal, layout)
    detected = session.self_check(signal)
    assert detected == 777, f"self-check found the blip at {detected}, not 777"
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIO_DIR / f"reamp_valvestate_mic_{RATE // 1000}k.wav"
    sf.write(path, signal, RATE, subtype="PCM_24")
    layout |= {
        "file": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seconds": round(len(signal) / RATE, 3),
        "di_source": "ToneTwist AFx DRY-with-markers (Zenodo 10455730, CC BY-NC 4.0)",
        "excerpt": {"file": EXCERPT[0], "start_s": EXCERPT[1], "seconds": EXCERPT[2]},
    }
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    (SHEET_DIR / "layout.json").write_text(json.dumps(layout, indent=1) + "\n")
    print(
        f"{path.name}: {layout['seconds']} s, peak {np.abs(signal).max():.3f}, "
        f"first sections identical to the first source (max diff {worst:.1e}), "
        f"first blip found at sample {detected}"
    )


if __name__ == "__main__":
    main()
