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
  alone as reference (recovery after a loud event: bias shift, blocking);
- two tones (added 2026-10-07, after the existing sections, which keep their positions),
  2 s each at combined peaks of -20, -10 and -3 dBFS, as the sweeps: 100 Hz + 2 kHz at
  4:1 (modulation of a high tone by a low one: cone excursion, power supply), and
  1000 + 1100 Hz at 1:1 (difference tone at 100 Hz, even order; 900 and 1200 Hz, odd
  order). Every frequency fits a whole number of cycles in 1 s at 192 kHz.

A second, short source (about 1 min) repeats the sweeps, the two tones and the plucks
for the cold / hot / warm takes of the session sheet: same markers and pre-roll, so the
first capture's alignment applies.

Writes datasets/raw/internal/valvestate_vs100r_mic/reamp_valvestate_mic_192k.wav and
reamp_valvestate_mic_probe_192k.wav (not committed: ToneTwist DI material, CC BY-NC),
demo/capture/valvestate_mic/layout.json and layout_probe.json. The processing scripts of
the first capture (product_valvestate_passes.py, _qc.py, _dataset.py,
product_vs100r_eval.py) read its layout and paths: takes recorded with these sources
need them pointed at these layouts (closing markers no longer at 212.0 s).
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
SWEEP_LEVELS_DBFS = (-20, -10, -3)  # as in the first source
# (name, low Hz, high Hz, low share of the combined peak)
TWO_TONES = (("mod", 100, 2000, 0.8), ("diff", 1000, 1100, 0.5))


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


def plucks() -> np.ndarray:
    parts = []
    for hz in (110, 440):
        for peak in PLUCK_AMPLITUDES_DBFS:
            parts += [pluck(hz, peak), silence(0.4)]
    return np.concatenate(parts)


def two_tone_sections() -> list[tuple[str, np.ndarray]]:
    """Two summed tones, 2 s; the shares of the combined peak add up to 1."""
    sections = []
    for name, low, high, share in TWO_TONES:
        for level in SWEEP_LEVELS_DBFS:
            x = tone(low, 2.0, level + 20 * np.log10(share))
            x += tone(high, 2.0, level + 20 * np.log10(1 - share))
            sections += [
                ("gap", silence(session.GAP_S)),
                (f"two_tone_{name}_{level}dBFS", x),
            ]
    return sections


def extra_sections() -> list[tuple[str, np.ndarray]]:
    x = excerpt()
    gap = silence(session.GAP_S)
    sections = [("gap", silence(1.0))]
    sections += [("polarity_plus", x), ("gap", gap), ("polarity_minus", -x)]
    for level in LEVELS_DB:
        sections += [("gap", gap), (f"level_{level:+d}dB", x * 10 ** (level / 20))]
    sections += [("gap", gap), ("bursts_plucks", plucks())]
    probe = tone(1000, 0.1, -30)
    recovery = [silence(2.0), probe]
    for ms in GAPS_MS:
        recovery += [silence(2.0), tone(110, 0.3, -3), silence(ms / 1000), probe]
    sections += [("gap", gap), ("bursts_recovery", np.concatenate(recovery))]
    return sections + two_tone_sections()


def assemble(sections: list[tuple[str, np.ndarray]]) -> tuple[np.ndarray, dict]:
    parts, out, position = [], [], 0
    for name, x in sections:
        if name != "gap":
            out.append({"section": name, "start": position, "samples": len(x)})
        parts.append(x)
        position += len(x)
    signal = np.concatenate(parts)
    assert np.max(np.abs(signal)) < 1.0
    return signal.astype(np.float32), {"sample_rate": RATE, "sections": out}


def build_probe() -> tuple[np.ndarray, dict]:
    """The short source of the cold / hot / warm takes: same pre-roll and markers as
    the main source, a short silence, the sweeps, the two tones, the plucks."""
    gap = silence(session.GAP_S)
    sections = [
        ("pad_start", np.zeros(session.PAD_START_S * RATE)),
        ("markers_start", session.markers()),
        ("silence", silence(2.0)),
    ]
    for level in SWEEP_LEVELS_DBFS:
        sections += [("gap", gap), (f"sweep_{level}dBFS", session.sweep(5, level))]
    sections += two_tone_sections()
    sections += [("gap", gap), ("bursts_plucks", plucks()), ("gap", gap)]
    sections += [
        ("markers_end", session.markers()),
        ("pad_end", np.zeros(session.PAD_END_S * RATE)),
    ]
    return assemble(sections)


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


def write(signal: np.ndarray, layout: dict, stem: str, sheet: str, extra: dict) -> dict:
    detected = session.self_check(signal)
    assert detected == 777, f"self-check found the blip at {detected}, not 777"
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIO_DIR / f"{stem}_{RATE // 1000}k.wav"
    sf.write(path, signal, RATE, subtype="PCM_24")
    layout |= {
        "file": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seconds": round(len(signal) / RATE, 3),
    } | extra
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    (SHEET_DIR / sheet).write_text(json.dumps(layout, indent=1) + "\n")
    print(
        f"{path.name}: {layout['seconds']} s, peak {np.abs(signal).max():.3f}, "
        f"first blip found at sample {detected}"
    )
    return layout


def main() -> None:
    signal, layout = build()
    worst = check_against_first_capture(signal, layout)
    print(f"first sections identical to the first source (max diff {worst:.1e})")
    write(
        signal,
        layout,
        "reamp_valvestate_mic",
        "layout.json",
        {
            "di_source": "ToneTwist AFx DRY-with-markers (Zenodo 10455730, "
            "CC BY-NC 4.0)",
            "excerpt": {
                "file": EXCERPT[0],
                "start_s": EXCERPT[1],
                "seconds": EXCERPT[2],
            },
        },
    )
    write(*build_probe(), "reamp_valvestate_mic_probe", "layout_probe.json", {})


if __name__ == "__main__":
    main()
