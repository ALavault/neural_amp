"""Renders of the software capture v2: Neural DSP Archetype: Nolly, Rhythm amp
(demo/capture/v2, research-memory D-0013 and D-0015). Product lane.

Every take shares one input, the source file itself: the renders are offline and Reaper
compensates the latency the plugin declares. Sections are read at the capture rate at
the positions of layout.json, then decimated by 4 to 48 kHz. Knobs gain, bass, middle,
treble, presence, 0 to 10 as on the sheet.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, resample_poly

ROOT = Path(__file__).resolve().parents[3]
SHEET = ROOT / "demo/capture/v2"
DATA = ROOT / "datasets/raw/internal/capture_v2"
SOURCE = DATA / "reamp_v2_192k.wav"
RENDERS = DATA / "nolly"
LAYOUT = json.loads((SHEET / "layout.json").read_text())
SECTIONS = {s["section"]: s for s in LAYOUT["sections"]}
RATE = LAYOUT["sample_rate"]
DOWN = RATE // 48_000
KNOBS = "GBMTP"
# Repeats of take 01, the determinism anchors of the probe reading. Kept out of the
# trained models: bit-identical copies would put the same segments in training and
# validation, and would weigh the centre setting three times.
REPEATS = (20, 39)


def takes() -> list[dict]:
    """Rows of settings.csv: order, setting, knobs (G, B, M, T, P), cab, split, file."""
    with (SHEET / "settings.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    return [
        {
            "order": int(r["order"]),
            "setting": r["setting"],
            "knobs": tuple(float(r[k]) for k in KNOBS),
            "cab": r["cab"],
            "split": r["split"],
            "file": r["file"],
        }
        for r in rows
    ]


def model_takes() -> list[dict]:
    """Takes of the trained models: the train and test rows, without the repeats (the
    cab-on takes are the control rows)."""
    out = [
        t
        for t in takes()
        if t["split"] in ("train", "test") and t["order"] not in REPEATS
    ]
    assert all(t["cab"] == "off" for t in out)
    return out


def span(first: str, last: str | None = None) -> tuple[int, int]:
    """Start and stop, at the capture rate, of the sections first..last."""
    end = SECTIONS[last or first]
    return SECTIONS[first]["start"], end["start"] + end["samples"]


def read(path: Path, start: int, stop: int) -> np.ndarray:
    """Mono samples start..stop of a file at the capture rate."""
    audio, rate = sf.read(path, dtype="float64", start=start, stop=stop, always_2d=True)
    assert rate == RATE and audio.shape == (stop - start, 1), (path, audio.shape)
    return audio[:, 0]


def source(first: str, last: str | None = None, margin: int = 0) -> np.ndarray:
    """Input over first..last at 48 kHz; margin: capture-rate samples on each side."""
    a, b = span(first, last)
    return resample_poly(read(SOURCE, a - margin, b + margin), 1, DOWN)


def output(
    take: dict,
    first: str,
    last: str | None = None,
    renders: Path = RENDERS,
    advance: int = 0,
    margin: int = 0,
) -> np.ndarray:
    """A take's render over first..last at 48 kHz, read `advance` samples (at 48 kHz)
    later: the common latency of demo/nolly_models/predictions.md."""
    a, b = span(first, last)
    shift = DOWN * advance
    y = read(renders / take["file"], a - margin + shift, b + margin + shift)
    return resample_poly(y, 1, DOWN)


def marker_drift(take: dict, renders: Path = RENDERS, search: int = 512) -> dict:
    """The take's output at the end markers against its output at the start markers
    (the same input, at the capture rate): the lag of the largest |correlation|
    within +-search samples, and the RMS of their difference at lag 0 relative to the
    start (dB). A lag of 0: no integer shift found between the two ends within the
    search; it says nothing on the cause of a large difference."""
    s0, s1 = SECTIONS["markers_start"], SECTIONS["markers_end"]
    n = s0["samples"]
    start = read(renders / take["file"], s0["start"], s0["start"] + n)
    end = read(renders / take["file"], s1["start"] - search, s1["start"] + n + search)
    c = correlate(end, start, mode="valid", method="fft")
    lag = int(np.argmax(np.abs(c))) - search
    difference = end[search : search + n] - start
    return {
        "lag": lag,
        "difference_db": float(
            10 * np.log10(np.mean(difference**2) / np.mean(start**2))
        ),
    }


def audio(
    take: dict,
    first: str = "di_test",
    last: str | None = None,
    renders: Path = RENDERS,
    advance: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """(input, output) of a take at 48 kHz."""
    return source(first, last), output(take, first, last, renders, advance)
