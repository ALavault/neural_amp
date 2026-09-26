#!/usr/bin/env python3
"""Find every pass of the VS100R reamp source in a session recording. (Product lane.)

Reaper files hold several passes each, in chronological order. For each recording:
- coarse search at 12 kHz: normalised correlation with the first 32 s of the source
  (pad, calibration blips, silence, pink noise, sweeps), peaks above 0.5;
- exact start at 192 kHz on the opening blips, then the closing blips are searched
  where the source puts them: any offset between the two means lost or added samples;
- gain of the recording against the source on the DI training section, and its polarity.

Writes demo/capture/valvestate/passes.json. Usage:
  product_valvestate_passes.py datasets/amp_capture/Media/02-Looopbck-*.wav
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "demo/capture/valvestate"
LAYOUT = json.loads((SHEET / "layout.json").read_text())
SOURCE = ROOT / LAYOUT["file"]
RATE = LAYOUT["sample_rate"]
DOWN = 16
TEMPLATE_S = 32
SEARCH = 4_000  # samples at RATE around each expected blip
CHUNK, STEP, MAX_LAG = RATE // 2, 2 * RATE, 256  # local lag profile
SECTIONS = {s["section"]: s for s in LAYOUT["sections"]}


def normalised_xcorr(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Normalised correlation of `template` at every start position in `signal`."""
    raw = fftconvolve(signal, template[::-1], mode="valid")
    cumulative = np.concatenate([[0.0], np.cumsum(signal.astype(np.float64) ** 2)])
    energy = cumulative[len(template) :] - cumulative[: -len(template)]
    return raw / np.sqrt(np.maximum(energy, 0) * (template**2).sum() + 1e-30)


def coarse_starts(recording: np.ndarray, source: np.ndarray) -> list[tuple[int, float]]:
    rec = resample_poly(recording, 1, DOWN)
    template = resample_poly(source[: TEMPLATE_S * RATE], 1, DOWN)
    ncc = normalised_xcorr(rec, template)
    found, gap = [], int(30 * RATE / DOWN)
    for index in np.argsort(-np.abs(ncc)):
        if abs(ncc[index]) < 0.5:
            break
        if all(abs(index - f) > gap for f, _ in found):
            found.append((int(index), float(ncc[index])))
    return sorted((i * DOWN, c) for i, c in found)


def exact(recording: np.ndarray, source: np.ndarray, section: str, guess: int):
    """Position in the recording of the source's `section` start, near `guess`."""
    s = SECTIONS[section]
    template = source[s["start"] : s["start"] + s["samples"]]
    lo = max(guess + s["start"] - SEARCH, 0)
    window = recording[lo : lo + s["samples"] + 2 * SEARCH]
    if len(window) < len(template):
        return None, 0.0
    ncc = normalised_xcorr(window, template)
    best = int(np.argmax(np.abs(ncc)))
    return lo + best - s["start"], float(ncc[best])


def lag_profile(recording: np.ndarray, source: np.ndarray, start: int) -> list:
    """Local lag, in samples, of the recording against the source every 2 s."""
    first = SECTIONS["markers_start"]["start"]
    last = SECTIONS["markers_end"]["start"]
    out = []
    for t in range(first, last - CHUNK, STEP):
        src = source[t : t + CHUNK]
        if np.sqrt(np.mean(src**2)) < 10 ** (-50 / 20):
            continue
        rec = recording[start + t - MAX_LAG : start + t + CHUNK + MAX_LAG]
        if len(rec) < CHUNK + 2 * MAX_LAG:
            break
        ncc = normalised_xcorr(rec, src)
        out.append((round(t / RATE, 1), int(np.argmax(np.abs(ncc))) - MAX_LAG))
    return out


def main() -> None:
    source, rate = sf.read(SOURCE, dtype="float32")
    assert rate == RATE
    passes = []
    for path in sorted(Path(p) for p in sys.argv[1:]):
        recording, rate = sf.read(path, dtype="float32")
        assert rate == RATE and recording.ndim == 1, path
        for coarse, score in coarse_starts(recording, source):
            start, ncc_start = exact(recording, source, "markers_start", coarse)
            if abs(ncc_start) < 0.5:
                continue
            end, ncc_end = exact(recording, source, "markers_end", start)
            complete = end is not None and abs(ncc_end) > 0.5
            row = {
                "file": path.name,
                "start": start,
                "start_s": round(start / RATE, 3),
                "coarse_ncc": round(score, 3),
                "ncc_start": round(ncc_start, 4),
                "complete": complete,
            }
            if complete:
                di = SECTIONS["di_train"]
                src = source[di["start"] : di["start"] + di["samples"]]
                rec = recording[
                    start + di["start"] : start + di["start"] + di["samples"]
                ]
                gain = float(np.dot(rec, src) / np.dot(src, src))
                row |= {
                    "lags": lag_profile(recording, source, start),
                    "ncc_end": round(ncc_end, 4),
                    "drift_samples": int(end - start),
                    "gain_db": round(20 * np.log10(abs(gain)), 2),
                    "polarity": int(np.sign(gain)),
                }
            passes.append(row)
            print({k: v for k, v in row.items() if k != "lags"}, flush=True)
    flag_slips(passes)
    map_settings(passes)
    (SHEET / "passes.json").write_text(json.dumps(passes, indent=1) + "\n")


def map_settings(passes: list[dict]) -> None:
    """Takes follow settings.csv in chronological order: file stamps sort in time."""
    with (SHEET / "settings.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    complete = sorted((p for p in passes if p["complete"]), key=lambda p: p["file"])
    if len(complete) != len(rows):
        print(
            f"{len(complete)} passes pour {len(rows)} reglages : pas d'appariement"
        )
        return
    for take, row in zip(complete, rows, strict=True):
        take |= {
            "order": int(row["order"]),
            "setting": row["setting"],
            "take_file": row["file"],
        }


def flag_slips(passes: list[dict]) -> None:
    """Lag against the median profile of all passes, in units of 24 samples.

    The local lag also carries the reamp box's group delay, which depends on the
    content and is the same in every pass: the median over passes removes it.
    """
    complete = [p for p in passes if p["complete"]]
    times = sorted({t for p in complete for t, _ in p["lags"]})
    table = {t: [dict(p["lags"]).get(t) for p in complete] for t in times}
    median = {
        t: float(np.median([v for v in vs if v is not None])) for t, vs in table.items()
    }
    for p in complete:
        deviation = [(t, lag - median[t]) for t, lag in p["lags"]]
        p["microframes"] = [(t, round(d / 24)) for t, d in deviation]
        p["slipped_s"] = [t for t, units in p["microframes"] if units != 0]
        print(p["file"], "hors median :", len(p["slipped_s"]), "points", flush=True)


if __name__ == "__main__":
    main()
