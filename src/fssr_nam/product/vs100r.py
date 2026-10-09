"""VS100R capture of 2026-09-26: take sheet and section audio, numpy only.

Each take is a pair of tracks recorded together: the loopback of the played source
(input 1) and the FX loop send (input 2). Sections follow `layout.json` (192 kHz).
Moved out of scripts/product_vs100r_eval.py so analysis code can load takes without
the training stack.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
SHEET = ROOT / "demo/capture/valvestate"
MEDIA = ROOT / "datasets/amp_capture/Media"
LAYOUT = json.loads((SHEET / "layout.json").read_text())
RATE = LAYOUT["sample_rate"]


def section(name: str) -> dict:
    """Layout entry of a section: start and length in samples at RATE."""
    return next(s for s in LAYOUT["sections"] if s["section"] == name)


def sheet() -> list[dict]:
    """Every take of settings.csv with its knobs, located pass and quality control."""
    passes = {p["order"]: p for p in json.loads((SHEET / "passes.json").read_text())}
    qc = {r["order"]: r for r in json.loads((SHEET / "qc.json").read_text())}
    with (SHEET / "settings.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    out = []
    for r in rows:
        order = int(r["order"])
        out.append(
            {
                "order": order,
                "channel": r["channel"],
                "power_dimension": r["power_dimension"],
                "split": r["split"],
                "setting": r["setting"],
                "knobs": {k: float(r[k]) if r[k] else None for k in "GCBMT"},
                "low_signal": qc[order]["low_signal"],
                "slipped": bool(passes[order]["slipped_s"]),
                "pass": passes[order],
                "spikes": qc[order]["spikes_wav_sample"],
            }
        )
    return out


def audio(take: dict, name: str = "di_test") -> tuple[np.ndarray, np.ndarray]:
    """(loopback, FX send) of a section of a take, float64 at RATE, spikes repaired."""
    part = section(name)
    p = take["pass"]
    a = p["start"] + part["start"]
    b = a + part["samples"]
    stamp = p["file"].removeprefix("02-Looopbck-")
    loop = sf.read(MEDIA / p["file"], dtype="float64", start=a, stop=b)[0]
    fx = sf.read(MEDIA / f"03-FX Out-{stamp}", dtype="float64", start=a, stop=b)[0]
    for i in take["spikes"]:
        if a < i < b - 1:
            fx[i - a] = 0.5 * (fx[i - a - 1] + fx[i - a + 1])
    return loop, fx
