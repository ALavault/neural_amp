#!/usr/bin/env python3
"""Take against take on the VS100R: repeatability and knob effects. (Product lane, CPU.)

Every take played the same source, so two takes can be compared sample by sample on the
FX loop send, in 2 s windows where each take holds one microframe offset against the
source (passes.json), shifted by that offset from its own pass start. For a pair (a, b):
  D = sum (y_a - y_b)^2 / sum y_a^2 over the DI sections, 48 kHz band (decimated by 4).
Two takes of one setting give the repeatability, twice the noise ESR floor if the
difference were pure independent noise (drift adds to it). A knob that does not reach
the FX send gives a pair at that same level; a knob that does gives much more.

Writes demo/capture/valvestate/repeat.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "demo/capture/valvestate"
MEDIA = ROOT / "datasets/amp_capture/Media"
LAYOUT = json.loads((SHEET / "layout.json").read_text())
SECTIONS = {s["section"]: s for s in LAYOUT["sections"]}
RATE = LAYOUT["sample_rate"]
FIRST, LAST = "di_train", "di_test"
# (a, b, what the pair tests): same setting, then one knob at a time from the centre.
PAIRS = [
    (1, 39, "centre contre centre (debut, fin de seance)"),
    (1, 20, "centre contre centre (debut, milieu)"),
    (1, 12, "aigus 5 -> 0"),
    (1, 23, "aigus 5 -> 10"),
    (1, 13, "graves 5 -> 0"),
    (1, 4, "graves 5 -> 10"),
    (1, 15, "mediums 5 -> 0"),
    (1, 34, "mediums 5 -> 10"),
    (1, 31, "contour 5 -> 0"),
    (1, 37, "contour 5 -> 10"),
    (1, 26, "gain 5 -> 2,5"),
    (1, 38, "gain 5 -> 7,5"),
]


def fx(order: int, passes: dict) -> tuple[np.ndarray, int]:
    p = passes[order]
    a = p["start"] + SECTIONS[FIRST]["start"]
    b = p["start"] + SECTIONS[LAST]["start"] + SECTIONS[LAST]["samples"]
    stamp = p["file"].removeprefix("02-Looopbck-")
    return sf.read(MEDIA / f"03-FX Out-{stamp}", dtype="float64", start=a, stop=b)[0], a


def steady_offsets(order: int, passes: dict) -> dict[float, int]:
    """Offset in samples on each 2 s window whose two ends agree, by source time."""
    units = dict(passes[order]["microframes"])
    return {t: 24 * u for t, u in units.items() if units.get(t + 2) == u}


def main() -> None:
    passes = {p["order"]: p for p in json.loads((SHEET / "passes.json").read_text())}
    first = SECTIONS[FIRST]["start"] / RATE
    last = (SECTIONS[LAST]["start"] + SECTIONS[LAST]["samples"]) / RATE
    rows = []
    for a, b, label in PAIRS:
        ya, _ = fx(a, passes)
        yb, _ = fx(b, passes)
        offsets_a, offsets_b = steady_offsets(a, passes), steady_offsets(b, passes)
        diff = energy = 0.0
        used = 0
        for t in sorted(set(offsets_a) & set(offsets_b)):
            if t < first + 0.1 or t + 2.1 > last:
                continue
            lo = round((t - first) * RATE)
            hi = lo + 2 * RATE
            sa, sb = offsets_a[t], offsets_b[t]
            wa = resample_poly(ya[lo + sa : hi + sa], 1, 4)[480:-480]  # filter edges
            wb = resample_poly(yb[lo + sb : hi + sb], 1, 4)[480:-480]
            diff += float(((wa - wb) ** 2).sum())
            energy += float((wa**2).sum())
            used += 1
        row = {"a": a, "b": b, "pair": label, "windows_2s": used, "D": diff / energy}
        rows.append(row)
        print(
            f"{a:>2}-{b:<2} {label:44s} D = {row['D']:.4f}"
            f" ({10 * np.log10(row['D']):5.1f} dB) sur {used} fenetres",
            flush=True,
        )
    (SHEET / "repeat.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
