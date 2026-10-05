#!/usr/bin/env python3
"""Per-arm summary of vs100r_hammerstein.json (NOTE.md tables).

For each level and arm: |D_k model - D_k device| in dB (k = 1..5), median over takes
and runs per third-octave band, then median over the bands of 200 Hz - 8 kHz (NOTE.md
table); and the signed D2 gap, median over takes, runs and bands. Per run at -20 dBFS:
the signed D2 gap per take (take profile). Read-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LEVELS = ("-20", "-10", "-3")


def main() -> None:
    data = json.loads((HERE / "vs100r_hammerstein.json").read_text())
    centres = np.array(data["centres"])
    band = (centres >= 200) & (centres <= 8000)
    takes = data["takes"]
    arms = sorted(
        {m["arm"] for t in takes.values() for m in t["-20"]["models"].values()}
    )
    for level in LEVELS:
        for arm in arms:
            rows = [
                np.array(m["d_errors"]["level_db"])[:, band]
                for t in takes.values()
                for m in t[level]["models"].values()
                if m["arm"] == arm
            ]
            stack = np.stack(rows)  # (take x run, order, band)
            absolute = np.nanmedian(np.nanmedian(np.abs(stack), axis=0), axis=1)
            signed = np.nanmedian(stack[:, 1])
            print(
                f"{level:>4} dBFS {arm:5s} |D1..D5| "
                + " ".join(f"{v:5.2f}" for v in absolute)
                + f"  D2 signé {signed:+.1f}"
            )
    print("D2 signé par prise à -20 dBFS (médiane sur bandes)")
    print(" " * 30 + " ".join(f"{t:>6}" for t in takes))
    runs = sorted({r for t in takes.values() for r in t["-20"]["models"]})
    for run in runs:
        gaps = [
            np.nanmedian(
                np.array(t["-20"]["models"][run]["d_errors"]["level_db"])[1, band]
            )
            for t in takes.values()
        ]
        print(f"{run:30s}" + " ".join(f"{g:+6.1f}" for g in gaps))


if __name__ == "__main__":
    main()
