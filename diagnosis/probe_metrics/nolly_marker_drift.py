#!/usr/bin/env python3
"""Post hoc check of the Nolly renders, after the frozen reading (NOLLY_READING.md,
section « Écarts »). CPU, about a minute.

    nolly_marker_drift.py [RENDERS]

The two marker sections carry the same input. For every take: the output at the end
markers against the output at the start markers (fssr_nam.product.nolly.marker_drift:
lag of the largest |correlation| within +-512 samples, RMS of the difference at lag 0
relative to the start); the frequency of the difference's spectral peak and the share
of its energy below 2 kHz (Welch, 8 192 points, 192 kHz); the RMS of the silence
section. A lag of 0: no integer shift found within +-512 samples; a large difference:
the plugin answered the same input, after a different history, differently (cause not
identified). Writes nolly_marker_drift.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

from fssr_nam.product import nolly

HERE = Path(__file__).resolve().parent


def take_drift(take: dict, renders: Path) -> dict:
    s0, s1, quiet = (
        nolly.SECTIONS[n] for n in ("markers_start", "markers_end", "silence")
    )
    n = s0["samples"]
    path = renders / take["file"]
    start = nolly.read(path, s0["start"], s0["start"] + n)
    difference = nolly.read(path, s1["start"], s1["start"] + n) - start
    f, p = welch(difference, fs=nolly.RATE, nperseg=8192)
    silence = nolly.read(path, quiet["start"], quiet["start"] + quiet["samples"])
    return {
        "setting": take["setting"],
        **nolly.marker_drift(take, renders),
        "difference_peak_hz": float(f[np.argmax(p)]),
        "difference_share_below_2k": float(p[f < 2_000].sum() / p.sum()),
        "silence_rms_dbfs": float(10 * np.log10(np.mean(silence**2))),
    }


def main() -> None:
    renders = Path(sys.argv[1]) if len(sys.argv) > 1 else nolly.RENDERS
    result = {str(t["order"]): take_drift(t, renders) for t in nolly.takes()}
    (HERE / "nolly_marker_drift.json").write_text(json.dumps(result, indent=1) + "\n")
    for order, r in result.items():
        print(
            f"{order:>2} {r['setting']:<32} lag {r['lag']:+d}"
            f"  difference {r['difference_db']:6.1f} dB"
            f"  peak {r['difference_peak_hz']:7.0f} Hz",
            flush=True,
        )


if __name__ == "__main__":
    main()
