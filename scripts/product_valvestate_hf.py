#!/usr/bin/env python3
"""What does 48 kHz lose on the VS100R? (Product lane, CPU, descriptive.)

The source DI comes from 48 kHz recordings and the test signals stop at 20 kHz: whatever
the FX loop send carries above that was made by the amp (harmonics, intermodulation) or
is noise. Per take, on the DI sections at 192 kHz, share of the FX power in each band,
with the take's own noise floor (its 5 s silence section) subtracted band by band. The
share above 24 kHz is what a 48 kHz capture discards, and roughly what a nonlinear model
running at 48 kHz would fold back into the audio band if it reproduced those products.

Writes demo/capture/valvestate/hf.json.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "demo/capture/valvestate"
MEDIA = ROOT / "datasets/amp_capture/Media"
LAYOUT = json.loads((SHEET / "layout.json").read_text())
SECTIONS = {s["section"]: s for s in LAYOUT["sections"]}
RATE = LAYOUT["sample_rate"]
EDGES = (20_000, 24_000, 48_000)


def psd(path: Path, start: int, first: str, last: str) -> tuple[np.ndarray, np.ndarray]:
    a = start + SECTIONS[first]["start"] + RATE // 10
    b = start + SECTIONS[last]["start"] + SECTIONS[last]["samples"] - RATE // 10
    x = sf.read(path, dtype="float64", start=a, stop=b)[0]
    return welch(x, RATE, nperseg=8192)


def main() -> None:
    passes = sorted(
        json.loads((SHEET / "passes.json").read_text()), key=lambda p: p["order"]
    )
    with (SHEET / "settings.csv").open() as stream:
        settings = {int(r["order"]): r for r in csv.DictReader(stream)}
    rows = []
    for p in passes:
        stamp = p["file"].removeprefix("02-Looopbck-")
        out = {}
        for track, path in (
            ("fx", MEDIA / f"03-FX Out-{stamp}"),
            ("loop", MEDIA / p["file"]),
        ):
            freqs, signal = psd(path, p["start"], "di_train", "di_test")
            _, noise = psd(path, p["start"], "silence", "silence")
            clean = np.maximum(signal - noise, 0.0)
            total = clean.sum()
            for edge in EDGES:
                share = clean[freqs >= edge].sum() / total
                out[f"{track}_above_{edge // 1000}k_db"] = round(
                    float(10 * np.log10(max(share, 1e-12))), 1
                )
        setting = settings[p["order"]]
        row = {
            "order": p["order"],
            "setting": p["setting"],
            "channel": setting["channel"],
            "gain": setting["G"],
            "power_dimension": setting["power_dimension"],
            **out,
        }
        rows.append(row)
        print(
            f"{row['order']:>3} {row['setting']:32s}"
            + "".join(
                f"  >{e // 1000}k {row[f'fx_above_{e // 1000}k_db']:6.1f}"
                for e in EDGES
            )
            + f"  | bouclage >24k {row['loop_above_24k_db']:6.1f} dB",
            flush=True,
        )
    (SHEET / "hf.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
