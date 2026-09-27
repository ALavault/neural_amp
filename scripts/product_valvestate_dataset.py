#!/usr/bin/env python3
"""Build the exploratory VS100R parametric dataset. (Product lane, CPU.)

Input = each take's own loopback (the signal the amp received), target = its FX loop
send: the two tracks share one USB stream, so the pair stays in step through the
microframe slips (demo/capture/valvestate/QC.md). OD2, Power Dimension off, knobs
gain, contour, bass, middle, treble. Decimated 192 -> 48 kHz, written as float WAV.

Layout read by nablafx's ParametricPluginDataset (params parsed from the target name):
  dry/trainval/takeNN.input.wav     di_train + di_validation of the train settings
  wet/trainval/<setting>/<setting>.takeNN.target.wav
  dry/test, wet/test                di_test of the 8 test settings (never trained on)
  dry/test_seen, wet/test_seen      di_test of the train settings (unseen audio)
Isolated spikes found by product_valvestate_qc.py are replaced by the mean of their two
neighbours before decimation.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "demo/capture/valvestate"
MEDIA = ROOT / "datasets/amp_capture/Media"
OUT = ROOT / "datasets/raw/internal/valvestate_vs100r/explore_48k"
LAYOUT = json.loads((SHEET / "layout.json").read_text())
SECTIONS = {s["section"]: s for s in LAYOUT["sections"]}
RATE, DOWN = LAYOUT["sample_rate"], 4


def span(first: str, last: str) -> tuple[int, int]:
    return SECTIONS[first]["start"], SECTIONS[last]["start"] + SECTIONS[last]["samples"]


def write(kind: str, split: str, take: int, setting: str, audio: np.ndarray) -> None:
    if kind == "dry":
        path = OUT / "dry" / split / f"take{take:02d}.input.wav"
    else:
        path = OUT / "wet" / split / setting / f"{setting}.take{take:02d}.target.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), RATE // DOWN, subtype="FLOAT")


def main() -> None:
    passes = {p["order"]: p for p in json.loads((SHEET / "passes.json").read_text())}
    qc = {r["order"]: r for r in json.loads((SHEET / "qc.json").read_text())}
    with (SHEET / "settings.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream)]
    rows = [r for r in rows if r["channel"] == "OD2" and r["power_dimension"] == "off"]
    if OUT.exists():
        shutil.rmtree(OUT)  # derived data, rebuilt whole from the raw takes
    manifest = []
    for row in rows:
        order = int(row["order"])
        p = passes[order]
        stamp = p["file"].removeprefix("02-Looopbck-")
        loop = sf.read(MEDIA / p["file"], dtype="float64")[0][p["start"] :]
        fx = sf.read(MEDIA / f"03-FX Out-{stamp}", dtype="float64")[0][p["start"] :]
        for sample in qc[order]["spikes_wav_sample"]:  # exact WAV index, not seconds
            i = sample - p["start"]
            fx[i] = 0.5 * (fx[i - 1] + fx[i + 1])
        parts = [
            (
                "test" if row["split"] == "test" else "test_seen",
                span("di_test", "di_test"),
            )
        ]
        if row["split"] == "train":
            parts.append(("trainval", span("di_train", "di_validation")))
        for split, (a, b) in parts:
            write(
                "dry", split, order, row["setting"], resample_poly(loop[a:b], 1, DOWN)
            )
            write("wet", split, order, row["setting"], resample_poly(fx[a:b], 1, DOWN))
            manifest.append(
                {
                    "take": order,
                    "setting": row["setting"],
                    "split": split,
                    "seconds": round((b - a) / RATE, 3),
                    "low_signal": qc[order]["low_signal"],
                    "spikes_repaired_wav_sample": qc[order]["spikes_wav_sample"],
                }
            )
        print(order, row["setting"], row["split"], flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    counts = {}
    for m in manifest:
        counts[m["split"]] = counts.get(m["split"], 0) + 1
    print(counts)


if __name__ == "__main__":
    main()
