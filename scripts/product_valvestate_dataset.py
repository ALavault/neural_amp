#!/usr/bin/env python3
"""Build the exploratory VS100R parametric dataset. (Product lane, CPU.)

Input = each take's own loopback (the signal the amp received), target = its FX loop
send: the two tracks share one USB stream, so the pair stays in step through the
microframe slips (demo/capture/valvestate/QC.md). OD2, Power Dimension off, knobs
gain, contour, bass, middle, treble. Decimated 192 -> 48 kHz (explore_48k), or kept at
192 kHz with --rate 192000 (explore_192k, demo/capture/valvestate/predictions_192k.md),
written as float WAV.

Layout read by nablafx's ParametricPluginDataset (params parsed from the target name):
  dry/trainval/takeNN.input.wav     di_train + di_validation of the train settings
  wet/trainval/<setting>/<setting>.takeNN.target.wav
  dry/test, wet/test                di_test of the 8 test settings (never trained on)
  dry/test_seen, wet/test_seen      di_test of the train settings (unseen audio)
Isolated spikes found by product_valvestate_qc.py are replaced by the mean of their two
neighbours before decimation.
"""

from __future__ import annotations

import argparse
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
DATA = ROOT / "datasets/raw/internal/valvestate_vs100r"
LAYOUT = json.loads((SHEET / "layout.json").read_text())
SECTIONS = {s["section"]: s for s in LAYOUT["sections"]}
RATE = LAYOUT["sample_rate"]


def span(first: str, last: str) -> tuple[int, int]:
    return SECTIONS[first]["start"], SECTIONS[last]["start"] + SECTIONS[last]["samples"]


def write(
    out: Path, rate: int, kind: str, split: str, take: int, setting: str, audio
) -> None:
    if kind == "dry":
        path = out / "dry" / split / f"take{take:02d}.input.wav"
    else:
        path = out / "wet" / split / setting / f"{setting}.take{take:02d}.target.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), rate, subtype="FLOAT")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, choices=(48_000, 192_000), default=48_000)
    rate = parser.parse_args().rate
    down = RATE // rate
    out = DATA / f"explore_{rate // 1000}k"

    def resampled(x: np.ndarray) -> np.ndarray:
        return x if down == 1 else resample_poly(x, 1, down)

    passes = {p["order"]: p for p in json.loads((SHEET / "passes.json").read_text())}
    qc = {r["order"]: r for r in json.loads((SHEET / "qc.json").read_text())}
    with (SHEET / "settings.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream)]
    rows = [r for r in rows if r["channel"] == "OD2" and r["power_dimension"] == "off"]
    if out.exists():
        shutil.rmtree(out)  # derived data, rebuilt whole from the raw takes
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
            write(out, rate, "dry", split, order, row["setting"], resampled(loop[a:b]))
            write(out, rate, "wet", split, order, row["setting"], resampled(fx[a:b]))
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
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    counts = {}
    for m in manifest:
        counts[m["split"]] = counts.get(m["split"], 0) + 1
    print(counts)


if __name__ == "__main__":
    main()
