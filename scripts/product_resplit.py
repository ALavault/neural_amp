#!/usr/bin/env python3
"""Re-split the Big Muff train file into three contiguous segments.

The data diagnosis showed the test file is 2.8x harder to overfit than the
validation file at matched budget. If the three original files come from
different takes (different playing style, pedal drift), cutting a single
take into train/val/test should eliminate that discrepancy.

This is NOT the capture chain (no pedal, no interface). It tests the
diagnosis's hypothesis using the one file we have that is a single take.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets/manifests/m4_internal.json"
DEVICE = "electro_harmonix_big_muff"
SAMPLE_RATE = 48_000
OUT_DIR = ROOT / "demo/resplit"

# Same 4:1:1 ratio as the original 120/30/30.
BOUNDARIES = {
    "train": (0, 80),
    "validation": (80, 100),
    "test": (100, 120),
}


def source_paths() -> tuple[Path, Path]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["device"] == DEVICE and entry["split"] == "train":
            return ROOT / entry["input_path"], ROOT / entry["target_path"]
    raise RuntimeError(f"no train pair for {DEVICE}")


def main() -> None:
    input_path, target_path = source_paths()
    input_audio, rate = sf.read(input_path, dtype="float32")
    target_audio, target_rate = sf.read(target_path, dtype="float32")
    if rate != SAMPLE_RATE or target_rate != SAMPLE_RATE:
        raise RuntimeError(f"source is at {rate}/{target_rate} Hz")
    if len(input_audio) != len(target_audio):
        raise RuntimeError("input and target differ in length")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    provenance = {
        "source_input": str(input_path.relative_to(ROOT)),
        "source_target": str(target_path.relative_to(ROOT)),
        "source_input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "source_target_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
        "sample_rate": SAMPLE_RATE,
        "splits": {},
    }

    pairs: dict[str, tuple[Path, Path]] = {}
    for split, (start_s, stop_s) in BOUNDARIES.items():
        start, stop = int(start_s * SAMPLE_RATE), int(stop_s * SAMPLE_RATE)
        for role, audio in (("input", input_audio), ("target", target_audio)):
            name = f"{split}_{role}.wav"
            sf.write(OUT_DIR / name, audio[start:stop], SAMPLE_RATE, subtype="FLOAT")
        pairs[split] = (OUT_DIR / f"{split}_input.wav", OUT_DIR / f"{split}_target.wav")
        provenance["splits"][split] = {
            "start_s": start_s,
            "stop_s": stop_s,
            "samples": stop - start,
            "input_sha256": hashlib.sha256(
                (OUT_DIR / f"{split}_input.wav").read_bytes()
            ).hexdigest(),
            "target_sha256": hashlib.sha256(
                (OUT_DIR / f"{split}_target.wav").read_bytes()
            ).hexdigest(),
        }

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    # Report the acoustic properties of each segment so the re-split test
    # is not confounded by a difficulty difference.
    print("=== Re-split du Big Muff depuis la prise d'entraînement ===\n")
    for split, (inp, tgt) in pairs.items():
        x, _ = sf.read(inp, dtype="float32")
        _y, _ = sf.read(tgt, dtype="float32")
        rms = float(np.sqrt(np.mean(np.square(x))))
        peak = float(np.max(np.abs(x)))
        crest = peak / rms
        print(
            f"  {split:11s} {len(x) / SAMPLE_RATE:5.1f} s  "
            f"RMS {rms:.4f}  crête {peak:.4f}  crête/RMS {crest:.2f}"
        )
    print(f"\nwrote {OUT_DIR}")
    print(json.dumps(pairs_to_dict(pairs), indent=2))


def pairs_to_dict(pairs: dict) -> dict:
    return {split: [str(i), str(t)] for split, (i, t) in pairs.items()}


if __name__ == "__main__":
    main()
