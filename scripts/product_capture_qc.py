#!/usr/bin/env python3
"""Align one reamp capture and cut it into training pairs.

Takes only the reference signal that was sent and the capture that came back.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

from fssr_nam.product.capture import (
    SAMPLE_RATE,
    estimate_latency,
    quality_control,
    split_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path, help="the reamp signal that was sent")
    parser.add_argument("capture", type=Path, help="what the device sent back")
    parser.add_argument("out_dir", type=Path, help="where to write the aligned pairs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference, reference_rate = sf.read(args.reference, dtype="float32")
    capture, capture_rate = sf.read(args.capture, dtype="float32")
    if reference_rate != SAMPLE_RATE or capture_rate != SAMPLE_RATE:
        raise RuntimeError(f"both files must be at {SAMPLE_RATE} Hz")

    report = quality_control(reference, capture)
    latency = estimate_latency(capture)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, (sent, got) in split_pairs(reference, capture, latency).items():
        sf.write(
            args.out_dir / f"{split}_input.wav", sent, SAMPLE_RATE, subtype="FLOAT"
        )
        sf.write(
            args.out_dir / f"{split}_target.wav", got, SAMPLE_RATE, subtype="FLOAT"
        )
    (args.out_dir / "qc.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
