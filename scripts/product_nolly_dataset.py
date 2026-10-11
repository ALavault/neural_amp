#!/usr/bin/env python3
"""Build the Nolly parametric dataset (D-0015, demo/nolly_models/predictions.md).
(Product lane, CPU.)

Input = the source file v2, common to every take; target = the take's render; knobs
gain, bass, middle, treble, presence. Both decimated 192 -> 48 kHz (fssr_nam.product.
nolly), written as float WAV. Takes: nolly.model_takes(), the train and test rows of
demo/capture/v2/settings.csv without the cab-on controls and the repeats 20 and 39.

Latency, rule of predictions.md: m = median marker lag of the cab-off takes, at
192 kHz, from the gates of nolly_reading.py (--reading); one common advance of
round(m / 4) samples at 48 kHz (half away from zero) for every take, the target read
that much later; 0 when it rounds to 0. Per-take lags are never compensated.
Takes kept against « same lag at both markers » (Écart of 2026-10-11) are listed in
the manifest under "waived".

Layout read by nablafx's ParametricPluginDataset, as product_valvestate_dataset.py:
  dry/<split>/takeNN.input.wav     symlinks to source/<split>.wav (one input per split)
  wet/<split>/<setting>/<setting>.takeNN.target.wav
  trainval: di_train + di_validation of the train takes; test: di_test of the 8
  combination-test takes (never trained on); test_seen: di_test of the train takes.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from fssr_nam.product import nolly

OUT = nolly.DATA / "nolly_48k"
PARTS = {
    "trainval": ("di_train", "di_validation"),
    "test": ("di_test", "di_test"),
    "test_seen": ("di_test", "di_test"),
}


def waivable(gate: dict, samples: int) -> bool:
    """Every per-take gate of nolly_reading.py except « same lag at both markers »
    (its frozen constants: lag search 4 096, markers 20 dB over silence, quiet steps
    above -100 dBFS)."""
    return (
        gate.get("format", {}).get("ok", False)
        and gate.get("finite", False)
        and gate.get("samples", 0) >= samples
        and max(abs(gate["lag_start"]), abs(gate["lag_end"])) < 4_096
        and gate["markers_over_silence_db"] >= 20.0
        and gate["quiet_step_min_db"] > -100.0
    )


def common_advance(reading: Path, renders: Path) -> tuple[float, int, dict]:
    """Median marker lag, common advance and the waived takes; no dataset unless the
    reading passed its gates (bouclage, stéréo, every model take, anchors, latency:
    predictions.md). Écart of 2026-10-11 (predictions.md): a model take that fails only
    « same lag at both markers » is kept when its output at the two marker sections
    shows no drift (nolly.marker_drift, lag 0)."""
    result = json.loads(reading.read_text())
    gates = result["gates"]
    samples = sf.info(nolly.SOURCE).frames
    failed, waived = [], {}
    for t in nolly.model_takes():
        gate = gates.get("takes", {}).get(str(t["order"]), {})
        if gate.get("pass"):
            continue
        drift = nolly.marker_drift(t, renders) if waivable(gate, samples) else None
        if drift and drift["lag"] == 0:
            waived[t["order"]] = {
                "lag_start": gate["lag_start"],
                "lag_end": gate["lag_end"],
                **drift,
            }
        else:
            failed.append(t["order"])
    if "stopped" in result or failed:
        raise SystemExit(
            f"probe gates not passed ({result.get('stopped')}; takes {failed}):"
            " no training (demo/nolly_models/predictions.md)"
        )
    median = gates["median_lag"]
    advance = int(math.copysign(math.floor(abs(median) / nolly.DOWN + 0.5), median))
    return median, advance, waived


def write(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), 48_000, subtype="FLOAT")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reading", type=Path, required=True)
    parser.add_argument("--renders", type=Path, default=nolly.RENDERS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    median, advance, waived = common_advance(args.reading, args.renders)
    if args.out.exists():
        shutil.rmtree(args.out)  # derived data, rebuilt whole from the renders
    for split, (first, last) in PARTS.items():
        write(args.out / "source" / f"{split}.wav", nolly.source(first, last))
    manifest = []
    for take in nolly.model_takes():
        splits = ["test"] if take["split"] == "test" else ["trainval", "test_seen"]
        for split in splits:
            first, last = PARTS[split]
            y = nolly.output(take, first, last, args.renders, advance)
            name = f"take{take['order']:02d}"
            dry = args.out / "dry" / split / f"{name}.input.wav"
            dry.parent.mkdir(parents=True, exist_ok=True)
            dry.symlink_to(Path("../../source") / f"{split}.wav")
            setting = take["setting"]
            write(
                args.out / "wet" / split / setting / f"{setting}.{name}.target.wav", y
            )
            manifest.append({"take": take["order"], "setting": setting, "split": split})
        print(take["order"], take["setting"], take["split"], flush=True)
    counts = {s: sum(m["split"] == s for m in manifest) for s in PARTS}
    summary = {
        "renders": str(args.renders.resolve()),
        "median_marker_lag_192k": median,
        "advance_48k": advance,
        "waived": waived,
        "counts": counts,
        "takes": manifest,
    }
    (args.out / "manifest.json").write_text(json.dumps(summary, indent=1) + "\n")
    print({"median_marker_lag_192k": median, "advance_48k": advance, **counts})


if __name__ == "__main__":
    main()
