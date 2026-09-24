#!/usr/bin/env python3
"""Model-to-target lag on the TRAINING files (DIAGNOSIS_ffuzz.md, H1). Read-only, CPU.

Same measurement as measure.py (best lag within +-400 samples, interior ESR at lag 0 and
at that lag), on 5 s segments of the five trainval files at F000 and F100: the first
ten segments after the first minute of each file. If the test lag of +4 is a property
the models failed to learn, it shows here too; if it is 0 here, the test files are
aligned differently from the training files. Written to train_lag.json.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from measure import RUNS, best_lag, interior_esr, load

ROOT = Path(__file__).resolve().parents[2]
PARAM = ROOT / "datasets/raw/external/tone_twist_param/extracted"
DRY = PARAM / "DIGITAL-PARAMETRIC/DRY-with-markers/trainval"
WET = PARAM / "MultidrivePedalPro-F-Fuzz/trainval"
N, START, COUNT = 240_000, 48_000 * 60, 10


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(16)
    out = {}
    for run in RUNS:
        model = load(run)
        rows = []
        for dry in sorted(DRY.glob("*.input.wav")):
            name = dry.name.split(".")[0]
            x_all, _ = sf.read(dry, dtype="float32")
            for setting, fuzz in (("F000_V100", 0.0), ("F100_V100", 1.0)):
                y_all, _ = sf.read(
                    WET / setting / f"{setting}.{name}.target.wav", dtype="float32"
                )
                starts = [START + k * N for k in range(COUNT)]
                x = torch.stack([torch.tensor(x_all[s : s + N]) for s in starts])[
                    :, None
                ]
                y = torch.stack([torch.tensor(y_all[s : s + N]) for s in starts])[
                    :, None
                ]
                model.reset_states()
                pred = model(x, torch.full((COUNT, 1), fuzz))
                for k in range(COUNT):
                    p, t = pred[k, 0], y[k, 0]
                    lag = best_lag(p, t)
                    rows.append(
                        {
                            "file": name,
                            "setting": setting,
                            "lag": lag,
                            "esr_lag0": interior_esr(p, t, 0),
                            "esr_best": interior_esr(p, t, lag),
                        }
                    )
        out[run] = rows
        print(run)
        for name in sorted({r["file"] for r in rows}):
            mine = [r for r in rows if r["file"] == name]
            print(
                f"  {name:14s} lags {Counter(r['lag'] for r in mine).most_common(3)}  "
                f"ESR lag0 {statistics.mean(r['esr_lag0'] for r in mine):.4f}  "
                f"best {statistics.mean(r['esr_best'] for r in mine):.4f}",
                flush=True,
            )
    (Path(__file__).with_name("train_lag.json")).write_text(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
