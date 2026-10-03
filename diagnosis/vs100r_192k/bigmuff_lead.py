#!/usr/bin/env python3
"""Hypothesis born during the measures (DIAGNOSIS_vs100r_192k.md §6): is the ~1-sample
lead of the 48 kHz model over the real amp the ZOH advance (T1), not a capture offset?
Check on Big Muff models trained at 48 kHz: oracle fractional delay (+-2 samples,
step 0.05) of each model's test output against the real target, ZOH and FOH at the
same seeds. T1 predicts a lead for ZOH and none for FOH. Read-only, CPU.
Writes bigmuff_lead.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import product_nablafx_bench as bench  # noqa: E402
import product_sr_transfer as transfer  # noqa: E402

RUNS = [f"paired_ssm_wavenet{k}_s{s}" for k in ("", "_foh") for s in (925, 736, 688)]


def best_delay(y: np.ndarray, r: np.ndarray) -> tuple[float, float, float]:
    spectrum, freqs = np.fft.rfft(y), np.fft.rfftfreq(len(y))
    base = float(((r - y) ** 2).sum() / (r**2).sum())
    best = (np.inf, 0.0)
    for d in np.arange(-2, 2.0001, 0.05):
        z = np.fft.irfft(spectrum * np.exp(-2j * np.pi * freqs * d), len(y))
        best = min(best, (float(((r - z) ** 2).sum() / (r**2).sum()), float(d)))
    return base, best[0], best[1]


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(16)
    data = bench.data_module("test")
    data.setup("test")
    x = torch.stack([s[0] for s in data.test_dataset])
    y = torch.stack([s[1] for s in data.test_dataset])
    out = {}
    for run_id in RUNS:
        model, _ = transfer.load_model(run_id, "last")
        pred = transfer.run(model, x)
        rows = [
            best_delay(
                transfer.interior(p, 48_000)[0].double().numpy(),
                transfer.interior(t, 48_000)[0].double().numpy(),
            )
            for p, t in zip(pred, y, strict=True)
        ]
        delays = [r[2] for r in rows]
        out[run_id] = {
            "median_delay": float(np.median(delays)),
            "delays": delays,
            "mean_esr": float(np.mean([r[0] for r in rows])),
            "mean_esr_after_delay": float(np.mean([r[1] for r in rows])),
        }
        print(
            run_id,
            round(out[run_id]["median_delay"], 2),
            round(out[run_id]["mean_esr"], 4),
            round(out[run_id]["mean_esr_after_delay"], 4),
            flush=True,
        )
    (HERE / "bigmuff_lead.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
