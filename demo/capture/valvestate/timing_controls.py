#!/usr/bin/env python3
"""Do other models show the VS100R model's one-sample lead? (Product lane. POST HOC.)

Control for timing_shift.py, written after it: the same fractional delay scan (-1.5 to
+2.5 samples) on 10 test segments of each model below, scored with the same window as
product_vs100r_eval.py (0.5 s at the start, 0.1 s at the end):
- Big Muff, pilot C decide arms, seeds 42 and 44 (SSM-WaveNet, ReLU head);
- Marshall, param_ssm_wavenet_marshall (FiLM, ReLU head);
- VS100R, vs100r_gc_film_joint_s925_r2 on 10 training segments (LeakyReLU head).

Writes demo/capture/valvestate/timing_controls.json.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
for sub in ("scripts", "diagnosis/butterfly", "demo/param", "demo/capture/valvestate"):
    sys.path.insert(0, str(ROOT / sub))
import product_nablafx_bench as bench  # noqa: E402
import product_vs100r_eval as ev  # noqa: E402
import rate_transfer  # noqa: E402
from mode_connectivity import processor, weights  # noqa: E402
from timing_shift import delayed  # noqa: E402

SHIFTS = np.round(np.arange(-1.5, 2.5001, 0.1), 1)


def scan(model, dataset, parametric: bool, count: int = 10) -> list[dict]:
    rows = []
    for i in np.linspace(0, len(dataset) - 1, count).astype(int).tolist():
        item = dataset[i]
        x, y = item[0][None], item[1][0].numpy().astype(np.float64)
        with torch.no_grad():
            model.reset_states()
            out = model(x, item[2][None]) if parametric else model(x)
        m = out[0, 0].numpy().astype(np.float64)
        curve = [ev.esr(delayed(m, d), y) for d in SHIFTS]
        best = int(np.argmin(curve))
        rows.append(
            {
                "segment": i,
                "esr_0": curve[int(np.argmin(np.abs(SHIFTS)))],
                "best_shift": float(SHIFTS[best]),
                "esr_best": curve[best],
            }
        )
    return rows


def main() -> None:
    torch.set_num_threads(16)
    out = {}
    bigmuff = bench.data_module("test", device="bigmuff")
    bigmuff.setup("test")
    for run in ("pilotC_decide_seed42", "pilotC_decide_seed44"):
        model = processor()
        model.load_state_dict(weights(run))
        model.eval()
        out[run] = scan(model, bigmuff.test_dataset, parametric=False)
    marshall = bench.data_module("test", device="marshall")
    marshall.setup("test")
    model = rate_transfer.load_marshall("param_ssm_wavenet_marshall")
    out["param_ssm_wavenet_marshall"] = scan(model, marshall.test_dataset, True)
    vs = bench.data_module("trainval", 16, "vs100r")
    vs.setup("fit")
    model = ev.load("vs100r_gc_film_joint_s925_r2")
    out["vs100r_r2_training"] = scan(model, vs.train_dataset, True)
    summary = {
        run: {
            "median_best_shift": statistics.median(r["best_shift"] for r in rows),
            "median_esr_0": statistics.median(r["esr_0"] for r in rows),
            "median_esr_best": statistics.median(r["esr_best"] for r in rows),
        }
        for run, rows in out.items()
    }
    print(json.dumps(summary, indent=1))
    Path(__file__).with_name("timing_controls.json").write_text(
        json.dumps({"summary": summary, "rows": out}, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
