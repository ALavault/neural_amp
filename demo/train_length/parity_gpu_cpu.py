#!/usr/bin/env python3
"""GPU inference of coreading_delay.py against the CPU inference of the P reading.

For every run (checkpoint last) and take, ln ESR on di_test without any delay, on the
GPU path of coreading_delay.py, against ln_esr_by_take of eval_train_length.json (CPU).
Writes parity_gpu_cpu.json: per-take differences and their effect on the paired d_i.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coreading_delay as cr  # noqa: E402

tl = cr.tl


@torch.no_grad()
def main() -> None:
    p_runs = json.loads((HERE / "eval_train_length.json").read_text())["runs"]
    takes = [t for t in cr.ev.takes() if not t["low_signal"]]
    gaps = {}
    for pattern in tl.ARMS.values():
        for seed in tl.SEEDS:
            run_id = pattern.format(seed)
            model = tl.load(run_id, "last").to(cr.DEVICE)
            cpu = p_runs[run_id]["last"]["ln_esr_by_take"]
            gaps[run_id] = {}
            for take in takes:
                loop, fx = cr.ev.audio(take)
                y = cr.run(model, resample_poly(loop, 1, 4), cr.controls(take))
                gpu = math.log(cr.ev.esr(y, resample_poly(fx, 1, 4)))
                gaps[run_id][take["order"]] = gpu - cpu[str(take["order"])]
    worst = max(abs(v) for run in gaps.values() for v in run.values())
    result = {"max_abs_ln_esr_gap": worst, "ln_esr_gap_by_run_and_take": gaps}
    print(f"max |ln ESR gpu - cpu| over runs and takes: {worst:.2e}")
    (HERE / "parity_gpu_cpu.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
