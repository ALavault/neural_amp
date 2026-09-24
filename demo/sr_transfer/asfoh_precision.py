#!/usr/bin/env python3
"""Control for FIX_foh.md (a): is the 48 kHz identity ESR of --as-foh (4e-11 on s925)
at the float32 floor of the network itself? ESR of the ZOH model against itself in
float64, and of the converted model in float32 and float64. All twelve test segments."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import product_nablafx_bench as bench
import product_sr_transfer as transfer

from fssr_nam.models.ssm_wavenet import zoh_to_foh


def esr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b) ** 2).sum() / (b**2).sum())


torch.set_num_threads(8)
data = bench.data_module("test")
data.setup("test")
x = torch.stack([data.test_dataset[i][0] for i in range(12)])
for seed in (925, 736, 688):
    zoh32, _ = transfer.load_model(f"paired_ssm_wavenet_s{seed}", "last")
    zoh64 = copy.deepcopy(zoh32).double()
    foh32 = copy.deepcopy(zoh32)
    zoh_to_foh(foh32)
    foh64 = copy.deepcopy(zoh64)
    zoh_to_foh(foh64)
    ref = transfer.run(zoh64, x.double())
    print(
        f"s{seed}: zoh32 vs zoh64 {esr(transfer.run(zoh32, x).double(), ref):.1e}  "
        f"foh32 vs zoh32 {esr(transfer.run(foh32, x), transfer.run(zoh32, x)):.1e}  "
        f"foh64 vs zoh64 {esr(transfer.run(foh64, x.double()), ref):.1e}",
        flush=True,
    )
