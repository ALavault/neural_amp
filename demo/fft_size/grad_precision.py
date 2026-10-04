#!/usr/bin/env python3
"""Gradient precision of the float32 training step against float64 (NOTE.md).

Same model and batch as bench_fft.py, first 2 segments (float64 must fit). Reference:
the model in float64 with the pow2 FFT. For each FFT length, TF32 on and off: relative
error of the whole gradient and of each parameter tensor; and fast against pow2 in
float64. GPU. Writes demo/fft_size/grad_precision.json.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_fft as b  # noqa: E402

from fssr_nam.models.ssm_wavenet import DiagonalSSMLayer  # noqa: E402


def grads(model, x, y, c, dtype, kind: str) -> dict[str, torch.Tensor]:
    from auraloss.freq import MultiResolutionSTFTLoss

    DiagonalSSMLayer.fft_size = kind
    loss_fn = MultiResolutionSTFTLoss().cuda()
    x, y, c = x.to(dtype), y.to(dtype), c.to(dtype)
    model.zero_grad(set_to_none=True)
    model.reset_states()
    out = model(x, c)
    (0.5 * (out - y).abs().mean() + 0.5 * loss_fn(out, y)).backward()
    return {
        n: p.grad.detach().double().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }


def whole(g: dict, ref: dict) -> float:
    num = sum(float((g[n] - ref[n]).norm() ** 2) for n in ref)
    return (num / sum(float(ref[n].norm() ** 2) for n in ref)) ** 0.5


def main() -> None:
    torch.use_deterministic_algorithms(True)
    torch.nn.functional.pad = b.bench._reflect_pad_by_slicing
    base = b.ev.load(b.RUN).cuda().train()
    x, y, c = (t[:2] for t in b.batch())
    m64 = copy.deepcopy(base).double()
    ref = grads(m64, x, y, c, torch.float64, "pow2")
    ref_fast = grads(m64, x, y, c, torch.float64, "fast")
    result = {"float64_fast_vs_pow2_whole": whole(ref_fast, ref)}
    for tf32 in (True, False):
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
        torch.backends.cudnn.allow_tf32 = tf32
        for kind in ("pow2", "fast"):
            g = grads(base, x, y, c, torch.float32, kind)
            key = f"{kind}_{'tf32' if tf32 else 'no_tf32'}"
            result[key] = {
                "whole": whole(g, ref),
                "per_tensor": {
                    n: float((g[n] - ref[n]).norm() / (ref[n].norm() + 1e-300))
                    for n in ref
                },
            }
            print(key, round(result[key]["whole"], 5), flush=True)
    DiagonalSSMLayer.fft_size = "pow2"
    (HERE / "grad_precision.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
