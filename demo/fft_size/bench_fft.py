#!/usr/bin/env python3
"""SSM convolution FFT length: next power of 2 (524288) against next 2-3-5-smooth
length (288000) for 144000-sample segments. Product lane, GPU, no training.

The VS100R 48 kHz model (vs100r_gc_film_joint_s925_r2 weights, so the poles are the
trained ones) on one real training batch (16 x 144000), the bench's loss (0.5 L1 +
0.5 MR-STFT), deterministic algorithms as in the bench. Per variant: time of forward +
loss + backward + gradient copy, no optimizer step (median of 20 after 3 warm-up
steps, two interleaved passes), peak allocated memory, with TF32 on (the runs' setting)
and off; and the difference of the outputs between the two FFT lengths.
Writes demo/fft_size/bench_fft.json.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import product_nablafx_bench as bench  # noqa: E402
import product_vs100r_eval as ev  # noqa: E402

from fssr_nam.models.ssm_wavenet import DiagonalSSMLayer  # noqa: E402

RUN = "vs100r_gc_film_joint_s925_r2"


def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(925)
    data = bench.data_module("trainval", 16, "vs100r")
    data.setup("fit")
    items = [data.train_dataset[i] for i in range(16)]
    x, y, c = (torch.stack(p) for p in zip(*items, strict=True))
    return x.cuda(), y.cuda(), c.cuda()


def step(model, loss_fn, x, y, c) -> tuple[torch.Tensor, list[torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    model.reset_states()
    out = model(x, c)
    loss = 0.5 * (out - y).abs().mean() + 0.5 * loss_fn(out, y)
    loss.backward()
    return out.detach(), [
        p.grad.detach().clone() for p in model.parameters() if p.grad is not None
    ]


def main() -> None:
    torch.use_deterministic_algorithms(True)
    # The bench's --deterministic replaces reflect padding by a deterministic version.
    torch.nn.functional.pad = bench._reflect_pad_by_slicing
    model = ev.load(RUN).cuda().train()
    # auraloss is a submodule the bench puts on sys.path when it is imported.
    from auraloss.freq import MultiResolutionSTFTLoss

    loss_fn = MultiResolutionSTFTLoss().cuda()
    x, y, c = batch()
    result = {"run": RUN, "batch": list(x.shape), "gpu": torch.cuda.get_device_name()}
    outputs = {}
    # Interleaved, twice each: TF32 on (the runs' setting) and off, both FFT lengths.
    for tf32 in (True, False, True, False):
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
        torch.backends.cudnn.allow_tf32 = tf32
        for kind in ("pow2", "fast"):
            DiagonalSSMLayer.fft_size = kind
            for _ in range(3):
                step(model, loss_fn, x, y, c)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(20):
                torch.cuda.synchronize()
                start = time.perf_counter()
                out, _ = step(model, loss_fn, x, y, c)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
            key = f"{kind}_tf32" if tf32 else f"{kind}_no_tf32"
            entry = result.setdefault(key, {"median_s": [], "peak_allocated_gib": []})
            entry["median_s"].append(statistics.median(times))
            entry["peak_allocated_gib"].append(
                torch.cuda.max_memory_allocated() / 2**30
            )
            if tf32:
                outputs[kind] = out
            print(key, round(entry["median_s"][-1], 4), "s", flush=True)
    DiagonalSSMLayer.fft_size = "pow2"
    o1, o2 = outputs["pow2"], outputs["fast"]
    result["output_max_abs_diff"] = float((o1 - o2).abs().max())
    result["output_relative_error_norm"] = float((o1 - o2).norm() / o1.norm())
    for key in [k for k in result if k.startswith(("pow2", "fast"))]:
        result[key]["mean_of_medians_s"] = statistics.mean(result[key]["median_s"])
    (HERE / "bench_fft.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
