#!/usr/bin/env python3
"""Where does g2_poles_additive_s688's NaN come from? (Product lane, CPU, diagnostic.)

The run stopped on "NaN loss encountered" between steps 9300 and about 9400. Its last
checkpoint (step 9300) is finite, no parameter is extreme and the gradient norm stayed
between 1 and 5 up to step 9299: the NaN came suddenly. With the step-9300 weights, this
runs every training and validation segment of split 688 forward and backward in batches
of 16 (0.5 L1 + 0.5 MR-STFT, as the run) and records, per batch: any non-finite module
output (with the module), the largest absolute output of each module type, the loss and
whether every gradient is finite. A batch that already produces a non-finite value at
these weights names the operation; if none does, the NaN needs further updates.

Writes demo/param/diag_nan_d688.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import product_nablafx_bench as bench  # noqa: E402
from nablafx.core import BlackBoxModel  # noqa: E402
from nablafx.evaluation.flexible_loss import FlexibleLoss  # noqa: E402

RUN = "g2_poles_additive_s688"


def main() -> None:
    torch.set_num_threads(16)
    torch.manual_seed(0)
    args = argparse.Namespace(
        model="ssm-wavenet",
        device="marshall",
        num_blocks=8,
        channels=16,
        state_dim=4,
        output_act="tanh",
        discretization="zoh",
        cond="poles",
        cond_head="additive",
        cond_hidden=[7, 7, 6, 6],
    )
    processor = bench.build_processor(args)
    path = bench.RUNS_DIR / f"nablafx_{RUN}/checkpoints/last.ckpt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    prefix = "model.processor."
    processor.load_state_dict(
        {
            k[len(prefix) :]: v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith(prefix)
        }
    )
    model = BlackBoxModel(processor).train()
    loss = FlexibleLoss(
        losses=[
            {"name": "l1_loss", "weight": 0.5, "alias": "l1"},
            {"name": "mrstft_loss", "weight": 0.5, "alias": "mrstft"},
        ]
    )
    peaks: dict[str, float] = {}
    bad: list[str] = []

    def hook(name):
        def record(_module, _inputs, output):
            if not torch.is_tensor(output):
                return
            value = output.detach()
            if value.is_complex():
                value = value.abs()
            if not torch.isfinite(value).all():
                bad.append(name)
            kind = name.split(".")[-1] if name else "root"
            peaks[kind] = max(peaks.get(kind, 0.0), float(value.abs().max()))

        return record

    for name, module in processor.named_modules():
        module.register_forward_hook(hook(name))
    data = bench.data_module("trainval", 16, "marshall")
    data.setup("fit")
    loader = DataLoader(
        ConcatDataset([data.train_dataset, data.val_dataset]), batch_size=16
    )
    log = []
    for index, (x, y, c) in enumerate(loader):
        peaks.clear()
        bad.clear()
        model.zero_grad()
        model.reset_states()
        values = loss(model(x, c), y)
        values[-1].backward()
        grads_finite = all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        )
        row = {
            "batch": index,
            "loss": float(values[-1]),
            "grads_finite": bool(grads_finite),
            "non_finite_modules": sorted(set(bad))[:10],
            "peaks": dict(peaks),
            "controls_min": c.min(0).values.tolist(),
            "controls_max": c.max(0).values.tolist(),
        }
        log.append(row)
        flag = "" if grads_finite and not bad else "  <-- NON FINI"
        print(
            f"lot {index:3d} perte {row['loss']:.4f}"
            f" pic max {max(peaks.values()):.3g}{flag}",
            flush=True,
        )
    Path(__file__).with_name("diag_nan_d688.json").write_text(
        json.dumps(log, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
