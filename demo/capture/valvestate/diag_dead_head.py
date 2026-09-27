#!/usr/bin/env python3
"""Why did vs100r_gc_film_joint_s925 learn nothing? (Product lane, CPU, diagnostic.)

The trained model outputs a constant: every input of the output head's first ReLU is
negative, so no gradient flows. Hypothesis: the four gain-0 takes, whose targets are
near-silent (-78 dBFS), give the log-magnitude part of MR-STFT large gradients that
push the output down until the head dies. Test: replay the first training steps with
the run's settings (seed 42, AdamW lr 0.01 with the SSM groups, value clipping 1.0,
0.5 L1 + 0.5 MR-STFT, batch 16), with and without the gain-0 takes, and log per step
the loss and the share of the head's first ReLU inputs that are positive.

Usage: diag_dead_head.py {all,no_gain0} STEPS
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import product_nablafx_bench as bench  # noqa: E402
from nablafx.core import BlackBoxModel  # noqa: E402
from nablafx.evaluation.flexible_loss import FlexibleLoss  # noqa: E402


def main() -> None:
    variant, steps = sys.argv[1], int(sys.argv[2])
    torch.set_num_threads(12)
    record = json.loads(
        (bench.RESULTS_DIR / "vs100r_gc_film_joint_s925.json").read_text()
    )
    pl.seed_everything(42, workers=True)
    args = bench.argparse.Namespace(
        model="ssm-wavenet", device="vs100r", **record["ssm"]
    )
    model = BlackBoxModel(bench.build_processor(args))
    loss = FlexibleLoss(
        losses=[
            {"name": "l1_loss", "weight": 0.5, "alias": "l1"},
            {"name": "mrstft_loss", "weight": 0.5, "alias": "mrstft"},
        ]
    )
    system = bench.System(model=model, loss=loss, lr=record["lr"])
    system.honor_optim = True
    optimizer = system.configure_optimizers()[0][0]
    data = bench.data_module("trainval", 16, "vs100r")
    data.setup("fit")
    train = data.train_dataset
    if variant == "no_gain0":
        keep = [i for i in range(len(train)) if float(train[i][2][0]) > 0]
        train = Subset(train, keep)
    loader = DataLoader(train, batch_size=16, shuffle=True, drop_last=True)
    head = model.processor.output_net[0]
    alive = {}
    head.register_forward_hook(
        lambda _m, inp, _o: alive.__setitem__("v", float((inp[0] > 0).float().mean()))
    )
    step = 0
    log = []
    while step < steps:
        for x, y, c in loader:
            model.train()
            pred = model(x, c)
            values = loss(pred, y)
            optimizer.zero_grad()
            values[-1].backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            gain0 = int((c[:, 0] == 0).sum())
            row = {
                "step": step,
                "total": float(values[-1]),
                "l1": float(values[0]),
                "mrstft": float(values[1]),
                "head_alive": alive["v"],
                "gain0_in_batch": gain0,
            }
            log.append(row)
            print(json.dumps(row), flush=True)
            step += 1
            if step >= steps:
                break
    out = Path(__file__).with_name(f"diag_dead_head_{variant}.json")
    out.write_text(json.dumps(log, indent=1) + "\n")


if __name__ == "__main__":
    main()
