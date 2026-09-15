#!/usr/bin/env python3
"""Test a benchmark run's last.ckpt as is and with its output negated.

Diagnoses the inverted solutions allowed by the sign-blind MR-STFT term. Takes
the same arguments as scripts/product_nablafx_bench.py, runs its nablafx test
loop on CPU (the GPU stays with the queue) for both signs, and writes
paper/icassp2027/data/polarity_<run_id>.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import product_nablafx_bench as bench  # noqa: E402
import torch  # noqa: E402
import wandb  # noqa: E402


def test(system: bench.System) -> dict[str, float]:
    tester = bench.pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[bench.metrics_callback()],
    )
    output = tester.test(system, datamodule=bench.data_module("test"), verbose=False)
    return {key: float(value) for key, value in output[0].items()}


def main() -> None:
    args = bench.parse_args()
    wandb.init(mode="disabled")
    torch.set_num_threads(16)
    _, (l1_weight, mrstft_weight) = bench.PUBLISHED.get(args.model, (0.01, (1.0, 0.1)))
    processor = bench.build_processor(args)
    system = bench.System(
        model=bench.BlackBoxModel(processor),
        loss=bench.FlexibleLoss(
            losses=[
                {"name": "l1_loss", "weight": l1_weight, "alias": "l1"},
                {"name": "mrstft_loss", "weight": mrstft_weight, "alias": "mrstft"},
            ]
        ),
        lr=0.01,
        log_media_every_n_steps=3000,
        use_callbacks=True,
    )
    path = bench.RUNS_DIR / f"nablafx_{args.run_id}/checkpoints/last.ckpt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    system.load_state_dict(checkpoint["state_dict"])
    as_is = test(system)
    layer = (
        processor.output_net[3]
        if isinstance(processor, bench.SSMWaveNet)
        else processor.contract
    )
    with torch.no_grad():
        layer.weight.neg_()
        layer.bias.neg_()
    record = {
        "run_id": args.run_id,
        "checkpoint": str(path.relative_to(ROOT)),
        "global_step": checkpoint["global_step"],
        "test_as_is": as_is,
        "test_negated": test(system),
    }
    print(json.dumps(record, indent=2))
    out = ROOT / f"paper/icassp2027/data/polarity_{args.run_id}.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
