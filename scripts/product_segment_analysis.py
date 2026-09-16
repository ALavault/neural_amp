#!/usr/bin/env python3
"""Per-segment test ESR of every recorded benchmark run, against segment level.

The protocol averages ESR over the twelve 5 s test segments, so quiet segments
weigh as much as loud ones. This runs each run's last checkpoint on CPU (the GPU
stays with the queue) and writes paper/icassp2027/data/test_segments.json with,
per run, the ESR of each segment and the mean over the six quietest and the six
loudest.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import product_nablafx_bench as bench  # noqa: E402
import torch  # noqa: E402

OUT = ROOT / "paper/icassp2027/data/test_segments.json"


def processor_args(record: dict) -> SimpleNamespace:
    ssm = record["ssm"] or {}
    return SimpleNamespace(
        model=record["model"],
        num_blocks=ssm.get("num_blocks", 8),
        channels=ssm.get("channels", 16),
        state_dim=ssm.get("state_dim", 4),
        output_act=ssm.get("output_act", "tanh"),
        discretization=ssm.get("discretization", "free"),
    )


def main() -> None:
    torch.set_num_threads(16)
    data = bench.data_module("test")
    data.setup("test")
    inputs = torch.stack([x for x, _ in data.test_dataset])
    targets = torch.stack([y for _, y in data.test_dataset])
    level = targets.pow(2).mean(-1).sqrt().flatten()
    half = len(level) // 2
    quiet = set(map(int, level.argsort()[:half]))
    # An inverted output has L1 = 2 E|y|; the published tables give validation
    # losses, so the level of the pooled training and validation recordings is
    # the reference for them.
    trainval = bench.data_module("trainval")
    trainval.setup("fit")
    pooled = torch.stack([target for _, target in trainval.trainval_dataset])
    record = {
        "trainval_target_mean_abs": float(pooled.abs().mean()),
        "segment_rms": level.tolist(),
        "quiet_segments": sorted(quiet),
        "runs": {},
    }
    for path in sorted((ROOT / "demo/nablafx_bench").glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = torch.load(
            ROOT / f"demo/runs/nablafx_{run['run_id']}/checkpoints/last.ckpt",
            map_location="cpu",
            weights_only=False,
        )
        processor = bench.build_processor(processor_args(run))
        processor.load_state_dict(
            {
                key.removeprefix("model.processor."): value
                for key, value in checkpoint["state_dict"].items()
                if key.startswith("model.processor.")
            }
        )
        processor.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(inputs), 8):
                processor.reset_states()
                predictions.append(processor(inputs[start : start + 8]))
        prediction = torch.cat(predictions)
        esr = ((targets - prediction).pow(2).sum(-1) / targets.pow(2).sum(-1)).flatten()
        by_half = {
            "quiet": statistics.fmean(
                v for i, v in enumerate(esr.tolist()) if i in quiet
            ),
            "loud": statistics.fmean(
                v for i, v in enumerate(esr.tolist()) if i not in quiet
            ),
        }
        record["runs"][run["run_id"]] = {"esr": esr.tolist(), **by_half}
        print(
            f"{run['run_id']:28s} mean {esr.mean():.4f}"
            f"  quiet half {by_half['quiet']:.4f}  loud half {by_half['loud']:.4f}"
            f"  ratio {by_half['quiet'] / by_half['loud']:.1f}",
            flush=True,
        )
    OUT.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
