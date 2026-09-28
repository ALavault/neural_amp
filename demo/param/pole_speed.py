#!/usr/bin/env python3
"""How fast do the SSM modes of the G2 Marshall models become? (Product lane, CPU.)

Per mode, the per-sample decay s = -Re(lambda dt), for every training setting of the
Marshall (the knob shifts of the "poles" arms included; the FiLM arms have none). A mode
with s >> 1 has no memory: it forgets within a sample and acts as an instantaneous gain
(demo/param/pole_speed_theory.md). Counts per run: modes faster than s = 10 for at least
one setting and for all; modes that are slow (s < 1) for one setting and memoryless
(s > 10) for another, i.e. switched by the knobs between filter and gain; the modes past
s = 30 with |C|.

Writes demo/param/pole_speed.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import product_nablafx_bench as bench  # noqa: E402

RUNS = [
    "g2_poles_additive_s925",
    "g2_poles_additive_s736",
    "g2_poles_additive_s688",  # stopped on a NaN; last checkpoint, step 9300
    "g2_poles_additive_s688_r2",  # its relaunch, floored poles (2d49ac7)
    "g2_poles_joint_s925",
    "param_ssm_wavenet_marshall",
    "g2_film_joint_s736",
    "g2_film_joint_s688",
    "g2_film_additive_s925",
]
ARMS = {
    "poles_additive": ("poles", "additive", [7, 7, 6, 6]),
    "poles_joint": ("poles", "joint", [25]),
    "film_additive": ("film", "additive", [8, 8, 8, 8]),
}


def load(run: str) -> tuple[torch.nn.Module, int]:
    arm = next((k for k in ARMS if k in run), None)
    cond, head, hidden = ARMS[arm] if arm else ("film", "joint", [32])
    args = argparse.Namespace(
        model="ssm-wavenet",
        device="marshall",
        num_blocks=8,
        channels=16,
        state_dim=4,
        output_act="tanh",
        discretization="zoh",
        cond=cond,
        cond_head=head,
        cond_hidden=hidden,
    )
    processor = bench.build_processor(args)
    path = bench.RUNS_DIR / f"nablafx_{run}/checkpoints/last.ckpt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    prefix = "model.processor."
    processor.load_state_dict(
        {
            k[len(prefix) :]: v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith(prefix)
        }
    )
    return processor.eval(), int(checkpoint["global_step"])


@torch.no_grad()
def main() -> None:
    data = bench.data_module("trainval", 16, "marshall")
    data.setup("fit")
    settings = sorted(
        {tuple(s["params"].tolist()) for s in data.train_dataset.dataset.samples}
    )
    controls = torch.tensor(settings)
    out = {}
    for run in RUNS:
        processor, step = load(run)
        shifts = (
            processor.pole_shifts(controls)
            if processor.pole_net is not None
            else [None] * len(processor.blocks)
        )
        counts = {"modes": 0, "fast_any": 0, "fast_all": 0, "switched": 0}
        extreme = []
        for index, (block, shift) in enumerate(
            zip(processor.blocks, shifts, strict=True)
        ):
            ssm = block.ssm
            dlog_dt, dlog_a = (
                shift
                if shift is not None
                else (
                    torch.zeros(len(settings), ssm.channels),
                    torch.zeros(len(settings), ssm.channels, ssm.state_dim),
                )
            )
            speed = torch.exp(ssm.log_A_real + dlog_a) * torch.exp(
                ssm.log_dt + dlog_dt
            ).unsqueeze(-1)  # s = -Re(lambda dt): (settings, channels, modes)
            fastest, slowest = speed.max(0).values, speed.min(0).values
            counts["modes"] += fastest.numel()
            counts["fast_any"] += int((fastest > 10).sum())
            counts["fast_all"] += int((slowest > 10).sum())
            counts["switched"] += int(((fastest > 10) & (slowest < 1)).sum())
            c = torch.view_as_complex(ssm.C.contiguous())
            for channel, mode in (fastest > 30).nonzero().tolist():
                extreme.append(
                    {
                        "block": index,
                        "channel": channel,
                        "mode": mode,
                        "s_max": round(float(fastest[channel, mode]), 1),
                        "s_min": round(float(slowest[channel, mode]), 2),
                        "abs_C": round(float(c[channel, mode].abs()), 3),
                    }
                )
        out[run] = {"step": step, **counts, "past_30": extreme}
        print(run, step, counts, extreme[:4], flush=True)
    Path(__file__).with_name("pole_speed.json").write_text(
        json.dumps({"settings": len(settings), "runs": out}, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
