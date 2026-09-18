#!/usr/bin/env python3
"""Does the direction of the one-float32-step nudge survive to the end? (Read-only, CPU.)

k = 1 is the child that breaks away in all three configurations where a spread exists
(decide at fork 100, both arms at fork 5), and its nudge direction is drawn with the same
seed 1000 + k on two different parents. With four children, three times in a row has about
one chance in sixteen of being fortuitous. If the direction carried information, the final
displacement of a child should keep some alignment with its initial nudge.

Measured here, on checkpoints already written: the cosine between the nudge delta applied
at the fork and the displacement the child ends up with, and the cosine between the nudges
themselves, which have no reason to be anything but orthogonal.

The nudge is reproduced exactly as scripts/product_fork_pilot.py applies it: a generator of
seed 1000 + k draws one bit per weight, and torch.nextafter moves each weight one float32
step in that direction, in the parameter order of the processor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import product_nablafx_bench as bench  # noqa: E402
import torch  # noqa: E402

FORKS = {100: "butterfly_ssm_seed42_f100", 5: "butterfly_ssm_seed42_f5"}
PARENT = "demo/runs/nablafx_butterfly_ssm_seed42_parent/checkpoints"


def processor() -> torch.nn.Module:
    return bench.build_processor(
        SimpleNamespace(
            model="ssm-wavenet",
            num_blocks=8,
            channels=16,
            state_dim=4,
            output_act="tanh",
            discretization="zoh",
        )
    )


def final(name: str) -> torch.Tensor:
    state = torch.load(
        bench.RUNS_DIR / f"nablafx_{name}/checkpoints/last.ckpt",
        map_location="cpu",
        weights_only=False,
    )["state_dict"]
    return torch.cat(
        [
            v.flatten()
            for k, v in state.items()
            if k.startswith("model.processor.") and v.is_floating_point()
        ]
    )


def nudge(fork: int, k: int) -> torch.Tensor:
    """The delta the fork applies, in the processor's own parameter order."""
    model = processor()
    state = torch.load(
        ROOT / PARENT / f"fork_epoch{fork}.ckpt", map_location="cpu", weights_only=False
    )["state_dict"]
    model.load_state_dict(
        {key.removeprefix("model.processor."): v for key, v in state.items() if key.startswith("model.processor.")}
    )
    generator = torch.Generator().manual_seed(1000 + k)
    deltas = []
    for parameter in model.parameters():
        up = torch.randint(0, 2, parameter.shape, generator=generator).bool()
        limit = torch.where(up, 1e30, -1e30).to(parameter)
        deltas.append((torch.nextafter(parameter.data, limit) - parameter.data).flatten())
    return torch.cat(deltas)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(a @ b / (a.norm() * b.norm()))


def main() -> None:
    out = {}
    for fork, prefix in FORKS.items():
        control = final(f"{prefix}_decide_k0")
        nudges = {k: nudge(fork, k) for k in (1, 2, 3, 4)}
        print(f"fourche {fork} : norme du coup de pouce {nudges[1].norm():.3e}")
        for arm in ("decide", "replay"):
            rows = {}
            for k in (1, 2, 3, 4):
                try:
                    theta = final(f"{prefix}_{arm}_k{k}")
                except FileNotFoundError:
                    continue
                displacement = theta - control
                # The nudge is applied to the processor's parameters, the final weights are
                # read in the state dict order; both follow the same declaration order.
                rows[k] = {
                    "cos_nudge_displacement": cosine(nudges[k], displacement),
                    "displacement_norm": float(displacement.norm()),
                }
            if rows:
                out[f"f{fork}_{arm}"] = rows
                print(
                    f"  {arm:6s} cos(coup de pouce, deplacement final) "
                    + "  ".join(f"k{k}:{v['cos_nudge_displacement']:+.4f}" for k, v in rows.items())
                )
        pairs = {
            f"{a}-{b}": cosine(nudges[a], nudges[b])
            for a in (1, 2, 3, 4)
            for b in (1, 2, 3, 4)
            if a < b
        }
        out[f"f{fork}_nudge_pairs"] = pairs
        print(
            "  cos entre coups de pouce "
            + "  ".join(f"{k}:{v:+.4f}" for k, v in pairs.items())
        )

    (Path(__file__).parent / "direction_survives.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
