#!/usr/bin/env python3
"""Check that SSM-WaveNet's per-sample recurrence reproduces its FFT form.

Training applies each diagonal SSM as an FFT convolution over the segment;
streaming inference runs DiagonalSSMLayer.step sample by sample. The two must
give the same output. Runs on CPU in float64, except that step casts each layer
input to single-precision complex, and writes the maximum deviation to
paper/icassp2027/data/recurrence_check.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from fssr_nam.models.ssm_wavenet import SSMWaveNet

ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT / "datasets/raw/external/tone_twist_bigmuff/extracted/DRY/test/test.input.wav"
)
OUT = ROOT / "paper/icassp2027/data/recurrence_check.json"


def load(checkpoint: Path, state_dim: int, discretization: str) -> SSMWaveNet:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    prefix = "model.processor."
    state = {
        k[len(prefix) :]: v
        for k, v in state["state_dict"].items()
        if k.startswith(prefix)
    }
    model = SSMWaveNet(
        num_blocks=8, channels=16, state_dim=state_dim, discretization=discretization
    )
    model.load_state_dict(state)
    return model.double().eval()


@torch.inference_mode()
def recurrent(model: SSMWaveNet, x: torch.Tensor) -> torch.Tensor:
    model.reset_states()
    out = []
    for t in range(x.shape[-1]):
        h = model.input_conv(x[..., t : t + 1])[..., 0]
        skip_sum = torch.zeros_like(h)
        for block in model.blocks:
            gates = block.pre_gate(block.ssm.step(h).unsqueeze(-1))
            a, b = gates.chunk(2, dim=1)
            g = torch.tanh(a) * torch.sigmoid(b)
            skip_sum = skip_sum + block.skip_conv(g)[..., 0]
            h = block.res_conv(g)[..., 0] + h
        out.append(model.output_net(skip_sum.unsqueeze(-1))[0, 0, 0])
    return torch.stack(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    # The loudest 0.1 s of the test input (RMS 0.23); the file opens with silence,
    # where the check would only compare the responses to a zero input.
    parser.add_argument("--start", type=int, default=264_600)
    parser.add_argument("--samples", type=int, default=4_410)
    parser.add_argument("--state-dim", type=int, default=4)
    parser.add_argument("--discretization", default="free", choices=["free", "zoh"])
    args = parser.parse_args()

    model = load(args.checkpoint, args.state_dim, args.discretization)
    audio, rate = sf.read(INPUT, dtype="float64", start=args.start, frames=args.samples)
    x = torch.from_numpy(audio).reshape(1, 1, -1)
    with torch.inference_mode():
        parallel = model(x).reshape(-1)
    deviation = (parallel - recurrent(model, x)).abs().max().item()
    record = {
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "state_dim": args.state_dim,
        "discretization": args.discretization,
        "input": str(INPUT.relative_to(ROOT)),
        "sample_rate": rate,
        "start": args.start,
        "samples": args.samples,
        "max_abs_deviation": deviation,
        "output_rms": parallel.pow(2).mean().sqrt().item(),
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
