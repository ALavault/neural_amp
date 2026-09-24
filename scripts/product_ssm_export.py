"""Export unconditioned, gated SSM-WaveNet to the native v1 FOH format."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from torch import nn

from fssr_nam.models.ssm_wavenet import SSMWaveNet, zoh_to_foh

ROOT = Path(__file__).resolve().parents[1]


def export_model(model: SSMWaveNet, out: Path, *, source: dict | None = None) -> Path:
    """Export a CPU copy; preserve the caller's parameters and discretization."""
    if model.num_controls or model.film_net is not None or model.pole_net is not None:
        raise ValueError("native v1 does not support knob conditioning")
    if not model.blocks or any(b.act_type != "gated" for b in model.blocks):
        raise ValueError("native v1 requires gated blocks")
    if any(b.ssm.discretization not in ("zoh", "foh") for b in model.blocks):
        raise ValueError("native v1 requires zoh or foh discretization")
    if model.input_conv.in_channels != 1 or model.output_net[3].out_channels != 1:
        raise ValueError("native v1 requires mono input and output")
    model = copy.deepcopy(model).cpu().eval()
    zoh_to_foh(model)

    def conv(layer: nn.Conv1d) -> dict:
        return {
            "weight": layer.weight.detach().squeeze(-1).tolist(),
            "bias": layer.bias.detach().tolist(),
        }

    blocks = []
    for block in model.blocks:
        ssm = block.ssm
        blocks.append(
            {
                "ssm": {
                    "log_dt": ssm.log_dt.detach().double().tolist(),
                    "lambda_re": (-ssm.log_A_real.detach().double().exp()).tolist(),
                    "lambda_im": ssm.A_imag.detach().double().tolist(),
                    "c_re": ssm.C.detach()[..., 0].double().tolist(),
                    "c_im": ssm.C.detach()[..., 1].double().tolist(),
                    "d": ssm.D.detach().tolist(),
                },
                "pre_gate": conv(block.pre_gate),
                "res_conv": conv(block.res_conv),
                "skip_conv": conv(block.skip_conv),
            }
        )
    output_act = {nn.Tanh: "tanh", nn.Softsign: "softsign", nn.Identity: "none"}[
        type(model.output_net[4])
    ]
    payload = {
        "format": "ssm-wavenet-native-v1",
        "version": 1,
        "precision": "float32",
        "trained_sample_rate_hz": 48000,
        "latency_samples": 0,
        "source": source or {},
        "architecture": {
            "num_blocks": len(blocks),
            "channels": model.blocks[0].ssm.channels,
            "state_dim": model.blocks[0].ssm.state_dim,
            "act_type": "gated",
            "output_act": output_act,
            "discretization": "foh",
        },
        "input_conv": {
            "weight": model.input_conv.weight.detach()[:, 0, 0].tolist(),
            "bias": model.input_conv.bias.detach().tolist(),
        },
        "blocks": blocks,
        "output_net": {
            "conv1": conv(model.output_net[1]),
            "conv2": conv(model.output_net[3]),
        },
    }
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", choices=("last", "best"), default="last")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    record = json.loads(
        (ROOT / "demo/nablafx_bench" / f"{args.run_id}.json").read_text()
    )
    ssm = record.get("ssm") or {}
    if record["model"] != "ssm-wavenet":
        raise ValueError("record must be ssm-wavenet")
    if ssm.get("discretization") not in ("zoh", "foh"):
        raise ValueError("record must use zoh or foh")
    if ssm.get("num_controls", 0) or record.get("num_controls", 0):
        raise ValueError("native v1 does not support knob conditioning")
    from product_nablafx_bench import DEVICES
    from product_sr_transfer import load_model

    device = ssm.get("device", record.get("device", "bigmuff"))
    if DEVICES.get(device, {}).get("params"):
        raise ValueError("native v1 does not support knob conditioning")

    wrapper, record = load_model(args.run_id, args.checkpoint)
    model = next(m for m in wrapper.modules() if isinstance(m, SSMWaveNet))
    export_model(
        model,
        args.out,
        source={
            "run_id": args.run_id,
            "checkpoint": args.checkpoint,
            "commit": record["commit"],
        },
    )


if __name__ == "__main__":
    main()
