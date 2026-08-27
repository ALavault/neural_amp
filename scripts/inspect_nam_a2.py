#!/usr/bin/env python3
"""Inspect the pinned official A2 config rather than relying on descriptions."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from nam.train.lightning_module import PackedLightningModule

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "third_party/neural-amp-modeler/nam/train/_resources/config_model_packed.json"
)
OUT_DIR = ROOT / "experiments/summaries/m2_a2_architecture"
BENCHMARK_BLOCK_SIZES = (1, 16, 64, 128)


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def fast_path_state_bytes(
    channels: int, kernel_sizes: list[int], dilations: list[int], block_size: int
) -> int:
    work_floats = (3 * channels + 2) * block_size
    layer_floats = sum(
        channels
        * (next_power_of_two((kernel_size - 1) * dilation + block_size) + block_size)
        for kernel_size, dilation in zip(kernel_sizes, dilations, strict=True)
    )
    head_floats = channels * (next_power_of_two(15 + block_size) + block_size)
    return 4 * (work_floats + layer_floats + head_floats)


def main() -> None:
    config_bytes = CONFIG_PATH.read_bytes()
    config = json.loads(config_bytes)
    module = PackedLightningModule.init_from_config(config)
    submodels = []
    kernel_sizes = config["net"]["config"]["submodels"][1]["config"]["layers_configs"][
        0
    ]["kernel_sizes"]
    dilations = config["net"]["config"]["submodels"][1]["config"]["layers_configs"][0][
        "dilations"
    ]
    for index, entry in enumerate(config["net"]["config"]["submodels"]):
        model = module.net.extract_submodel(index)
        exported = model._get_export_dict()
        channels = entry["config"]["layers_configs"][0]["channels"]
        linear_macs = (
            channels
            + sum(
                channels * channels * kernel_size + channels + channels * channels
                for kernel_size in kernel_sizes
            )
            + 16 * channels
            + 1
        )
        submodels.append(
            {
                "name": entry["name"],
                "channels": channels,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "exported_weight_count": len(exported["weights"]),
                "receptive_field_samples": model.receptive_field,
                "receptive_field_ms_at_48khz": model.receptive_field / 48.0,
                "export_architecture": exported["architecture"],
                "linear_macs_per_sample": linear_macs,
                "minimum_linear_flops_per_sample": 2 * linear_macs,
                "fast_path_float_buffer_bytes_by_block": {
                    str(block_size): fast_path_state_bytes(
                        channels, kernel_sizes, dilations, block_size
                    )
                    for block_size in BENCHMARK_BLOCK_SIZES
                },
            }
        )
    layer = config["net"]["config"]["submodels"][1]["config"]["layers_configs"][0]
    report = {
        "schema_version": 1,
        "source": str(CONFIG_PATH.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "trainer_commit": git_commit(ROOT / "third_party/neural-amp-modeler"),
        "core_commit": git_commit(ROOT / "third_party/NeuralAmpModelerCore"),
        "packed_trainable_parameters": sum(
            parameter.numel() for parameter in module.parameters()
        ),
        "num_layers": len(layer["dilations"]),
        "kernel_sizes": layer["kernel_sizes"],
        "dilations": layer["dilations"],
        "activation": layer["activation"],
        "gated": layer["gated"],
        "head": layer["head"],
        "head_scale": config["net"]["config"]["submodels"][1]["config"]["head_scale"],
        "loss": config["loss"],
        "optimizer": config["optimizer"],
        "lr_scheduler": config["lr_scheduler"],
        "submodels": submodels,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "architecture.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
