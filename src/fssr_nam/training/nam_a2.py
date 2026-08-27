"""Configuration adapter for the pinned official NAM A2 trainer."""

from __future__ import annotations

import json
from pathlib import Path


def make_configs(
    campaign: dict,
    manifest: dict,
    *,
    root: Path,
    model_config_path: Path,
) -> tuple[dict, dict, dict]:
    """Resolve one official packed A2 run without altering upstream sources."""
    files = {entry["name"]: entry for entry in manifest["files"]}
    data = {
        "common": {"delay": 0, "require_input_pre_silence": None},
        "train": {
            "x_path": str(root / files["train"]["input_path"]),
            "y_path": str(root / files["train"]["output_path"]),
            "ny": int(campaign["segment_output_samples"]),
        },
        "validation": {
            "x_path": str(root / files["validation"]["input_path"]),
            "y_path": str(root / files["validation"]["output_path"]),
            "ny": None,
        },
        "joint": [],
    }
    model = json.loads(model_config_path.read_text(encoding="utf-8"))
    learning = {
        "train_dataloader": dict(campaign["train_dataloader"]),
        "val_dataloader": dict(campaign["validation_dataloader"]),
        "trainer": {
            "accelerator": campaign["accelerator"],
            "devices": int(campaign["devices"]),
            "max_epochs": int(campaign["max_epochs"]),
            "precision": campaign["precision"],
            "deterministic": True,
            "num_sanity_val_steps": 0,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": True,
        },
        "trainer_fit_kwargs": {},
    }
    return data, model, learning
