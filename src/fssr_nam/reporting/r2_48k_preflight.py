"""Repository-only preflight for FSSR-R2-48K-v1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from fssr_nam.campaign.r2 import validate_repository_configs as validate_r2_v1
from fssr_nam.campaign.r2_48k import validate_repository_configs
from fssr_nam.inference.r2_native import build_r2_native_payload
from fssr_nam.models import AAFSSR, AAFSSRXL, AANAM

REQUIRED_MAKE_TARGETS = (
    "r2-48k-preflight",
    "r2-48k-data-audit",
    "r2-48k-mechanism",
    "r2-48k-screen",
    "r2-48k-teacher",
    "r2-48k-distill",
    "r2-48k-lock",
    "r2-48k-confirm",
    "r2-48k-benchmark",
    "r2-48k-listen",
    "r2-48k-audit",
)


def run_r2_48k_preflight(root: Path) -> dict[str, Any]:
    """Validate code/config contracts without reading any physical waveform."""
    protocol = validate_repository_configs(root)
    preserved = validate_r2_v1(root)
    required_configs = (
        "configs/data/r2_48k_archive.yaml",
        "configs/models/r2_48k/aa_nam.yaml",
        "configs/models/r2_48k/aa_fssr.yaml",
        "configs/models/r2_48k/aa_fssr_xl.yaml",
        "configs/training/r2_48k_screen.yaml",
        "configs/training/r2_48k_teacher_distill.yaml",
        "configs/training/r2_48k_confirm.yaml",
        "configs/r2_48k/mushra_exclusion.yaml",
    )
    for relative in required_configs:
        path = root / relative
        if not path.is_file() or not isinstance(
            yaml.safe_load(path.read_text(encoding="utf-8")), dict
        ):
            raise RuntimeError(f"invalid or missing R2-48K config: {relative}")
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    missing_targets = [
        target for target in REQUIRED_MAKE_TARGETS if f"{target}:" not in makefile
    ]
    if missing_targets:
        raise RuntimeError(f"missing R2-48K Make targets: {missing_targets}")

    constructors = {
        "aa-nam": lambda mode: AANAM(aa_mode=mode, root=root),
        "aa-fssr": lambda mode: AAFSSR(core_kind="mono", aa_mode=mode),
        "aa-fssr-xl": lambda mode: AAFSSRXL(aa_mode=mode),
    }
    model_checks: dict[str, Any] = {}
    with torch.inference_mode():
        for family, constructor in constructors.items():
            model_checks[family] = {}
            for mode in ("off", "adaa1", "full_island_x2", "teacher_x4"):
                model = constructor(mode)
                output = model(torch.zeros(1, 17))
                if output.shape != (1, 17) or not torch.isfinite(output).all():
                    raise RuntimeError(f"R2-48K {family}/{mode} preflight failed")
                payload = build_r2_native_payload(model)
                model_checks[family][mode] = {
                    "internal_sample_rate": payload["internal_sample_rate"],
                    "latency_samples": payload["latency_samples"],
                    "parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "finite": True,
                }
    return {
        "format": "fssr-r2-48k-preflight-v1",
        "campaign_version": protocol["campaign_version"],
        "parent_campaign_version": preserved["campaign_version"],
        "status": "passed",
        "scientific_runs_launched": 0,
        "physical_waveform_samples_read": False,
        "internal_validation_test_opened": False,
        "external_report_only_locked": True,
        "physical_192khz_reference_available": False,
        "fm9_proxy_used": False,
        "r2_v1_preserved": True,
        "make_targets": list(REQUIRED_MAKE_TARGETS),
        "model_checks": model_checks,
    }
