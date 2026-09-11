"""Repository-only R2 preflight; it consumes no capture or holdout samples."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from fssr_nam.campaign.r2 import validate_repository_configs
from fssr_nam.inference.r2_native import build_r2_native_payload
from fssr_nam.models import AAFSSR, AANAM

REQUIRED_MAKE_TARGETS = (
    "r2-preflight",
    "r2-capture-audit",
    "r2-mechanism",
    "r2-screen",
    "r2-teacher",
    "r2-distill",
    "r2-lock",
    "r2-confirm",
    "r2-benchmark",
    "r2-listen",
    "r2-audit",
)


def run_r2_preflight(root: Path) -> dict[str, Any]:
    protocol = validate_repository_configs(root)
    for relative in (
        "configs/data/r2_capture.yaml",
        "configs/models/r2/aa_nam.yaml",
        "configs/models/r2/aa_fssr.yaml",
        "configs/training/r2_screen.yaml",
        "configs/training/r2_teacher_distill.yaml",
        "configs/training/r2_confirm.yaml",
        "configs/r2/mushra_exclusion.yaml",
    ):
        path = root / relative
        if not path.is_file() or not isinstance(
            yaml.safe_load(path.read_text(encoding="utf-8")), dict
        ):
            raise RuntimeError(f"invalid or missing R2 configuration: {relative}")
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    missing_targets = [
        target for target in REQUIRED_MAKE_TARGETS if f"{target}:" not in makefile
    ]
    if missing_targets:
        raise RuntimeError(f"missing R2 Make targets: {missing_targets}")
    r1_changes = subprocess.run(
        ["git", "status", "--porcelain", "--", ".codex_campaign/r1"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if r1_changes:
        raise RuntimeError("R1 canonical history has uncommitted modifications")
    model_checks: dict[str, Any] = {"aa-fssr": {}, "aa-nam": {}}
    with torch.inference_mode():
        for mode in ("off", "adaa1", "full_island_x2", "teacher_x4"):
            model = AAFSSR(core_kind="mono", aa_mode=mode)
            output = model(torch.zeros(1, 17))
            if output.shape != (1, 17) or not torch.isfinite(output).all():
                raise RuntimeError(f"R2 {mode} model preflight failed")
            payload = build_r2_native_payload(model)
            model_checks["aa-fssr"][mode] = {
                "internal_sample_rate": payload["internal_sample_rate"],
                "latency_samples": payload["latency_samples"],
                "finite": True,
            }
            nam_model = AANAM(aa_mode=mode, root=root)
            nam_payload = build_r2_native_payload(nam_model)
            model_checks["aa-nam"][mode] = {
                "internal_sample_rate": nam_payload["internal_sample_rate"],
                "latency_samples": nam_payload["latency_samples"],
                "activation_count": nam_model.activation_count,
                "finite": all(
                    torch.isfinite(parameter).all()
                    for parameter in nam_model.parameters()
                ),
            }
    return {
        "format": "fssr-r2-preflight-v1",
        "campaign_version": protocol["campaign_version"],
        "status": "passed",
        "scientific_runs_launched": 0,
        "capture_samples_read": False,
        "sealed_test_opened": False,
        "external_report_only_locked": True,
        "r1_historical_artifacts_modified": False,
        "make_targets": list(REQUIRED_MAKE_TARGETS),
        "model_checks": model_checks,
    }
