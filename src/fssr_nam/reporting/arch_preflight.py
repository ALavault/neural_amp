"""Metadata-only preflight for AMP-QUALITY-ARCH-v1."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from fssr_nam.campaign.amp_quality_arch_v1 import (
    CAMPAIGN_PATH,
    CANDIDATES,
    COMPARATORS,
    TRAINING_CANDIDATES,
    loss_qualification_specs,
    validate_repository_state,
)
from fssr_nam.models import DEPLOYMENT_PROFILES, build_arch_v1_candidate
from fssr_nam.models.sota_comparators import build_sota_comparator


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_source_pins(root: Path, protocol: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, pin in protocol["source_pins"].items():
        path_text = pin.get("path")
        if path_text is None:
            continue
        path = root / path_text
        if not path.is_dir():
            raise RuntimeError(f"missing pinned source directory: {path}")
        observed[name] = _git_head(path)
        if observed[name] != pin["commit"]:
            raise RuntimeError(
                f"source pin mismatch for {name}: {observed[name]} != {pin['commit']}"
            )
    return observed


def _validate_data_metadata(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    manifest_path = root / protocol["data"]["manifest"]
    split_path = root / protocol["data"]["split_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise RuntimeError("physical manifest has no file records")
    expected_devices = {"fulltone", "bigmuff", "blackstar", "ua1176"}
    observed_devices = {record.get("device") for record in records}
    if observed_devices != expected_devices:
        raise RuntimeError(f"physical device set changed: {observed_devices}")
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        device = str(record["device"])
        split_name = str(record["split"])
        counts.setdefault(device, {}).setdefault(split_name, 0)
        counts[device][split_name] += 1
        for field in ("input_path", "target_path"):
            path = root / record[field]
            if not path.is_file():
                raise RuntimeError(
                    f"missing prepared audio path declared by metadata: {path}"
                )
        if record.get("sample_rate") != 48_000:
            raise RuntimeError(f"{device}/{split_name} is not prepared at 48 kHz")
    for device in expected_devices:
        if set(counts[device]) != {"train", "validation", "test"}:
            raise RuntimeError(f"{device} does not have train/validation/test metadata")
    if split.get("path_overlap") != [] or split.get("leakage_check") != "passed":
        raise RuntimeError("frozen split leakage audit no longer passes")
    return {
        "manifest": str(protocol["data"]["manifest"]),
        "split_manifest": str(protocol["data"]["split_manifest"]),
        "device_split_file_counts": counts,
        "audio_samples_read": 0,
    }


def _validate_model_configs(root: Path) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for family in CANDIDATES:
        path = root / "configs/models/arch_v1" / f"{family}.yaml"
        if not path.is_file():
            raise RuntimeError(f"missing frozen candidate config: {path}")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if config.get("campaign_version") != "AMP-QUALITY-ARCH-v1":
            raise RuntimeError(f"candidate config has wrong campaign: {path}")
        if config.get("family") != family:
            raise RuntimeError(f"candidate config family mismatch: {path}")
        deployable = config.get("deployable", {})
        if deployable.get("training_eligible") != (family in TRAINING_CANDIDATES):
            raise RuntimeError(f"candidate training eligibility mismatch: {path}")
        if deployable.get("profile_selection") != (
            "largest_profile_passing_blind_native_skeleton_gate"
        ):
            raise RuntimeError(f"candidate profile selection mismatch: {path}")
        if deployable.get("selected_profile") != "max":
            raise RuntimeError(f"candidate selected profile mismatch: {path}")
        if deployable.get("selection_evidence") != (
            ".codex_campaign/amp_quality_arch_v1/NATIVE_SKELETON.json"
        ):
            raise RuntimeError(f"candidate selection evidence mismatch: {path}")
        if list(deployable.get("profiles", {})) != list(DEPLOYMENT_PROFILES):
            raise RuntimeError(f"candidate width profiles mismatch: {path}")
        profiles: dict[str, dict[str, Any]] = {}
        parameter_counts: list[int] = []
        for profile in DEPLOYMENT_PROFILES:
            model = build_arch_v1_candidate(family, profile=profile)
            signal = torch.linspace(-0.5, 0.5, 65)
            output = model(signal)
            if output.shape != signal.shape or not torch.isfinite(output).all():
                raise RuntimeError(
                    f"candidate skeleton failed finite shape smoke: {family}/{profile}"
                )
            parameters = sum(parameter.numel() for parameter in model.parameters())
            parameter_counts.append(parameters)
            profiles[profile] = {
                "parameters": parameters,
                "latency_samples": int(model.latency_samples),
                "aa_mode": str(model.aa_mode),
            }
        if parameter_counts != sorted(set(parameter_counts)):
            raise RuntimeError(f"candidate widths are not strictly ordered: {family}")
        configs[family] = {
            "path": str(path.relative_to(root)),
            "profiles": profiles,
        }
    return configs


def _validate_comparator_configs(
    root: Path, protocol: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    latencies = protocol["comparators"]["conservative_causal_latency_samples"]
    for family in COMPARATORS:
        path = root / "configs/models/arch_v1" / f"comparator_{family}.yaml"
        if not path.is_file():
            raise RuntimeError(f"missing frozen comparator config: {path}")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if config.get("campaign_version") != "AMP-QUALITY-ARCH-v1":
            raise RuntimeError(f"comparator config has wrong campaign: {path}")
        if config.get("family") != family:
            raise RuntimeError(f"comparator config family mismatch: {path}")
        variants: dict[str, dict[str, int]] = {}
        for variant in protocol["comparators"]["variants"][family]:
            model = build_sota_comparator(family, variant=variant, root=root)
            declaration = (
                config if variant == "official" else config["variants"][variant]
            )
            parameters = sum(parameter.numel() for parameter in model.parameters())
            if declaration.get("parameters") != parameters:
                raise RuntimeError(f"comparator parameter mismatch: {family}/{variant}")
            latency = int(getattr(model, "latency_samples", 0))
            if latencies[family][variant] != latency:
                raise RuntimeError(f"comparator latency mismatch: {family}/{variant}")
            signal = torch.linspace(-0.5, 0.5, 65)
            output = model(signal)
            if output.shape != signal.shape or not torch.isfinite(output).all():
                raise RuntimeError(
                    f"comparator failed finite shape smoke: {family}/{variant}"
                )
            variants[variant] = {
                "parameters": parameters,
                "latency_samples": latency,
            }
        reports[family] = {
            "path": str(path.relative_to(root)),
            "variants": variants,
        }
    return reports


def build_preflight_report(
    root: Path, *, require_frozen: bool = False
) -> dict[str, Any]:
    """Validate metadata and code without opening any physical audio sample."""
    protocol = validate_repository_state(root, require_frozen=require_frozen)
    maturity = json.loads(
        (root / CAMPAIGN_PATH / "MATURITY.json").read_text(encoding="utf-8")
    )
    if maturity.get("scientific_runs_launched") != 0:
        raise RuntimeError("preflight requires zero prior scientific runs")
    required_documents = (
        "SOTA_SNAPSHOT.md",
        "SIDE_REPORT_AUDIT.md",
        "TECHNICAL_AUDIT.md",
        "PROTOCOL.md",
    )
    missing = [
        name
        for name in required_documents
        if not (root / CAMPAIGN_PATH / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"missing architecture campaign documents: {missing}")
    if not torch.cuda.is_available():
        raise RuntimeError("frozen 24 GB CUDA training device is unavailable")
    properties = torch.cuda.get_device_properties(0)
    if properties.total_memory < 23 * 1024**3:
        raise RuntimeError("CUDA device has less than the frozen 23 GiB minimum")
    report = {
        "schema_version": 1,
        "campaign_version": "AMP-QUALITY-ARCH-v1",
        "stage": "preflight",
        "status": "passed",
        "protocol_status": protocol["status"],
        "scientific_runs_launched_before_preflight": 0,
        "source_pins": _validate_source_pins(root, protocol),
        "data": _validate_data_metadata(root, protocol),
        "comparators": _validate_comparator_configs(root, protocol),
        "candidate_skeletons": _validate_model_configs(root),
        "loss_qualification_trajectories": len(loss_qualification_specs(protocol)),
        "hardware": {
            "cuda_device": properties.name,
            "cuda_total_memory_bytes": properties.total_memory,
        },
        "sealed_outputs_accessed": {"blackstar": False, "ua1176": False},
        "external_report_only_locked": True,
    }
    return report
