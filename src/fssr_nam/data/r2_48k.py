"""Metadata-only audit of the reused 48 kHz physical archives."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from fssr_nam.campaign.r2_48k import (
    CAMPAIGN_VERSION,
    DEVELOPMENT_DEVICES,
    PRIMARY_DEVICES,
)

EXPECTED_COUNTS = {
    "fulltone": {"train": 1, "validation": 1, "test": 1},
    "bigmuff": {"train": 1, "validation": 1, "test": 1},
    "blackstar": {"train": 1, "validation": 1, "test": 1},
    "ua1176": {"train": 5, "validation": 2, "test": 2},
}
EXPECTED_SECONDS = {"train": 120.0, "validation": 30.0, "test": 30.0}


class R248KDataAuditError(ValueError):
    """Raised when the archive inventory violates the frozen 48 kHz contract."""


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise R248KDataAuditError(f"missing {label}: {path}")
    try:
        if path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise R248KDataAuditError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise R248KDataAuditError(f"{label} must contain a mapping")
    return value


def audit_r2_48k_archive(
    root: Path, *, require_prepared_files: bool = True
) -> dict[str, Any]:
    """Audit manifest metadata and path presence without reading waveform samples."""
    config_path = root / "configs/data/r2_48k_archive.yaml"
    config = _load_mapping(config_path, "R2-48K data config")
    if config.get("campaign_version") != CAMPAIGN_VERSION:
        raise R248KDataAuditError("R2-48K data campaign version changed")
    if config.get("target_sample_rate_hz") != 48_000:
        raise R248KDataAuditError("R2-48K data must remain at 48 kHz")
    for field in (
        "metadata_audit_reads_waveform_samples",
        "new_capture_required",
        "additional_resampling_allowed",
        "independent_normalization_allowed",
        "proxy_hardware_allowed",
    ):
        if config.get(field) is not False:
            raise R248KDataAuditError(f"R2-48K data {field} must be false")
    if config.get("prohibited_devices") != ["fractal_fm9"]:
        raise R248KDataAuditError("FM9 must remain an explicitly prohibited proxy")

    manifest_relative = config.get("source_manifest")
    if not isinstance(manifest_relative, str) or not manifest_relative:
        raise R248KDataAuditError("R2-48K source_manifest must be a path")
    manifest = _load_mapping(root / manifest_relative, "R2-48K source manifest")
    if manifest.get("campaign_version") != "FSSR-R1-v1":
        raise R248KDataAuditError("source manifest provenance campaign changed")
    if manifest.get("target_sample_rate") != 48_000:
        raise R248KDataAuditError("source manifest target rate is not 48 kHz")
    if manifest.get("external_report_only_accessed") is not False:
        raise R248KDataAuditError("EXTERNAL_REPORT_ONLY must remain unaccessed")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise R248KDataAuditError("source manifest files must be a non-empty list")

    expected_devices = {*DEVELOPMENT_DEVICES, *PRIMARY_DEVICES}
    counts: dict[str, dict[str, int]] = {
        device: defaultdict(int) for device in expected_devices
    }
    samples: dict[str, dict[str, int]] = {
        device: defaultdict(int) for device in expected_devices
    }
    source_splits: dict[str, dict[str, str]] = {
        device: {} for device in expected_devices
    }
    missing_paths: list[str] = []
    for entry in files:
        if not isinstance(entry, dict):
            raise R248KDataAuditError("source manifest file entries must be mappings")
        device = entry.get("device")
        split = entry.get("split")
        if device not in expected_devices or split not in EXPECTED_SECONDS:
            raise R248KDataAuditError(
                f"unexpected R2-48K device/split: {device!r}/{split!r}"
            )
        expected_tier = (
            "INTERNAL_DEV" if device in DEVELOPMENT_DEVICES else "INTERNAL_VALIDATION"
        )
        if entry.get("tier") != expected_tier:
            raise R248KDataAuditError(f"{device} data tier changed")
        if entry.get("sample_rate") != 48_000:
            raise R248KDataAuditError(f"{device}/{split} is not prepared at 48 kHz")
        source_rate = entry.get("source_rate")
        expected_resampling = (
            "none"
            if source_rate == 48_000
            else "joint_resample_poly_kaiser_beta_8.6"
            if source_rate == 44_100
            else None
        )
        if (
            expected_resampling is None
            or entry.get("resampling") != expected_resampling
        ):
            raise R248KDataAuditError(
                f"{device}/{split} does not use the frozen preparation chain"
            )
        if entry.get("normalization") != "none":
            raise R248KDataAuditError(f"{device}/{split} was independently normalized")
        sample_count = entry.get("samples")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int):
            raise R248KDataAuditError(f"{device}/{split} sample count is invalid")
        source_id = entry.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise R248KDataAuditError(f"{device}/{split} source_id is invalid")
        prior_split = source_splits[device].get(source_id)
        if prior_split is not None and prior_split != split:
            raise R248KDataAuditError(
                f"{device} source {source_id!r} crosses split boundaries"
            )
        source_splits[device][source_id] = str(split)
        counts[device][str(split)] += 1
        samples[device][str(split)] += sample_count
        for path_field in ("input_path", "target_path"):
            relative = entry.get(path_field)
            if not isinstance(relative, str) or not relative.startswith(
                "datasets/raw/internal_r1/"
            ):
                raise R248KDataAuditError(
                    f"{device}/{split} {path_field} escaped the prepared archive"
                )
            if require_prepared_files and not (root / relative).is_file():
                missing_paths.append(relative)
    if missing_paths:
        raise R248KDataAuditError(
            f"missing prepared R2-48K files: {sorted(missing_paths)}"
        )

    durations: dict[str, dict[str, float]] = {}
    for device in sorted(expected_devices):
        observed_counts = dict(counts[device])
        if observed_counts != EXPECTED_COUNTS[device]:
            raise R248KDataAuditError(
                f"{device} split counts changed: {observed_counts}"
            )
        durations[device] = {}
        for split, expected_seconds in EXPECTED_SECONDS.items():
            seconds = samples[device][split] / 48_000
            if seconds != expected_seconds:
                raise R248KDataAuditError(
                    f"{device}/{split} duration must equal {expected_seconds}s"
                )
            durations[device][split] = seconds

    roles = config.get("roles")
    if not isinstance(roles, dict) or set(roles) != expected_devices:
        raise R248KDataAuditError("R2-48K device role mapping changed")
    for device in DEVELOPMENT_DEVICES:
        if roles[device].get("evidence_role") != "historical_development":
            raise R248KDataAuditError(f"{device} must remain historical development")
        if roles[device].get("test_is_prospectively_sealed") is not False:
            raise R248KDataAuditError(f"{device} test cannot be relabelled as sealed")
    for device in PRIMARY_DEVICES:
        if roles[device].get("evidence_role") != "prospective_primary":
            raise R248KDataAuditError(f"{device} must remain prospective primary")
        if roles[device].get("test_is_prospectively_sealed") is not True:
            raise R248KDataAuditError(f"{device} test must remain sealed")

    return {
        "format": "fssr-r2-48k-data-audit-v1",
        "campaign_version": CAMPAIGN_VERSION,
        "status": "passed",
        "source_manifest": manifest_relative,
        "sample_rate_hz": 48_000,
        "source_sample_rates_hz": [44_100, 48_000],
        "additional_resampling_performed": False,
        "devices": sorted(expected_devices),
        "file_pairs": len(files),
        "split_counts": {device: dict(counts[device]) for device in sorted(counts)},
        "durations_seconds": durations,
        "development_evidence_role": "historical_development",
        "primary_evidence_role": "prospective_internal_validation",
        "development_tests_previously_observed": True,
        "internal_validation_test_waveforms_opened": False,
        "waveform_samples_read": False,
        "metadata_only": True,
        "physical_192khz_reference_available": False,
        "fm9_proxy_used": False,
        "external_report_only_locked": True,
    }
