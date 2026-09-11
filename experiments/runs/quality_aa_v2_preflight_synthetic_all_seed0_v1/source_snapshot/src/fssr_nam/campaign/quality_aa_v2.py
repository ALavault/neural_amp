"""Transport-corrected prospective contract for QUALITY-AA-v2."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .quality_aa_v1 import validate_protocol_config as validate_v1_protocol

CAMPAIGN_VERSION = "FSSR-QUALITY-AA-v2"
CAMPAIGN_KEY = "quality_aa_v2"
PARENT_CAMPAIGN_VERSION = "FSSR-QUALITY-AA-v1"
BASE_PROTOCOL_PATH = "configs/quality_aa_v1/protocol.yaml"
OVERLAY_PATH = "configs/quality_aa_v2/protocol.yaml"
CAMPAIGN_PATH = ".codex_campaign/quality_aa_v2"
STAGES = ("preflight", "mechanism", "native", "audit")
RUN_ID_PATTERN = re.compile(
    r"^quality_aa_v2_(?P<stage>preflight|mechanism|native|audit)_"
    r"synthetic_(?P<route>[a-z0-9_]+)_seed0_v1$"
)


class QualityAAV2ConfigError(RuntimeError):
    """Raised when the v2 transport-only contract changed."""


class QualityAAV2AuthorizationError(RuntimeError):
    """Raised when a v2 stage lacks literal prior authorization."""


def _require_equal(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise QualityAAV2ConfigError(f"{label} must equal {expected!r}, got {value!r}")


def make_run_id(stage: str, route: str = "all") -> str:
    if stage not in STAGES:
        raise ValueError(f"unknown QUALITY-AA-v2 stage: {stage}")
    run_id = f"quality_aa_v2_{stage}_synthetic_{route}_seed0_v1"
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("QUALITY-AA-v2 run identifier contains an invalid component")
    return run_id


def _validate_string_keys(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                location = "/".join(path) or "<root>"
                raise QualityAAV2ConfigError(
                    f"mapping key at {location} must be a string, got {key!r}"
                )
            _validate_string_keys(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_string_keys(item, (*path, str(index)))


def resolve_protocol(root: Path) -> dict[str, Any]:
    """Normalize only the known YAML-1.1 ``off`` key from the frozen v1 base."""
    base = yaml.safe_load((root / BASE_PROTOCOL_PATH).read_text(encoding="utf-8"))
    overlay = yaml.safe_load((root / OVERLAY_PATH).read_text(encoding="utf-8"))
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        raise QualityAAV2ConfigError("v2 base and overlay must be mappings")
    validate_v1_protocol(base)
    for name, expected in {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "campaign_key": CAMPAIGN_KEY,
        "parent_campaign": PARENT_CAMPAIGN_VERSION,
        "base_protocol_path": BASE_PROTOCOL_PATH,
        "amendment_scope": "yaml_key_typing_and_strict_mapping_validation_only",
        "scientific_thresholds_changed": False,
        "fixtures_routes_probes_changed": False,
        "parent_numeric_evidence_reused": False,
    }.items():
        _require_equal(overlay.get(name), expected, f"overlay.{name}")
    routes = base.get("routes")
    if not isinstance(routes, dict) or False not in routes or "off" in routes:
        raise QualityAAV2ConfigError("v1 base no longer has the known boolean off key")
    resolved = deepcopy(base)
    resolved_routes = resolved["routes"]
    resolved_routes["off"] = resolved_routes.pop(False)
    resolved["campaign_version"] = CAMPAIGN_VERSION
    resolved["campaign_key"] = CAMPAIGN_KEY
    resolved["parent_campaign"] = PARENT_CAMPAIGN_VERSION
    resolved["run_id_format"] = overlay["run_id_format"]
    resolved["transport_amendment"] = deepcopy(overlay["transport"])
    resolved["version_amendment"] = {
        "scope": overlay["amendment_scope"],
        "scientific_thresholds_changed": False,
        "fixtures_routes_probes_changed": False,
        "parent_numeric_evidence_reused": False,
    }
    _validate_string_keys(resolved)
    return resolved


def validate_repository_configs(root: Path) -> dict[str, Any]:
    required = (
        root / BASE_PROTOCOL_PATH,
        root / OVERLAY_PATH,
        root / CAMPAIGN_PATH / "PROTOCOL_LOCK.yaml",
        root / CAMPAIGN_PATH / "EXTERNAL_FREEZE.json",
        root / ".codex_campaign/LINEAGES.json",
        root / ".codex_campaign/quality_aa_v1/VERDICT.json",
    )
    for path in required:
        if not path.is_file():
            raise QualityAAV2ConfigError(f"missing canonical v2 file: {path}")
    resolved = resolve_protocol(root)
    lock = yaml.safe_load(required[2].read_text(encoding="utf-8"))
    for name, expected in {
        "campaign_version": CAMPAIGN_VERSION,
        "campaign_key": CAMPAIGN_KEY,
        "parent_campaign": PARENT_CAMPAIGN_VERSION,
        "base_protocol_path": BASE_PROTOCOL_PATH,
        "overlay_protocol_path": OVERLAY_PATH,
        "status": "frozen_before_first_scientific_run",
        "scientific_runs_launched": 0,
        "amendment_scope": "yaml_key_typing_and_strict_mapping_validation_only",
        "scientific_thresholds_changed": False,
        "fixtures_routes_probes_changed": False,
        "parent_numeric_evidence_reused": False,
        "parent_artifacts_immutable": True,
        "parent_verdict": "INVALID",
        "parent_candidate_route_outputs_observed": False,
        "source_snapshot_required_for_every_run": True,
    }.items():
        _require_equal(lock.get(name), expected, f"lock.{name}")
    freeze = json.loads(required[3].read_text(encoding="utf-8"))
    lineages = json.loads(required[4].read_text(encoding="utf-8"))
    parent = json.loads(required[5].read_text(encoding="utf-8"))
    _require_equal(freeze.get("physical_audio_samples_read"), 0, "physical audio")
    _require_equal(freeze.get("external_report_only_locked"), True, "external lock")
    _require_equal(parent.get("status"), "INVALID", "parent verdict")
    _require_equal(
        parent.get("candidate_route_outputs_observed"), False, "parent candidates"
    )
    _require_equal(
        (root / ".codex_campaign/ACTIVE_CAMPAIGN").read_text().strip(),
        CAMPAIGN_KEY,
        "active campaign",
    )
    _require_equal(lineages.get("active"), CAMPAIGN_KEY, "lineage active")
    entry = lineages.get("lineages", {}).get(CAMPAIGN_KEY, {})
    _require_equal(entry.get("campaign_version"), CAMPAIGN_VERSION, "lineage")
    _require_equal(entry.get("parent"), "quality_aa_v1", "lineage parent")
    return resolved


_STAGE_REQUIREMENTS = {
    "mechanism": ("preflight", "passed"),
    "native": ("mechanism", "passed"),
    "audit": ("native", "passed"),
}


def validate_stage_authorization(stage: str, decisions: dict[str, str]) -> None:
    if stage == "preflight":
        return
    if stage not in _STAGE_REQUIREMENTS:
        raise ValueError(f"unsupported QUALITY-AA-v2 stage: {stage}")
    name, expected = _STAGE_REQUIREMENTS[stage]
    if decisions.get(name) != expected:
        raise QualityAAV2AuthorizationError(
            f"{stage} requires {name}={expected}; got {decisions.get(name)!r}"
        )
