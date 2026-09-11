"""Versioned contracts for the serialization-corrected R2-48K-v2 campaign."""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .r2_48k import R2_48K_STAGES, validate_protocol_config

CAMPAIGN_VERSION = "FSSR-R2-48K-v2"
CAMPAIGN_KEY = "r2_48k_v2"
PARENT_CAMPAIGN_VERSION = "FSSR-R2-48K-v1"
OVERLAY_PATH = "configs/r2_48k_v2/protocol.yaml"
BASE_PROTOCOL_PATH = "configs/r2_48k/protocol.yaml"
RUN_ID_PATTERN = re.compile(
    r"^r2_48k_v2_(?P<stage>[a-z0-9-]+)_(?P<device>[a-z0-9-]+)_"
    r"(?P<family>[a-z0-9-]+)_(?P<aa>[a-z0-9_]+)_"
    r"seed(?P<seed>0|[1-9][0-9]*)_v1$"
)


class R248KV2ConfigError(RuntimeError):
    """Raised when the frozen v2 overlay or lineage contract changed."""


class R248KV2AuthorizationError(RuntimeError):
    """Raised when a v2 stage is not unlocked by literal prior evidence."""


def _require_equal(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise R248KV2ConfigError(f"{label} must equal {expected!r}, got {value!r}")


def make_run_id(stage: str, device: str, family: str, aa_mode: str, seed: int) -> str:
    """Create one immutable v2 run identifier."""
    if stage not in R2_48K_STAGES:
        raise ValueError(f"unknown R2-48K-v2 stage: {stage}")
    if seed < 0:
        raise ValueError("R2-48K-v2 seed must be non-negative")
    run_id = f"r2_48k_v2_{stage}_{device}_{family}_{aa_mode}_seed{seed}_v1"
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("R2-48K-v2 run identifier contains an invalid component")
    return run_id


def _validate_overlay(overlay: dict[str, Any]) -> None:
    literals = {
        "campaign_version": CAMPAIGN_VERSION,
        "parent_campaign": PARENT_CAMPAIGN_VERSION,
        "base_protocol_path": BASE_PROTOCOL_PATH,
        "amendment_scope": "evidence_serialization_only",
        "scientific_thresholds_changed": False,
        "fixtures_modes_probes_changed": False,
        "valid_terminal_verdicts": [
            "GO-R2-48K-v2",
            "NO-GO-R2-48K-v2",
            "INVALID",
        ],
    }
    for name, expected in literals.items():
        _require_equal(overlay.get(name), expected, f"overlay.{name}")
    encoding = overlay.get("mechanism_evidence_encoding", {})
    for name, expected in {
        "format": "fssr-extended-real-json-v1",
        "strict_standard_json": True,
        "finite_decision_aggregates_required": True,
        "retry_policy": "single_v2_mechanism_run_no_result_dependent_retry",
    }.items():
        _require_equal(encoding.get(name), expected, f"encoding.{name}")
    exact_zero = encoding.get("exact_zero_ratio", {})
    _require_equal(exact_zero.get("linear_value"), 0.0, "exact-zero linear value")
    _require_equal(
        exact_zero.get("db_representation"),
        "tagged_negative_infinity",
        "exact-zero dB representation",
    )
    boundary = overlay.get("claim_boundary", {})
    for name in (
        "physical_hardware_aliasing_claim_allowed",
        "global_state_of_the_art_claim_allowed",
        "fm9_proxy_allowed",
        "physical_192khz_reference_allowed",
    ):
        _require_equal(boundary.get(name), False, f"claim_boundary.{name}")


def resolve_protocol(root: Path) -> dict[str, Any]:
    """Resolve v2 by applying only its declared serialization overlay to v1."""
    base = yaml.safe_load((root / BASE_PROTOCOL_PATH).read_text(encoding="utf-8"))
    overlay = yaml.safe_load((root / OVERLAY_PATH).read_text(encoding="utf-8"))
    validate_protocol_config(base)
    _validate_overlay(overlay)
    resolved = deepcopy(base)
    resolved["campaign_version"] = CAMPAIGN_VERSION
    resolved["parent_campaign"] = PARENT_CAMPAIGN_VERSION
    resolved["run_id_format"] = overlay["run_id_format"]
    resolved["valid_terminal_verdicts"] = overlay["valid_terminal_verdicts"]
    resolved["decision"]["valid_failure_verdict"] = "NO-GO-R2-48K-v2"
    resolved["decision"]["instrumentation_failure_verdict"] = "INVALID"
    resolved["version_amendment"] = {
        "scope": overlay["amendment_scope"],
        "scientific_thresholds_changed": False,
        "fixtures_modes_probes_changed": False,
    }
    resolved["mechanism_evidence_encoding"] = overlay["mechanism_evidence_encoding"]
    return resolved


def validate_repository_configs(root: Path) -> dict[str, Any]:
    """Validate v2 and prove the terminal v1 record remains its parent."""
    paths = (
        root / OVERLAY_PATH,
        root / BASE_PROTOCOL_PATH,
        root / ".codex_campaign/r2_48k_v2/PROTOCOL_LOCK.yaml",
        root / ".codex_campaign/r2_48k_v2/EXTERNAL_FREEZE.json",
        root / ".codex_campaign/r2_48k/VERDICT.json",
        root / ".codex_campaign/LINEAGES.json",
    )
    for path in paths:
        if not path.is_file():
            raise R248KV2ConfigError(f"missing canonical R2-48K-v2 file: {path}")
    resolved = resolve_protocol(root)
    lock = yaml.safe_load(paths[2].read_text(encoding="utf-8"))
    freeze = yaml.safe_load(paths[3].read_text(encoding="utf-8"))
    parent_verdict = yaml.safe_load(paths[4].read_text(encoding="utf-8"))
    lineages = yaml.safe_load(paths[5].read_text(encoding="utf-8"))
    for name, expected in {
        "campaign_version": CAMPAIGN_VERSION,
        "parent_campaign": PARENT_CAMPAIGN_VERSION,
        "status": "frozen_before_first_scientific_run",
        "scientific_runs_launched": 0,
        "amendment_scope": "evidence_serialization_only",
        "physical_192khz_dataset_available": False,
        "fm9_proxy_allowed": False,
        "external_report_only_locked": True,
    }.items():
        _require_equal(lock.get(name), expected, f"lock.{name}")
    for name, expected in {
        "campaign_version": CAMPAIGN_VERSION,
        "internal_validation_outputs_locked": True,
        "internal_validation_tests_locked": True,
        "internal_validation_test_open_count": 0,
        "external_report_only_locked": True,
        "external_results_accessed": False,
    }.items():
        _require_equal(freeze.get(name), expected, f"freeze.{name}")
    _require_equal(parent_verdict.get("verdict"), "INVALID", "parent verdict")
    _require_equal(
        parent_verdict.get("x2_gate_evaluated"), False, "parent x2 gate status"
    )
    active = (root / ".codex_campaign/ACTIVE_CAMPAIGN").read_text().strip()
    _require_equal(lineages.get("active"), active, "lineage active campaign")
    if active not in lineages.get("lineages", {}):
        raise R248KV2ConfigError("active lineage entry is missing")
    entry = lineages.get("lineages", {}).get(CAMPAIGN_KEY, {})
    _require_equal(entry.get("campaign_version"), CAMPAIGN_VERSION, "lineage version")
    _require_equal(entry.get("parent"), "r2_48k", "lineage parent")
    _require_equal(
        entry.get("historical_artifacts_immutable"), True, "lineage immutability"
    )
    return resolved


_STAGE_REQUIREMENTS = {
    "data": ("preflight", "passed"),
    "mechanism": ("data", "passed"),
    "screen": ("mechanism_x2", "passed"),
}


def validate_stage_authorization(stage: str, decisions: dict[str, str]) -> None:
    """Fail closed unless the literal preceding v2 gate passed."""
    if stage == "preflight":
        return
    try:
        name, expected = _STAGE_REQUIREMENTS[stage]
    except KeyError as error:
        raise ValueError(
            f"unsupported R2-48K-v2 authorization stage: {stage}"
        ) from error
    if decisions.get(name) != expected:
        raise R248KV2AuthorizationError(
            f"R2-48K-v2 {stage} requires {name}={expected}; got {decisions.get(name)!r}"
        )
