"""Versioned contracts for the prospective QUALITY-AA-v1 lineage."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

CAMPAIGN_VERSION = "FSSR-QUALITY-AA-v1"
CAMPAIGN_KEY = "quality_aa_v1"
PARENT_CAMPAIGN_VERSION = "FSSR-R2-48K-v2"
PROTOCOL_PATH = "configs/quality_aa_v1/protocol.yaml"
CAMPAIGN_PATH = ".codex_campaign/quality_aa_v1"
STAGES = ("preflight", "mechanism", "native", "audit")
RUN_ID_PATTERN = re.compile(
    r"^quality_aa_v1_(?P<stage>preflight|mechanism|native|audit)_"
    r"synthetic_(?P<route>[a-z0-9_]+)_seed0_v1$"
)


class QualityAAConfigError(RuntimeError):
    """Raised when the frozen prospective contract changed."""


class QualityAAAuthorizationError(RuntimeError):
    """Raised when a stage lacks literal prior gate authorization."""


def _require_equal(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise QualityAAConfigError(f"{label} must equal {expected!r}, got {value!r}")


def make_run_id(stage: str, route: str = "all") -> str:
    """Create an immutable single-seed run identifier."""
    if stage not in STAGES:
        raise ValueError(f"unknown QUALITY-AA stage: {stage}")
    run_id = f"quality_aa_v1_{stage}_synthetic_{route}_seed0_v1"
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("QUALITY-AA run identifier contains an invalid component")
    return run_id


def load_protocol(root: Path) -> dict[str, Any]:
    protocol = yaml.safe_load((root / PROTOCOL_PATH).read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise QualityAAConfigError("QUALITY-AA protocol must be a mapping")
    return protocol


def validate_protocol_config(protocol: dict[str, Any]) -> None:
    """Validate decision-critical literals before any run is authorized."""
    literals = {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "campaign_key": CAMPAIGN_KEY,
        "parent_campaign": PARENT_CAMPAIGN_VERSION,
        "stages": list(STAGES),
        "fixtures": [
            "tanh",
            "asymmetric_clipping",
            "two_clippers",
            "short_memory",
            "slow_sag",
            "rf2047_residual",
        ],
    }
    for name, expected in literals.items():
        _require_equal(protocol.get(name), expected, f"protocol.{name}")
    scope = protocol.get("scope", {})
    for name in (
        "amplifier_architecture_claim_allowed",
        "physical_hardware_aliasing_claim_allowed",
        "global_state_of_the_art_claim_allowed",
        "physical_audio_allowed",
        "physical_192khz_dataset_required",
        "fm9_proxy_allowed",
    ):
        _require_equal(scope.get(name), False, f"scope.{name}")
    _require_equal(scope.get("external_report_only_locked"), True, "external lock")
    probe = protocol.get("probe", {})
    for name, expected in {
        "sample_rate_hz": 48_000,
        "dft_samples": 65_536,
        "k0": [1705, 8191, 12287],
        "amplitudes": [0.10, 0.25, 0.48],
        "frames": 6,
        "window": "none",
        "zero_padding": False,
    }.items():
        _require_equal(probe.get(name), expected, f"probe.{name}")
    routes = protocol.get("routes", {})
    for name, factor, latency, taps in (
        ("full_island_x2", 2, 32, 65),
        ("teacher_x4", 4, 32, 129),
    ):
        route = routes.get(name, {})
        _require_equal(route.get("factor"), factor, f"routes.{name}.factor")
        _require_equal(route.get("latency_samples"), latency, f"routes.{name}.latency")
        _require_equal(route.get("filter_taps"), taps, f"routes.{name}.taps")
    _require_equal(
        protocol.get("mechanism_gate", {}).get("route_decisions_independent"),
        True,
        "route independence",
    )
    _require_equal(
        protocol.get("native_gate", {}).get("maximum_latency_samples"),
        48,
        "native latency cap",
    )


def validate_repository_configs(root: Path) -> dict[str, Any]:
    """Prove the active lineage and its immutable historical parent."""
    required = (
        root / PROTOCOL_PATH,
        root / CAMPAIGN_PATH / "PROTOCOL_LOCK.yaml",
        root / CAMPAIGN_PATH / "EXTERNAL_FREEZE.json",
        root / ".codex_campaign/LINEAGES.json",
        root / ".codex_campaign/r2_48k_v2/VERDICT.json",
    )
    for path in required:
        if not path.is_file():
            raise QualityAAConfigError(f"missing canonical file: {path}")
    protocol = load_protocol(root)
    validate_protocol_config(protocol)
    lock = yaml.safe_load(required[1].read_text(encoding="utf-8"))
    for name, expected in {
        "campaign_version": CAMPAIGN_VERSION,
        "campaign_key": CAMPAIGN_KEY,
        "parent_campaign": PARENT_CAMPAIGN_VERSION,
        "protocol_path": PROTOCOL_PATH,
        "status": "frozen_before_first_scientific_run",
        "scientific_runs_launched": 0,
        "parent_artifacts_immutable": True,
        "physical_audio_allowed": False,
        "fm9_proxy_allowed": False,
        "external_report_only_locked": True,
        "source_snapshot_required_for_every_run": True,
    }.items():
        _require_equal(lock.get(name), expected, f"lock.{name}")
    import json

    freeze = json.loads(required[2].read_text(encoding="utf-8"))
    lineages = json.loads(required[3].read_text(encoding="utf-8"))
    parent_verdict = json.loads(required[4].read_text(encoding="utf-8"))
    _require_equal(freeze.get("external_report_only_locked"), True, "freeze")
    _require_equal(freeze.get("physical_audio_samples_read"), 0, "physical audio")
    _require_equal(parent_verdict.get("verdict"), "INVALID", "parent verdict")
    _require_equal(
        (root / ".codex_campaign/ACTIVE_CAMPAIGN").read_text().strip(),
        CAMPAIGN_KEY,
        "active campaign",
    )
    _require_equal(lineages.get("active"), CAMPAIGN_KEY, "lineage active")
    entry = lineages.get("lineages", {}).get(CAMPAIGN_KEY, {})
    _require_equal(entry.get("campaign_version"), CAMPAIGN_VERSION, "lineage")
    _require_equal(entry.get("parent"), "r2_48k_v2", "lineage parent")
    return protocol


_STAGE_REQUIREMENTS = {
    "mechanism": ("preflight", "passed"),
    "native": ("mechanism", "passed"),
    "audit": ("native", "passed"),
}


def validate_stage_authorization(stage: str, decisions: dict[str, str]) -> None:
    """Fail closed unless the exact preceding gate passed."""
    if stage == "preflight":
        return
    if stage not in _STAGE_REQUIREMENTS:
        raise ValueError(f"unsupported QUALITY-AA stage: {stage}")
    name, expected = _STAGE_REQUIREMENTS[stage]
    if decisions.get(name) != expected:
        raise QualityAAAuthorizationError(
            f"{stage} requires {name}={expected}; got {decisions.get(name)!r}"
        )
