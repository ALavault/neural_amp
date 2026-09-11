"""Prospective contract for the competence-first architecture campaign."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CAMPAIGN_VERSION = "AMP-COMPETENCE-ARCH-v2"
CAMPAIGN_KEY = "amp_competence_arch_v2"
PROTOCOL_PATH = Path("configs/amp_competence_arch_v2/protocol.yaml")
CAMPAIGN_PATH = Path(".codex_campaign/amp_competence_arch_v2")
SYSTEMS = ("static_composite", "dynamic_composite", "two_clippers")
CONTROL_FAMILY = "micro_tcn_x2"
CANDIDATE_FAMILIES = (
    "phys_s6_tcn_x2",
    "phys_det_tcn_x2",
    "rf2047_tfilm_x2",
    "cascade_rf2047_tfilm_x2",
)
ALL_FAMILIES = (CONTROL_FAMILY, *CANDIDATE_FAMILIES)
SEEDS = (0, 1, 2)
CHECKPOINTS = (500, 1000, 2000, 5000, 10000, 15000)
RUN_PATTERN = re.compile(
    r"^arch_v2_(?P<stage>competence|comparison)_"
    r"(?P<system>static_composite|dynamic_composite|two_clippers)_"
    r"(?P<family>[a-z0-9_]+)_seed(?P<seed>[0-9]+)_v1$"
)


class ArchV2ConfigError(RuntimeError):
    """Raised when the prospective v2 contract is incomplete or changed."""


class ArchV2AuthorizationError(RuntimeError):
    """Raised when a v2 stage is attempted before its prerequisite gate."""


@dataclass(frozen=True)
class ArchV2RunSpec:
    run_id: str
    stage: str
    system: str
    family: str
    seed: int


def _require_equal(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ArchV2ConfigError(
            f"{label} changed: expected {expected!r}, got {value!r}"
        )


def load_protocol(root: Path) -> dict[str, Any]:
    path = root / PROTOCOL_PATH
    if not path.is_file():
        raise ArchV2ConfigError(f"missing architecture v2 protocol: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArchV2ConfigError("architecture v2 protocol must be a mapping")
    validate_protocol_config(value)
    return value


def validate_protocol_config(protocol: Mapping[str, Any]) -> None:
    """Reject any drift in the decision-bearing v2 contract."""
    _require_equal(protocol.get("schema_version"), 1, "schema_version")
    _require_equal(protocol.get("campaign_version"), CAMPAIGN_VERSION, "campaign")
    _require_equal(protocol.get("campaign_key"), CAMPAIGN_KEY, "campaign_key")
    _require_equal(protocol.get("parent_campaign"), "AMP-QUALITY-ARCH-v1", "parent")
    _require_equal(protocol.get("sample_rate_hz"), 48_000, "sample_rate")
    _require_equal(protocol.get("precision"), "float32", "precision")

    boundaries = protocol.get("boundaries", {})
    for key in (
        "v1_runs_resumed",
        "v1_runs_retuned",
        "v1_results_used_for_v2_selection",
        "physical_audio_allowed",
        "physical_192khz_dataset_assumed",
        "fm9_capture_allowed",
    ):
        _require_equal(boundaries.get(key), False, f"boundaries.{key}")
    for key in (
        "blackstar_ua1176_holdouts_locked",
        "external_report_only_locked",
    ):
        _require_equal(boundaries.get(key), True, f"boundaries.{key}")
    _require_equal(boundaries.get("physical_audio_samples_maximum"), 0, "physical cap")

    stages = protocol.get("stages", {})
    _require_equal(
        stages.get("ordered"),
        ["preflight", "competence", "comparison", "audit"],
        "stages.ordered",
    )
    _require_equal(
        stages.get("comparison_requires_competence_pass"),
        True,
        "stages.competence_first",
    )
    _require_equal(stages.get("result_dependent_retry_allowed"), False, "stages.retry")
    _require_equal(
        stages.get("failed_or_invalid_run_resume_allowed"), False, "stages.resume"
    )

    data = protocol.get("synthetic_data", {})
    _require_equal(data.get("systems"), list(SYSTEMS), "data.systems")
    _require_equal(data.get("episode_samples"), 8192, "data.samples")
    _require_equal(data.get("train_episodes"), 16, "data.train_episodes")
    _require_equal(data.get("validation_episodes"), 4, "data.validation_episodes")
    _require_equal(data.get("normalization"), "none", "data.normalization")
    source_seeds = [
        data.get("competence_train_source_seed"),
        data.get("competence_validation_source_seed"),
        data.get("comparison_train_source_seed"),
        data.get("comparison_validation_source_seed"),
    ]
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) for seed in source_seeds
    ):
        raise ArchV2ConfigError("data source seeds must be integers")
    if len(set(source_seeds)) != len(source_seeds):
        raise ArchV2ConfigError("v2 source seeds must be disjoint")
    _require_equal(data.get("v1_sources_reused"), False, "data.v1_reuse")

    optimization = protocol.get("optimization", {})
    _require_equal(optimization.get("profile"), "max", "optimization.profile")
    _require_equal(optimization.get("loss"), "esr", "optimization.loss")
    _require_equal(optimization.get("optimizer"), "AdamW", "optimization.optimizer")
    _require_equal(optimization.get("learning_rate"), 0.001, "optimization.lr")
    _require_equal(optimization.get("weight_decay"), 0.0, "optimization.weight_decay")
    _require_equal(optimization.get("batch_size"), 1, "optimization.batch")
    _require_equal(optimization.get("loss_tail_samples"), 2048, "optimization.tail")

    competence = protocol.get("competence", {})
    _require_equal(
        competence.get("control_family"), CONTROL_FAMILY, "competence.control"
    )
    _require_equal(competence.get("seeds"), list(SEEDS), "competence.seeds")
    _require_equal(
        competence.get("checkpoint_updates"),
        list(CHECKPOINTS),
        "competence.checkpoints",
    )
    _require_equal(competence.get("trajectory_count"), 9, "competence.count")
    _require_equal(
        competence.get("gain_error_strictly_greater_than"), -0.2, "competence.gain"
    )
    _require_equal(
        competence.get("correlation_strictly_greater_than"),
        0.9,
        "competence.correlation",
    )
    _require_equal(
        competence.get("plateau_median_relative_esr_improvement_minimum"),
        0.0,
        "competence.plateau_minimum",
    )
    _require_equal(
        competence.get("plateau_median_relative_esr_improvement_maximum"),
        0.05,
        "competence.plateau_maximum",
    )
    _require_equal(
        competence.get("failure_verdict"),
        "NO-GO-COMPETENCE-v2",
        "competence.failure",
    )

    comparison = protocol.get("comparison", {})
    _require_equal(
        comparison.get("control_family"), CONTROL_FAMILY, "comparison.control"
    )
    _require_equal(
        comparison.get("candidate_families"),
        list(CANDIDATE_FAMILIES),
        "comparison.candidates",
    )
    if len(CANDIDATE_FAMILIES) > int(comparison.get("maximum_candidate_families", -1)):
        raise ArchV2ConfigError("comparison candidate cap exceeded")
    _require_equal(comparison.get("seeds"), list(SEEDS), "comparison.seeds")
    _require_equal(
        comparison.get("paired_median_relative_esr_improvement_minimum"),
        0.10,
        "comparison.improvement",
    )
    _require_equal(
        comparison.get("per_system_median_relative_esr_regression_maximum"),
        0.05,
        "comparison.regression",
    )
    _require_equal(
        comparison.get("bootstrap_replicates"), 10_000, "comparison.bootstrap"
    )
    _require_equal(
        comparison.get("bootstrap_seed"), 20_260_828, "comparison.bootstrap_seed"
    )
    _require_equal(
        comparison.get("lower_95_confidence_bound_strictly_greater_than"),
        0.0,
        "comparison.lower_bound",
    )

    resource = protocol.get("resource_budget", {})
    _require_equal(
        resource.get("competence_run_directories_maximum"), 9, "resources.competence"
    )
    if int(resource.get("comparison_run_directories_maximum", -1)) < len(
        ALL_FAMILIES
    ) * len(SYSTEMS) * len(SEEDS):
        raise ArchV2ConfigError("comparison run-directory cap is too small")


def validate_repository_state(
    root: Path, *, require_frozen: bool = False
) -> dict[str, Any]:
    """Validate v1 terminality, v2 boundaries, and an optional exact lock."""
    protocol = load_protocol(root)
    v1_maturity = json.loads(
        (root / ".codex_campaign/amp_quality_arch_v1/MATURITY.json").read_text(
            encoding="utf-8"
        )
    )
    _require_equal(v1_maturity.get("status"), "terminal_no_go", "v1 status")
    _require_equal(v1_maturity.get("verdict"), "NO-GO-ARCH", "v1 verdict")
    v1_verdict = json.loads(
        (root / ".codex_campaign/amp_quality_arch_v1/VERDICT.json").read_text(
            encoding="utf-8"
        )
    )
    _require_equal(v1_verdict.get("verdict"), "NO-GO-ARCH", "v1 verdict artifact")
    v1_protocol = yaml.safe_load(
        (root / "configs/amp_quality_arch_v1/protocol.yaml").read_text(encoding="utf-8")
    )
    v1_lock = yaml.safe_load(
        (root / ".codex_campaign/amp_quality_arch_v1/PROTOCOL_LOCK.yaml").read_text(
            encoding="utf-8"
        )
    )
    _require_equal(v1_protocol, v1_lock, "v1 protocol lock")
    lineages = json.loads(
        (root / ".codex_campaign/LINEAGES.json").read_text(encoding="utf-8")
    )
    v1_lineage = lineages.get("lineages", {}).get("amp_quality_arch_v1", {})
    _require_equal(
        v1_lineage.get("historical_artifacts_immutable"), True, "v1 immutability"
    )
    freeze = json.loads(
        (root / CAMPAIGN_PATH / "EXTERNAL_FREEZE.json").read_text(encoding="utf-8")
    )
    _require_equal(freeze.get("physical_audio_allowed"), False, "v2 physical boundary")
    _require_equal(
        freeze.get("sealed_test_output_accessed"), False, "v2 sealed boundary"
    )
    _require_equal(
        freeze.get("external_report_only_locked"), True, "v2 external boundary"
    )
    lock_path = root / CAMPAIGN_PATH / "PROTOCOL_LOCK.yaml"
    if lock_path.exists():
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        _require_equal(lock, protocol, "v2 protocol lock")
    elif require_frozen:
        raise ArchV2ConfigError("frozen v2 protocol lock is required")
    return protocol


def make_run_id(stage: str, system: str, family: str, seed: int) -> str:
    if stage not in {"competence", "comparison"}:
        raise ValueError("v2 scientific stage is invalid")
    if system not in SYSTEMS:
        raise ValueError("v2 system is invalid")
    allowed = (CONTROL_FAMILY,) if stage == "competence" else ALL_FAMILIES
    if family not in allowed:
        raise ValueError("v2 family is invalid for stage")
    if seed not in SEEDS:
        raise ValueError("v2 seed is invalid")
    return f"arch_v2_{stage}_{system}_{family}_seed{seed}_v1"


def parse_run_id(run_id: str) -> ArchV2RunSpec:
    match = RUN_PATTERN.fullmatch(run_id)
    if match is None:
        raise ValueError("invalid architecture v2 run identifier")
    seed = int(match.group("seed"))
    canonical = make_run_id(
        match.group("stage"), match.group("system"), match.group("family"), seed
    )
    if canonical != run_id:
        raise ValueError("non-canonical architecture v2 run identifier")
    return ArchV2RunSpec(
        run_id=run_id,
        stage=match.group("stage"),
        system=match.group("system"),
        family=match.group("family"),
        seed=seed,
    )


def validate_stage_authorization(stage: str, decisions: Mapping[str, str]) -> None:
    if stage == "preflight":
        if decisions:
            raise ArchV2AuthorizationError("v2 preflight requires an empty gate ledger")
        return
    if decisions.get("preflight") != "passed":
        raise ArchV2AuthorizationError("v2 preflight has not passed")
    if stage == "competence":
        if "competence" in decisions:
            raise ArchV2AuthorizationError("v2 competence is already recorded")
        return
    if stage == "comparison":
        if decisions.get("competence") != "passed":
            raise ArchV2AuthorizationError("v2 comparison requires passed competence")
        if "comparison" in decisions:
            raise ArchV2AuthorizationError("v2 comparison is already recorded")
        return
    if stage == "audit":
        competence = decisions.get("competence")
        comparison = decisions.get("comparison")
        if competence not in {"passed", "failed"}:
            raise ArchV2AuthorizationError("v2 audit requires a competence decision")
        if competence == "passed" and comparison not in {"passed", "failed"}:
            raise ArchV2AuthorizationError("v2 audit requires a comparison decision")
        return
    raise ArchV2AuthorizationError("unknown v2 stage")
