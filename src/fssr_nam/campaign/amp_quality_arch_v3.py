"""Prospective contract for the representability-first v3 architecture campaign."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CAMPAIGN_VERSION = "AMP-QUALITY-ARCH-v3"
CAMPAIGN_KEY = "amp_quality_arch_v3"
PARENT_KEY = "amp_competence_arch_v2"
PROTOCOL_PATH = Path("configs/amp_quality_arch_v3/protocol.yaml")
ROUND_ONE_CONFIG_PATH = Path("configs/amp_quality_arch_v3/round_1.yaml")
CAMPAIGN_PATH = Path(".codex_campaign/amp_quality_arch_v3")
PRIMARY_SYSTEMS = (
    "static_primary",
    "dynamic_primary",
    "two_clippers_primary",
)
DIAGNOSTIC_SYSTEMS = (
    "tanh_component",
    "asymmetric_component",
    "memory_component",
    "sag_component",
    "envelope_component",
    "blocking_component",
    "level_component",
)
STRESS_SYSTEMS = ("v2_dynamic_rail_stress",)
INITIAL_FAMILIES = (
    "gainhead_micro_tcn_x2",
    "slow_state_micro_tcn_x2",
    "long_rf_tcn_x2",
)
COMPETENCE_SEEDS = (0, 1, 2, 3, 4)
ROUND_ONE_SEEDS = (0, 1, 2)
ROUND_ONE_CHECKPOINTS = (500, 1000, 2000, 5000, 10000, 15000)
COMPETENCE_CHECKPOINTS = (1000, 2000, 5000, 10000, 15000, 20000)
RUN_PATTERN = re.compile(
    r"^arch_v3_(?P<stage>round_[123]|competence|native|internal_dev)_"
    r"(?P<system>[a-z0-9_]+)__(?P<family>[a-z0-9_]+)_seed(?P<seed>[0-9]+)_v1$"
)


class ArchV3ConfigError(RuntimeError):
    """Raised when the prospective v3 contract is incomplete or changed."""


class ArchV3AuthorizationError(RuntimeError):
    """Raised when a v3 stage is attempted before its prerequisite gate."""


@dataclass(frozen=True)
class ArchV3RunSpec:
    run_id: str
    stage: str
    system: str
    family: str
    seed: int


def _require_equal(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ArchV3ConfigError(
            f"{label} changed: expected {expected!r}, got {value!r}"
        )


def load_protocol(root: Path) -> dict[str, Any]:
    path = root / PROTOCOL_PATH
    if not path.is_file():
        raise ArchV3ConfigError(f"missing architecture v3 protocol: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArchV3ConfigError("architecture v3 protocol must be a mapping")
    validate_protocol_config(value)
    return value


def validate_protocol_config(protocol: Mapping[str, Any]) -> None:
    """Reject drift in the decision-bearing prospective v3 contract."""
    _require_equal(protocol.get("schema_version"), 1, "schema_version")
    _require_equal(protocol.get("campaign_version"), CAMPAIGN_VERSION, "campaign")
    _require_equal(protocol.get("campaign_key"), CAMPAIGN_KEY, "campaign_key")
    _require_equal(protocol.get("parent_campaign"), "AMP-COMPETENCE-ARCH-v2", "parent")
    _require_equal(protocol.get("sample_rate_hz"), 48_000, "sample_rate")
    _require_equal(protocol.get("precision"), "float32", "precision")

    boundaries = protocol.get("boundaries", {})
    for key in (
        "parent_runs_resumed",
        "parent_runs_retuned",
        "physical_192khz_dataset_assumed",
        "fm9_capture_allowed",
        "new_physical_capture_allowed",
        "internal_dev_physical_allowed_before_native_gate",
        "physical_superiority_claim_allowed",
    ):
        _require_equal(boundaries.get(key), False, f"boundaries.{key}")
    for key in (
        "parent_results_used_for_hypotheses_only",
        "parent_scientific_artifacts_immutable",
        "blackstar_ua1176_holdouts_locked",
        "external_report_only_locked",
    ):
        _require_equal(boundaries.get(key), True, f"boundaries.{key}")

    stages = protocol.get("stages", {})
    _require_equal(
        stages.get("ordered"),
        [
            "preflight",
            "round_1",
            "round_2",
            "round_3",
            "competence",
            "lock",
            "native",
            "internal_dev",
            "audit",
        ],
        "stages.ordered",
    )
    _require_equal(stages.get("maximum_exploration_rounds"), 3, "stages.rounds")
    _require_equal(
        stages.get("stop_after_consecutive_rounds_below_improvement"),
        2,
        "stages.stop",
    )
    _require_equal(
        stages.get("round_median_esr_improvement_minimum"),
        0.05,
        "stages.improvement",
    )
    _require_equal(
        stages.get("result_dependent_retry_within_round_allowed"),
        False,
        "stages.retry",
    )
    _require_equal(
        stages.get("failed_or_invalid_run_resume_allowed"), False, "stages.resume"
    )

    data = protocol.get("synthetic_data", {})
    _require_equal(
        data.get("primary_systems"), list(PRIMARY_SYSTEMS), "data.primary_systems"
    )
    _require_equal(
        data.get("diagnostic_systems"),
        list(DIAGNOSTIC_SYSTEMS),
        "data.diagnostic_systems",
    )
    _require_equal(
        data.get("stress_systems"), list(STRESS_SYSTEMS), "data.stress_systems"
    )
    _require_equal(data.get("episode_samples"), 72_000, "data.episode_samples")
    _require_equal(data.get("preroll_samples"), 14_400, "data.preroll_samples")
    _require_equal(data.get("scored_samples"), 57_600, "data.scored_samples")
    _require_equal(data.get("minimum_episode_seconds"), 1.2, "data.duration")
    if data.get("episode_samples") != data.get("preroll_samples") + data.get(
        "scored_samples"
    ):
        raise ArchV3ConfigError("v3 episode must be exactly preroll plus scored audio")
    if int(data["scored_samples"]) < int(
        float(data["minimum_episode_seconds"]) * 48_000
    ):
        raise ArchV3ConfigError("v3 scored duration is shorter than 1.2 seconds")
    _require_equal(data.get("normalization"), "none", "data.normalization")
    source_seeds = [
        data.get("train_source_seed"),
        data.get("internal_dev_source_seed"),
        data.get("validation_source_seed"),
    ]
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) for seed in source_seeds
    ):
        raise ArchV3ConfigError("v3 source seeds must be integers")
    if len(set(source_seeds)) != len(source_seeds):
        raise ArchV3ConfigError("v3 source seeds must be disjoint")
    _require_equal(
        data.get("validation_locked_until_competence"),
        True,
        "data.validation_lock",
    )

    gate = protocol.get("representability_gate", {})
    _require_equal(gate.get("evidence_tier"), "train_only", "gate.tier")
    _require_equal(
        gate.get("required_residual_absolute_quantile"), 0.999, "gate.quantile"
    )
    _require_equal(gate.get("residual_initialization_margin"), 1.25, "gate.margin")
    _require_equal(
        gate.get("finite_residual_scale_ceiling_allowed"), False, "gate.ceiling"
    )
    _require_equal(gate.get("near_peak_fraction_maximum"), 0.50, "gate.near_peak")
    _require_equal(gate.get("minimum_training_eligible_families"), 3, "gate.families")

    architectures = protocol.get("architectures", {})
    _require_equal(
        architectures.get("initial_families"),
        list(INITIAL_FAMILIES),
        "architectures.families",
    )
    _require_equal(
        architectures.get("residual_scale_parameterization"),
        "softplus_unbounded",
        "architectures.residual_scale",
    )
    _require_equal(architectures.get("latency_samples"), 32, "architectures.latency")

    round_one = protocol.get("round_1", {})
    _require_equal(round_one.get("families"), list(INITIAL_FAMILIES), "round1.family")
    _require_equal(round_one.get("seeds"), list(ROUND_ONE_SEEDS), "round1.seeds")
    _require_equal(round_one.get("systems"), list(PRIMARY_SYSTEMS), "round1.systems")
    _require_equal(
        round_one.get("checkpoint_updates"),
        list(ROUND_ONE_CHECKPOINTS),
        "round1.checkpoints",
    )
    _require_equal(round_one.get("loss"), "esr_plus_projection_gain", "round1.loss")
    _require_equal(
        round_one.get("aggregate_training_metrics_over_all_episodes"),
        True,
        "round1.logging",
    )

    competence = protocol.get("competence", {})
    _require_equal(competence.get("seeds"), list(COMPETENCE_SEEDS), "competence.seeds")
    _require_equal(
        competence.get("checkpoint_updates"),
        list(COMPETENCE_CHECKPOINTS),
        "competence.checkpoints",
    )
    _require_equal(
        competence.get("gain_error_strictly_greater_than"), -0.2, "competence.gain"
    )
    _require_equal(
        competence.get("correlation_strictly_greater_than"),
        0.9,
        "competence.correlation",
    )
    for key in (
        "v2_compatibility_dynamic_median_esr_improvement_minimum",
        "v2_compatibility_two_clippers_median_esr_improvement_minimum",
    ):
        _require_equal(competence.get(key), 0.30, f"competence.{key}")
    _require_equal(
        competence.get("v2_compatibility_static_median_esr_regression_maximum"),
        0.05,
        "competence.static_regression",
    )

    runtime = protocol.get("runtime_gate", {})
    _require_equal(runtime.get("baseline"), "NAM_A2_Full", "runtime.baseline")
    _require_equal(
        runtime.get("python_cpp_max_absolute_error"), 2.0e-5, "runtime.parity"
    )
    _require_equal(runtime.get("latency_samples_maximum"), 48, "runtime.latency")
    _require_equal(runtime.get("block_64_cpu_ratio_maximum"), 1.25, "runtime.cpu")

    resources = protocol.get("resource_budget", {})
    _require_equal(
        resources.get("campaign_gpu_hours_maximum"), 324, "resources.gpu_hours"
    )
    _require_equal(resources.get("campaign_disk_gib_maximum"), 50, "resources.disk")


def load_round_one_lock(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Load and validate the prospective round-one lock against the protocol."""
    config_path = root / ROUND_ONE_CONFIG_PATH
    lock_path = root / CAMPAIGN_PATH / "ROUND_1_LOCK.yaml"
    if not config_path.is_file() or not lock_path.is_file():
        raise ArchV3ConfigError("v3 round-one config and lock are required")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(lock, dict):
        raise ArchV3ConfigError("v3 round-one config and lock must be mappings")
    _require_equal(lock, config, "v3 round-one lock")
    _require_equal(lock.get("schema_version"), 1, "round1.schema_version")
    _require_equal(lock.get("campaign_version"), CAMPAIGN_VERSION, "round1.campaign")
    _require_equal(lock.get("stage"), "round_1", "round1.stage")
    _require_equal(lock.get("evidence_tier"), "INTERNAL_DEV", "round1.tier")
    matrix = lock.get("matrix", {})
    _require_equal(matrix.get("families"), list(INITIAL_FAMILIES), "round1.families")
    _require_equal(matrix.get("systems"), list(PRIMARY_SYSTEMS), "round1.systems")
    _require_equal(matrix.get("seeds"), list(ROUND_ONE_SEEDS), "round1.seeds")
    _require_equal(
        matrix.get("trajectories"),
        len(INITIAL_FAMILIES) * len(PRIMARY_SYSTEMS) * len(ROUND_ONE_SEEDS),
        "round1.trajectories",
    )
    optimization = lock.get("optimization", {})
    protocol_round = protocol["round_1"]
    for key in (
        "profile",
        "checkpoint_updates",
        "loss",
        "projection_gain_weight",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "batch_size",
        "training_chunk_samples",
        "state_carry_between_contiguous_chunks",
        "aggregate_training_metrics_over_all_episodes",
    ):
        expected = (
            protocol["architectures"]["profile"]
            if key == "profile"
            else protocol_round[key]
        )
        _require_equal(optimization.get(key), expected, f"round1.optimization.{key}")
    selection = lock.get("selection", {})
    _require_equal(
        selection.get("guard_checkpoints"), [10_000, 15_000], "round1.guards"
    )
    _require_equal(
        selection.get("gain_error_strictly_greater_than"),
        protocol["competence"]["gain_error_strictly_greater_than"],
        "round1.gain",
    )
    _require_equal(
        selection.get("correlation_strictly_greater_than"),
        protocol["competence"]["correlation_strictly_greater_than"],
        "round1.correlation",
    )
    _require_equal(
        selection.get("score"),
        "median_paired_condition_normalized_esr_at_15000",
        "round1.score",
    )
    _require_equal(selection.get("tie_relative_tolerance"), 0.01, "round1.tie")
    boundaries = lock.get("boundaries", {})
    for key in (
        "result_dependent_retry_within_round_allowed",
        "failed_or_invalid_run_resume_allowed",
        "validation_source_accessed",
        "blackstar_accessed",
        "ua1176_accessed",
    ):
        _require_equal(boundaries.get(key), False, f"round1.boundaries.{key}")
    _require_equal(
        boundaries.get("physical_audio_samples_read"), 0, "round1.boundaries.physical"
    )
    _require_equal(
        boundaries.get("external_report_only_locked"),
        True,
        "round1.boundaries.external",
    )
    return lock


def validate_repository_state(
    root: Path, *, require_frozen: bool = False
) -> dict[str, Any]:
    """Validate lineage, parent immutability, boundaries, and optional lock."""
    protocol = load_protocol(root)
    lineages_path = root / ".codex_campaign/LINEAGES.json"
    lineages = json.loads(lineages_path.read_text(encoding="utf-8"))
    active = lineages.get("active")
    if not isinstance(active, str) or not active:
        raise ArchV3ConfigError("active lineage is missing")
    lineage_entries = lineages.get("lineages", {})
    descendant = active
    visited: set[str] = set()
    while descendant != CAMPAIGN_KEY:
        if descendant in visited:
            raise ArchV3ConfigError("active lineage ancestry contains a cycle")
        visited.add(descendant)
        parent_key = lineage_entries.get(descendant, {}).get("parent")
        if not isinstance(parent_key, str) or not parent_key:
            raise ArchV3ConfigError(
                f"active lineage {active!r} does not descend from {CAMPAIGN_KEY!r}"
            )
        descendant = parent_key
    parent = lineage_entries.get(PARENT_KEY, {})
    _require_equal(
        parent.get("historical_artifacts_immutable"), True, "parent immutability"
    )
    current = lineage_entries.get(CAMPAIGN_KEY, {})
    _require_equal(current.get("parent"), PARENT_KEY, "v3 lineage parent")
    _require_equal(current.get("campaign_version"), CAMPAIGN_VERSION, "v3 lineage")

    parent_maturity = json.loads(
        (root / ".codex_campaign/amp_competence_arch_v2/MATURITY.json").read_text(
            encoding="utf-8"
        )
    )
    _require_equal(
        parent_maturity.get("current_stage"), "terminal_audited", "parent stage"
    )
    _require_equal(
        parent_maturity.get("verdict"), "NO-GO-COMPETENCE-v2", "parent verdict"
    )
    parent_protocol = yaml.safe_load(
        (root / "configs/amp_competence_arch_v2/protocol.yaml").read_text(
            encoding="utf-8"
        )
    )
    parent_lock = yaml.safe_load(
        (root / ".codex_campaign/amp_competence_arch_v2/PROTOCOL_LOCK.yaml").read_text(
            encoding="utf-8"
        )
    )
    _require_equal(parent_protocol, parent_lock, "parent protocol lock")

    freeze = json.loads(
        (root / CAMPAIGN_PATH / "EXTERNAL_FREEZE.json").read_text(encoding="utf-8")
    )
    _require_equal(
        freeze.get("physical_192khz_dataset_assumed"), False, "192k boundary"
    )
    _require_equal(freeze.get("fm9_capture_allowed"), False, "FM9 boundary")
    _require_equal(
        freeze.get("new_physical_capture_allowed"), False, "capture boundary"
    )
    _require_equal(freeze.get("blackstar_accessed"), False, "Blackstar boundary")
    _require_equal(freeze.get("ua1176_accessed"), False, "UA1176 boundary")
    _require_equal(freeze.get("external_report_only_locked"), True, "external boundary")

    lock_path = root / CAMPAIGN_PATH / "PROTOCOL_LOCK.yaml"
    if lock_path.exists():
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        _require_equal(lock, protocol, "v3 protocol lock")
    elif require_frozen:
        raise ArchV3ConfigError("frozen v3 protocol lock is required")
    round_lock_path = root / CAMPAIGN_PATH / "ROUND_1_LOCK.yaml"
    if round_lock_path.exists():
        load_round_one_lock(root, protocol)
    return protocol


def make_run_id(stage: str, system: str, family: str, seed: int) -> str:
    if stage not in {
        "round_1",
        "round_2",
        "round_3",
        "competence",
        "native",
        "internal_dev",
    }:
        raise ValueError("v3 scientific stage is invalid")
    if not re.fullmatch(r"[a-z0-9_]+", system):
        raise ValueError("v3 system identifier is invalid")
    if not re.fullmatch(r"[a-z0-9_]+", family):
        raise ValueError("v3 family identifier is invalid")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("v3 seed is invalid")
    return f"arch_v3_{stage}_{system}__{family}_seed{seed}_v1"


def parse_run_id(run_id: str) -> ArchV3RunSpec:
    match = RUN_PATTERN.fullmatch(run_id)
    if match is None:
        raise ValueError("invalid architecture v3 run identifier")
    seed = int(match.group("seed"))
    canonical = make_run_id(
        match.group("stage"), match.group("system"), match.group("family"), seed
    )
    if canonical != run_id:
        raise ValueError("non-canonical architecture v3 run identifier")
    return ArchV3RunSpec(
        run_id=run_id,
        stage=match.group("stage"),
        system=match.group("system"),
        family=match.group("family"),
        seed=seed,
    )


def validate_stage_authorization(stage: str, decisions: Mapping[str, str]) -> None:
    """Fail closed at every v3 stage boundary."""
    if stage == "preflight":
        if decisions:
            raise ArchV3AuthorizationError("v3 preflight requires an empty gate ledger")
        return
    if decisions.get("preflight") != "passed":
        raise ArchV3AuthorizationError("v3 preflight has not passed")
    if stage == "round_1":
        if "round_1" in decisions:
            raise ArchV3AuthorizationError("v3 round 1 is already recorded")
        return
    if stage in {"round_2", "round_3"}:
        previous = f"round_{int(stage[-1]) - 1}"
        if decisions.get(previous) not in {"passed", "failed"}:
            raise ArchV3AuthorizationError(f"v3 {stage} requires {previous}")
        if stage in decisions:
            raise ArchV3AuthorizationError(f"v3 {stage} is already recorded")
        return
    if stage == "competence":
        if not any(decisions.get(f"round_{index}") == "passed" for index in (1, 2, 3)):
            raise ArchV3AuthorizationError("v3 competence requires a successful round")
        if "competence" in decisions:
            raise ArchV3AuthorizationError("v3 competence is already recorded")
        return
    if stage in {"lock", "native"}:
        prerequisite = "competence" if stage == "lock" else "lock"
        if decisions.get(prerequisite) != "passed":
            raise ArchV3AuthorizationError(f"v3 {stage} requires passed {prerequisite}")
        return
    if stage == "internal_dev":
        if decisions.get("native") != "passed":
            raise ArchV3AuthorizationError("v3 physical INTERNAL_DEV requires native")
        return
    if stage == "audit":
        if "preflight" not in decisions:
            raise ArchV3AuthorizationError("v3 audit requires preflight evidence")
        return
    raise ArchV3AuthorizationError("unknown v3 stage")
