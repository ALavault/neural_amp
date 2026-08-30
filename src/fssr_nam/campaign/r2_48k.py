"""Fail-closed contracts for the versioned 48 kHz R2 campaign."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CAMPAIGN_VERSION = "FSSR-R2-48K-v1"
CAMPAIGN_KEY = "r2_48k"
CANDIDATE_FAMILIES = ("aa-nam", "aa-fssr", "aa-fssr-xl")
DEVELOPMENT_DEVICES = ("fulltone", "bigmuff")
PRIMARY_DEVICES = ("blackstar", "ua1176")
FINAL_DEVICES = (*DEVELOPMENT_DEVICES, *PRIMARY_DEVICES)
SCREEN_INITIAL_TRAJECTORIES = 16
SCREEN_ADAA_TRAJECTORIES_MAXIMUM = 6
DEVELOPMENT_ROBUSTNESS_TRAJECTORIES = 16
INTERNAL_VALIDATION_TRAJECTORIES = 20
EVALUATION_CONDITIONS = 40
PROSPECTIVE_CONDITIONS = 20
R2_48K_STAGES = (
    "mechanism",
    "screen",
    "teacher",
    "distill",
    "robustness",
    "confirm",
    "benchmark",
    "listen",
)
_RUN_ID = re.compile(
    r"^r2_48k_(?P<stage>[a-z0-9-]+)_(?P<device>[a-z0-9-]+)_"
    r"(?P<family>[a-z0-9-]+)_(?P<aa>[a-z0-9_]+)_"
    r"seed(?P<seed>0|[1-9][0-9]*)_v1$"
)


class R248KCampaignError(RuntimeError):
    """Base class for R2-48K campaign contract failures."""


class R248KConfigError(R248KCampaignError):
    """Raised when a frozen R2-48K repository contract changed."""


class R248KAuthorizationError(R248KCampaignError):
    """Raised when a sequential R2-48K stage is not unlocked."""


@dataclass(frozen=True)
class RunSpec:
    stage: str
    device: str
    family: str
    aa_mode: str
    seed: int

    @property
    def run_id(self) -> str:
        return make_run_id(
            self.stage, self.device, self.family, self.aa_mode, self.seed
        )


def make_run_id(stage: str, device: str, family: str, aa_mode: str, seed: int) -> str:
    """Create an immutable identifier that cannot collide with R2-v1."""
    if stage not in R2_48K_STAGES:
        raise ValueError(f"unknown R2-48K stage: {stage}")
    if seed < 0:
        raise ValueError("R2-48K seed must be non-negative")
    run_id = f"r2_48k_{stage}_{device}_{family}_{aa_mode}_seed{seed}_v1"
    if _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("R2-48K run identifier contains an invalid component")
    return run_id


def parse_run_id(run_id: str) -> RunSpec:
    """Parse one canonical R2-48K ID, rejecting retries and aliases."""
    match = _RUN_ID.fullmatch(run_id)
    if match is None or match.group("stage") not in R2_48K_STAGES:
        raise ValueError(f"invalid R2-48K run_id: {run_id}")
    return RunSpec(
        stage=match.group("stage"),
        device=match.group("device"),
        family=match.group("family"),
        aa_mode=match.group("aa"),
        seed=int(match.group("seed")),
    )


def initial_screen_specs() -> tuple[RunSpec, ...]:
    """Return twelve ambitious x2 and four pinned A2 trajectories."""
    specs = [
        RunSpec("screen", device, f"{family}-{loss}", "full_island_x2", 0)
        for device in DEVELOPMENT_DEVICES
        for family in CANDIDATE_FAMILIES
        for loss in ("m4", "wright")
    ]
    specs.extend(
        RunSpec("screen", device, f"a2-{loss}", "off", 0)
        for device in DEVELOPMENT_DEVICES
        for loss in ("m4", "wright")
    )
    if len(specs) != SCREEN_INITIAL_TRAJECTORIES:
        raise AssertionError("internal R2-48K screen matrix count changed")
    return tuple(specs)


def adaa_screen_specs(promoted_loss: str) -> tuple[RunSpec, ...]:
    """Return ADAA challengers only after the synthetic route qualifies."""
    if promoted_loss not in {"m4", "wright"}:
        raise ValueError("promoted R2-48K loss must be m4 or wright")
    specs = tuple(
        RunSpec("screen", device, f"{family}-{promoted_loss}", "adaa1", 0)
        for device in DEVELOPMENT_DEVICES
        for family in CANDIDATE_FAMILIES
    )
    if len(specs) != SCREEN_ADAA_TRAJECTORIES_MAXIMUM:
        raise AssertionError("internal R2-48K ADAA matrix count changed")
    return specs


def teacher_specs(promoted_family: str) -> tuple[RunSpec, ...]:
    _validate_candidate(promoted_family, "full_island_x2")
    return tuple(
        RunSpec("teacher", device, promoted_family, "teacher_x4", 0)
        for device in DEVELOPMENT_DEVICES
    )


def development_robustness_specs(
    promoted_family: str, aa_mode: str
) -> tuple[RunSpec, ...]:
    """Return seeds 1--4 for A2 and the candidate on historical development."""
    _validate_candidate(promoted_family, aa_mode)
    specs = tuple(
        RunSpec("robustness", device, family, mode, seed)
        for device in DEVELOPMENT_DEVICES
        for seed in (1, 2, 3, 4)
        for family, mode in (("a2", "off"), (promoted_family, aa_mode))
    )
    if len(specs) != DEVELOPMENT_ROBUSTNESS_TRAJECTORIES:
        raise AssertionError("internal R2-48K robustness matrix count changed")
    return specs


def internal_validation_specs(
    promoted_family: str, aa_mode: str
) -> tuple[RunSpec, ...]:
    """Return the locked, no-tuning Blackstar/UA five-seed matrix."""
    _validate_candidate(promoted_family, aa_mode)
    specs = tuple(
        RunSpec("confirm", device, family, mode, seed)
        for device in PRIMARY_DEVICES
        for seed in range(5)
        for family, mode in (("a2", "off"), (promoted_family, aa_mode))
    )
    if len(specs) != INTERNAL_VALIDATION_TRAJECTORIES:
        raise AssertionError("internal R2-48K validation matrix count changed")
    return specs


def evaluation_condition_keys() -> tuple[tuple[str, str, int], ...]:
    """Return all 40 report conditions and preserve their evidence roles."""
    conditions = tuple(
        (device, family, seed)
        for device in FINAL_DEVICES
        for family in ("a2", "candidate")
        for seed in range(5)
    )
    if len(conditions) != EVALUATION_CONDITIONS:
        raise AssertionError("internal R2-48K evaluation count changed")
    return conditions


def prospective_condition_keys() -> tuple[tuple[str, str, int], ...]:
    conditions = tuple(
        (device, family, seed)
        for device in PRIMARY_DEVICES
        for family in ("a2", "candidate")
        for seed in range(5)
    )
    if len(conditions) != PROSPECTIVE_CONDITIONS:
        raise AssertionError("internal R2-48K prospective count changed")
    return conditions


def _validate_candidate(family: str, aa_mode: str) -> None:
    if family not in CANDIDATE_FAMILIES:
        raise ValueError("R2-48K candidate family is unsupported")
    if aa_mode not in {
        "full_island_x2",
        "adaa1",
        "distilled_x2",
        "distilled_adaa1",
    }:
        raise ValueError("R2-48K candidate AA mode is unsupported")


def _require_equal(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise R248KConfigError(f"{label} must equal {expected!r}, got {value!r}")


def validate_protocol_config(config: dict[str, Any]) -> None:
    """Validate every decision-critical R2-48K declaration."""
    _require_equal(config.get("campaign_version"), CAMPAIGN_VERSION, "campaign")
    _require_equal(config.get("parent_campaign"), "FSSR-R2-v1", "parent")
    _require_equal(config.get("base_sample_rate_hz"), 48_000, "base sample rate")
    _require_equal(
        config.get("valid_terminal_verdicts"),
        ["GO-R2-48K", "NO-GO-R2-48K", "INVALID"],
        "terminal verdicts",
    )
    _require_equal(config.get("external_report_only_locked"), True, "external lock")

    scope = config.get("claim_scope", {})
    for name, expected in {
        "global_state_of_the_art_claim_allowed": False,
        "physical_hardware_aliasing_claim_allowed": False,
        "synthetic_aliasing_mechanism_claim_allowed": True,
        "synthetic_reference_sample_rate_hz": 192_000,
        "physical_reference_sample_rate_hz": 48_000,
        "derived_192khz_hardware_reference_allowed": False,
        "proxy_hardware_allowed": False,
        "prohibited_proxy_devices": ["fractal_fm9"],
    }.items():
        _require_equal(scope.get(name), expected, f"claim_scope.{name}")

    decision = config.get("decision", {})
    for name, expected in {
        "primary_devices": ["blackstar", "ua1176"],
        "heldout_esr_relative_improvement_lower_bound_minimum": 0.15,
        "devices_won_minimum": 3,
        "mandatory_devices": ["blackstar", "ua1176"],
        "cpp_block_size": 64,
        "cpp_cpu_ratio_maximum": 1.25,
        "added_latency_samples_maximum": 48,
        "primary_mushra_advantage_points_strictly_greater_than": 10.0,
        "primary_mushra_lower_bound_strictly_greater_than": 0.0,
        "synthetic_mechanism_x2_must_pass": True,
        "physical_asr_is_a_decision_metric": False,
    }.items():
        _require_equal(decision.get(name), expected, f"decision.{name}")

    architectures = config.get("architectures", {})
    _require_equal(
        architectures.get("selection_eligible_families"),
        list(CANDIDATE_FAMILIES),
        "candidate families",
    )
    data = config.get("data", {})
    for name, expected in {
        "source_manifest": "datasets/manifests/r1_physical.json",
        "new_capture_required": False,
        "development_devices": list(DEVELOPMENT_DEVICES),
        "development_outputs_previously_observed": True,
        "internal_validation_devices": list(PRIMARY_DEVICES),
        "primary_test_devices": list(PRIMARY_DEVICES),
        "internal_validation_test_open_count_maximum": 1,
    }.items():
        _require_equal(data.get(name), expected, f"data.{name}")

    mechanism = config.get("mechanism", {})
    _require_equal(len(mechanism.get("fixtures", [])), 6, "mechanism fixture count")
    for name, expected in {
        "reference_kind": "synthetic_192khz",
        "reference_sample_rate_hz": 192_000,
        "spearman_rho_minimum": 0.90,
        "median_asr_gain_db_minimum": 10.0,
        "per_fixture_asr_gain_db_minimum": 6.0,
        "fundamental_complex_error_maximum": 1.0e-5,
        "supports_hardware_aliasing_claim": False,
    }.items():
        _require_equal(mechanism.get(name), expected, f"mechanism.{name}")

    screening = config.get("screening", {})
    for name, expected in {
        "families": list(CANDIDATE_FAMILIES),
        "initial_x2_trajectories": 12,
        "baseline_trajectories": 4,
        "adaa_challenger_trajectories_maximum": 6,
        "non_promoted_loss_stop_updates": 5000,
        "promoted_loss_final_updates": 15000,
        "checkpoints": [200, 1000, 5000, 15000],
        "selection_metrics": ["esr"],
        "physical_asr_selection_forbidden": True,
    }.items():
        _require_equal(screening.get(name), expected, f"screening.{name}")

    teacher = config.get("teacher_and_distillation", {})
    for name, expected in {
        "teacher_internal_sample_rate_hz": 192_000,
        "physical_target_sample_rate_hz": 48_000,
        "teacher_role": "model_side_regularizer_not_hardware_reference",
        "teacher_esr_improvement_minimum_each_device": 0.15,
    }.items():
        _require_equal(teacher.get(name), expected, f"teacher.{name}")

    confirmation = config.get("confirmation", {})
    for name, expected in {
        "evaluation_conditions": EVALUATION_CONDITIONS,
        "prospective_conditions": PROSPECTIVE_CONDITIONS,
        "historical_development_conditions": 20,
        "no_tuning_on_internal_validation": True,
    }.items():
        _require_equal(confirmation.get(name), expected, f"confirmation.{name}")
    bootstrap = confirmation.get("bootstrap", {})
    for name, expected in {
        "replicates": 10_000,
        "seed": 20_260_828,
        "hierarchy": ["seed", "source"],
        "primary_devices": list(PRIMARY_DEVICES),
        "development_devices_excluded_from_primary_interval": True,
    }.items():
        _require_equal(bootstrap.get(name), expected, f"bootstrap.{name}")

    physical_asr = config.get("physical_asr", {})
    _require_equal(physical_asr.get("status"), "diagnostic_only", "physical ASR")
    _require_equal(
        physical_asr.get("final_decision_allowed"), False, "physical ASR decision"
    )
    benchmark = config.get("benchmark", {})
    for name, expected in {
        "block_sizes": [1, 16, 64, 128],
        "primary_block_size": 64,
        "repetitions": 30,
        "python_cpp_max_abs_error": 2.0e-5,
    }.items():
        _require_equal(benchmark.get(name), expected, f"benchmark.{name}")
    listening = config.get("listening", {})
    for name, expected in {
        "recruited_participants": 24,
        "retained_participants_minimum": 20,
        "excerpts_total": 8,
        "primary_excerpts": 4,
        "primary_devices": list(PRIMARY_DEVICES),
        "preregistered_exclusion_rule": "configs/r2_48k/mushra_exclusion.yaml",
    }.items():
        _require_equal(listening.get(name), expected, f"listening.{name}")
    export = config.get("export", {})
    _require_equal(
        export.get("implementation_format"), "fssr-r2-native-v1", "export format"
    )
    _require_equal(export.get("families"), list(CANDIDATE_FAMILIES), "export families")


def validate_repository_configs(root: Path) -> dict[str, Any]:
    """Validate this lineage and any declared active descendant preserving it."""
    protocol_path = root / "configs/r2_48k/protocol.yaml"
    lock_path = root / ".codex_campaign/r2_48k/PROTOCOL_LOCK.yaml"
    lineage_path = root / ".codex_campaign/LINEAGES.json"
    old_lock_path = root / ".codex_campaign/r2/PROTOCOL_LOCK.yaml"
    for path in (protocol_path, lock_path, lineage_path, old_lock_path):
        if not path.is_file():
            raise R248KConfigError(f"missing canonical R2-48K file: {path}")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    lineage = yaml.safe_load(lineage_path.read_text(encoding="utf-8"))
    old_lock = yaml.safe_load(old_lock_path.read_text(encoding="utf-8"))
    validate_protocol_config(protocol)
    _require_equal(lock.get("campaign_version"), CAMPAIGN_VERSION, "lock campaign")
    _require_equal(
        lock.get("status"), "frozen_before_first_scientific_run", "lock status"
    )
    _require_equal(lock.get("scientific_runs_launched"), 0, "lock run count")
    _require_equal(lock.get("physical_192khz_dataset_available"), False, "192k data")
    _require_equal(lock.get("fm9_proxy_allowed"), False, "FM9 proxy")
    active = (
        (root / ".codex_campaign/ACTIVE_CAMPAIGN").read_text(encoding="utf-8").strip()
    )
    _require_equal(lineage.get("active"), active, "lineage active campaign")
    lineages = lineage.get("lineages", {})
    current = active
    visited: set[str] = set()
    while current != CAMPAIGN_KEY:
        if current in visited:
            raise R248KConfigError("active lineage ancestry contains a cycle")
        visited.add(current)
        descendant = lineages.get(current, {})
        parent = descendant.get("parent")
        if not isinstance(parent, str) or not parent:
            raise R248KConfigError(
                f"active lineage {active!r} does not descend from {CAMPAIGN_KEY!r}"
            )
        current = parent
    entry = lineages.get(CAMPAIGN_KEY, {})
    _require_equal(entry.get("campaign_version"), CAMPAIGN_VERSION, "lineage version")
    _require_equal(entry.get("parent"), "r2", "lineage parent")
    _require_equal(old_lock.get("campaign_version"), "FSSR-R2-v1", "old R2 lock")
    _require_equal(old_lock.get("status"), "frozen_before_capture", "old R2 status")
    _require_equal(old_lock.get("scientific_runs_launched"), 0, "old R2 runs")
    old_entry = lineages.get("r2", {})
    _require_equal(
        old_entry.get("historical_artifacts_immutable"), True, "old R2 immutability"
    )
    return protocol


_STAGE_REQUIREMENTS = {
    "data": (("preflight",), ("passed",)),
    "mechanism": (("data",), ("passed",)),
    "screen": (("mechanism_x2",), ("passed",)),
    "teacher": (("screen",), ("promoted",)),
    "distill": (
        ("teacher", "deployable"),
        ("passed", "failed_fidelity_gate"),
    ),
    "lock": (("final_candidate",), ("selected",)),
    "confirm": (("lock",), ("frozen",)),
    "benchmark": (("confirm_validation",), ("passed",)),
    "sealed_test": (
        ("confirm_validation", "python_cpp_parity", "benchmark", "mechanism_x2"),
        ("passed", "passed", "passed", "passed"),
    ),
    "listen": (
        ("sealed_esr", "cpu", "mechanism_x2"),
        ("passed", "passed", "passed"),
    ),
    "audit": (("sealed_test", "listening"), ("opened_once", "complete")),
}


def validate_stage_authorization(stage: str, decisions: dict[str, str]) -> None:
    """Require literal prior evidence; missing evidence always fails closed."""
    if stage == "preflight":
        return
    if stage not in _STAGE_REQUIREMENTS:
        raise ValueError(f"unknown R2-48K authorization stage: {stage}")
    names, expected = _STAGE_REQUIREMENTS[stage]
    for name, status in zip(names, expected, strict=True):
        if decisions.get(name) != status:
            raise R248KAuthorizationError(
                f"R2-48K {stage} requires {name}={status}; got {decisions.get(name)!r}"
            )


def validate_sealed_test_boundary(
    *,
    decisions: dict[str, str],
    internal_validation_tests_open_count: int,
    external_report_only_locked: bool,
    development_tests_previously_observed: bool,
) -> None:
    """Authorize one opening of Blackstar/UA tests, never dev or external data."""
    validate_stage_authorization("sealed_test", decisions)
    if internal_validation_tests_open_count != 0:
        raise R248KAuthorizationError(
            "R2-48K internal-validation tests may be opened only once"
        )
    if not external_report_only_locked:
        raise R248KAuthorizationError("EXTERNAL_REPORT_ONLY must remain locked")
    if development_tests_previously_observed is not True:
        raise R248KAuthorizationError(
            "R2-48K must disclose that development tests were previously observed"
        )


def unlock_internal_validation(freeze: Mapping[str, Any]) -> dict[str, Any]:
    """Release Blackstar/UA train-validation while preserving their test seal."""
    updated = dict(freeze)
    if updated.get("campaign_version") != CAMPAIGN_VERSION:
        raise R248KAuthorizationError("internal-validation freeze campaign changed")
    if updated.get("external_report_only_locked") is not True:
        raise R248KAuthorizationError("EXTERNAL_REPORT_ONLY must remain locked")
    if updated.get("internal_validation_tests_locked") is not True:
        raise R248KAuthorizationError("internal-validation tests are not locked")
    if updated.get("internal_validation_test_open_count") != 0:
        raise R248KAuthorizationError("validation cannot unlock after test opening")
    current = updated.get("internal_validation_train_validation_locked")
    if current not in {True, False}:
        raise R248KAuthorizationError("train-validation freeze state is invalid")
    updated["internal_validation_train_validation_locked"] = False
    return updated


def open_internal_validation_tests(freeze: Mapping[str, Any]) -> dict[str, Any]:
    """Record the single prospective test opening after all earlier gates."""
    updated = dict(freeze)
    if updated.get("campaign_version") != CAMPAIGN_VERSION:
        raise R248KAuthorizationError("internal-validation freeze campaign changed")
    if updated.get("external_report_only_locked") is not True:
        raise R248KAuthorizationError("EXTERNAL_REPORT_ONLY must remain locked")
    if updated.get("internal_validation_train_validation_locked") is not False:
        raise R248KAuthorizationError("train-validation must unlock before testing")
    if updated.get("internal_validation_test_open_count") != 0:
        raise R248KAuthorizationError("internal-validation tests were already opened")
    if updated.get("internal_validation_tests_locked") is not True:
        raise R248KAuthorizationError("internal-validation test lock is inconsistent")
    updated["internal_validation_tests_opened"] = True
    updated["internal_validation_test_open_count"] = 1
    updated["internal_validation_tests_locked"] = False
    updated["internal_validation_outputs_locked"] = False
    return updated
