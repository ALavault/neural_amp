"""Prospective identifiers and authorization for AMP-QUALITY-TEACHER-v1."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fssr_nam.data.quality_teacher import (
    CAMPAIGN_DEVICES,
    CONFIRMATION_DEVICES,
    DEVELOPMENT_DEVICES,
)
from fssr_nam.models.quality_teacher import QUALITY_TEACHER_FAMILIES
from fssr_nam.models.quality_teacher_comparators import QUALITY_TEACHER_COMPARATORS

CAMPAIGN_VERSION = "AMP-QUALITY-TEACHER-v1"
CAMPAIGN_KEY = "amp_quality_teacher_v1"
PARENT_CAMPAIGN = "AMP-SOTA-PROTOTYPE-v1.2"
PROTOCOL_POINTER_PATH = Path("configs/amp_quality_teacher_v1/protocol.yaml")
PROTOCOL_LOCK_PATH = Path(".codex_campaign/amp_quality_teacher_v1/PROTOCOL_LOCK.yaml")
STAGES = (
    "preflight",
    "data_audit",
    "ampeg_slow_value",
    "development",
    "candidate_lock",
    "comparator_lock",
    "recipes_lock",
    "confirmation_train",
    "confirmation_checkpoints_lock",
    "test_open",
    "objective_verdict",
    "audit",
)
DEVICES = (*CAMPAIGN_DEVICES, "all")
FAMILIES = (*QUALITY_TEACHER_FAMILIES, *QUALITY_TEACHER_COMPARATORS, "protocol")


def _alternatives(values: Sequence[str]) -> str:
    return "|".join(re.escape(value) for value in sorted(values, key=len, reverse=True))


RUN_ID_PATTERN = re.compile(
    rf"^quality_teacher_v1_(?P<stage>{_alternatives(STAGES)})_"
    rf"(?P<device>{_alternatives(DEVICES)})_"
    rf"(?P<family>{_alternatives(FAMILIES)})_"
    r"seed(?P<seed>0|[1-9][0-9]*)_v1$"
)
REQUIRED_LOCKS_FOR_TEST = (
    "candidate_lock",
    "comparator_lock",
    "recipes_lock",
    "confirmation_checkpoints_lock",
)


class QualityTeacherConfigError(RuntimeError):
    """Raised when a decision-bearing field drifts from the prospective lock."""


class QualityTeacherAuthorizationError(RuntimeError):
    """Raised when a stage is requested before its frozen prerequisites."""


@dataclass(frozen=True)
class QualityTeacherRunSpec:
    stage: str
    device: str
    family: str
    seed: int
    run_id: str


def _require_equal(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise QualityTeacherConfigError(
            f"{label} must equal {expected!r}, got {value!r}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualityTeacherConfigError(f"{label} must be a mapping")
    return value


def load_protocol(root: Path) -> dict[str, Any]:
    pointer_path = root / PROTOCOL_POINTER_PATH
    if not pointer_path.is_file():
        raise QualityTeacherConfigError(f"missing protocol pointer: {pointer_path}")
    pointer = yaml.safe_load(pointer_path.read_text(encoding="utf-8"))
    if not isinstance(pointer, dict):
        raise QualityTeacherConfigError("protocol pointer must be a mapping")
    _require_equal(pointer.get("schema_version"), 1, "pointer.schema_version")
    _require_equal(pointer.get("campaign_version"), CAMPAIGN_VERSION, "pointer.version")
    _require_equal(
        pointer.get("canonical_lock"), str(PROTOCOL_LOCK_PATH), "pointer.lock"
    )
    lock_path = root / PROTOCOL_LOCK_PATH
    if not lock_path.is_file():
        raise QualityTeacherConfigError(f"missing protocol lock: {lock_path}")
    protocol = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise QualityTeacherConfigError("protocol lock must be a mapping")
    validate_protocol_config(protocol)
    return protocol


def validate_protocol_config(protocol: Mapping[str, Any]) -> None:
    for key, expected in {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "campaign_key": CAMPAIGN_KEY,
        "parent_campaign": PARENT_CAMPAIGN,
        "status": "prospective_before_preflight_and_first_scientific_run",
        "supersedes_parent_before_parent_scientific_run": True,
        "sample_rate_hz": 48_000,
        "channels": 1,
        "precision": "float32",
        "run_id_format": "quality_teacher_v1_<stage>_<device>_<family>_seed<n>_v1",
        "valid_terminal_verdicts": [
            "GO-OBJECTIVE-SOTA",
            "NO-GO-OBJECTIVE-SOTA",
            "INVALID",
        ],
    }.items():
        _require_equal(protocol.get(key), expected, key)
    question = _mapping(protocol.get("scientific_question"), "scientific_question")
    _require_equal(
        question.get("sota_definition"),
        "best_open_reproducible_comparator_retrained_locally",
        "scientific_question.sota_definition",
    )
    _require_equal(
        question.get("claim"),
        "best_objective_fidelity_on_the_frozen_public_benchmark",
        "scientific_question.claim",
    )
    for field in (
        "perceptual_superiority_claim_allowed",
        "commercial_superiority_claim_allowed",
        "proprietary_comparator_claim_allowed",
    ):
        _require_equal(question.get(field), False, f"scientific_question.{field}")
    _require_equal(protocol.get("stage_order"), list(STAGES), "stage_order")

    data = _mapping(protocol.get("data"), "data")
    _require_equal(
        data.get("development_devices"), list(DEVELOPMENT_DEVICES), "data.development"
    )
    _require_equal(
        data.get("sealed_confirmation_devices"),
        list(CONFIRMATION_DEVICES),
        "data.confirmation",
    )
    _require_equal(
        data.get("catalog_commit"),
        "76ae7c875781bc7e2cec2df73b395d1a12d9cdbd",
        "data.catalog_commit",
    )
    for field in (
        "all_eligible_files_used",
        "historical_120_second_limit_removed",
        "source_group_boundaries_preserved",
        "upstream_test_directories_sealed_before_locks",
        "all_five_test_directories_opened_in_one_stage",
    ):
        _require_equal(data.get(field), True, f"data.{field}")
    _require_equal(
        data.get("development_tests_selection_eligible"), False, "data.dev_test"
    )

    architecture = _mapping(protocol.get("architecture"), "architecture")
    candidate = _mapping(architecture.get("candidate"), "architecture.candidate")
    _require_equal(
        candidate.get("family"), QUALITY_TEACHER_FAMILIES[0], "candidate.family"
    )
    _require_equal(candidate.get("latency_samples"), 32, "candidate.latency")
    _require_equal(candidate.get("final_tanh"), False, "candidate.final_tanh")
    _require_equal(candidate.get("bounded_dry_gain"), False, "candidate.dry_gain")
    fir = _mapping(candidate.get("fir"), "candidate.fir")
    _require_equal(fir.get("taps"), 257, "candidate.fir.taps")
    fast = _mapping(candidate.get("fast"), "candidate.fast")
    _require_equal(fast.get("channels"), 64, "candidate.fast.channels")
    _require_equal(fast.get("layers"), 24, "candidate.fast.layers")
    _require_equal(fast.get("kernel_size"), 3, "candidate.fast.kernel")
    _require_equal(
        fast.get("receptive_field_samples_at_48khz"), 8_191, "candidate.fast.rf"
    )
    expected_dilations = [2**index for _ in range(2) for index in range(12)]
    _require_equal(
        fast.get("dilations"), expected_dilations, "candidate.fast.dilations"
    )
    slow = _mapping(candidate.get("slow"), "candidate.slow")
    for field, expected in {"blocks": 8, "channels": 64, "state_dim": 64}.items():
        _require_equal(slow.get(field), expected, f"candidate.slow.{field}")
    control = _mapping(architecture.get("control"), "architecture.control")
    _require_equal(control.get("family"), QUALITY_TEACHER_FAMILIES[1], "control.family")
    _require_equal(control.get("s4_film_forced_to_exact_zero"), True, "control.zero")

    optimization = _mapping(protocol.get("optimization"), "optimization")
    expected_updates = list(range(500, 7_501, 500))
    for field, expected in {
        "optimizer": "AdamW",
        "gradient_clip_norm": 1.0,
        "chunk_samples": 48_000,
        "microchunks_per_update": 3,
        "checkpoint_interval_updates": 500,
        "checkpoint_updates": expected_updates,
        "checkpoint_selection": "minimum_validation_total_loss_only",
    }.items():
        _require_equal(optimization.get(field), expected, f"optimization.{field}")
    loss = _mapping(optimization.get("loss"), "optimization.loss")
    _require_equal(
        dict(loss),
        {
            "l1_weight": 10.0,
            "mrstft_weight": 1.0,
            "preemphasized_esr_weight": 1.0,
            "preemphasis": 0.95,
            "projection_gain_weight": 0.05,
        },
        "optimization.loss",
    )
    curriculum = optimization.get("curriculum")
    _require_equal(
        [row.get("updates") for row in curriculum]
        if isinstance(curriculum, list)
        else None,
        [2_500, 2_500, 2_500],
        "optimization.curriculum",
    )

    comparators = _mapping(protocol.get("comparators"), "comparators")
    _require_equal(
        comparators.get("families"), list(QUALITY_TEACHER_COMPARATORS), "comparators"
    )
    selection = _mapping(comparators.get("global_selection"), "comparators.selection")
    _require_equal(
        selection.get("devices"), list(DEVELOPMENT_DEVICES), "selection.devices"
    )
    _require_equal(selection.get("seed"), 0, "selection.seed")
    _require_equal(
        selection.get("relative_tie_threshold_strictly_less_than"),
        0.01,
        "selection.tie",
    )

    matrix = _mapping(protocol.get("trajectory_matrix"), "trajectory_matrix")
    _require_equal(matrix.get("maximum_trajectories"), 25, "matrix.trajectories")
    _require_equal(matrix.get("gpu_hours_per_trajectory_maximum"), 12, "matrix.per_run")
    _require_equal(matrix.get("scientific_gpu_hours_maximum"), 300, "matrix.science")
    _require_equal(
        matrix.get("preflight_and_evaluation_gpu_hours_reserved"), 24, "matrix.reserve"
    )
    if (
        matrix["maximum_trajectories"] * matrix["gpu_hours_per_trajectory_maximum"]
        != matrix["scientific_gpu_hours_maximum"]
    ):
        raise QualityTeacherConfigError(
            "trajectory matrix no longer equals 300 GPU hours"
        )

    statistics = _mapping(protocol.get("statistics"), "statistics")
    _require_equal(
        statistics.get("bootstrap_replicates"), 10_000, "statistics.bootstrap"
    )
    _require_equal(
        statistics.get("resampling_order_within_device"),
        ["files", "seeds"],
        "statistics.order",
    )
    _require_equal(statistics.get("devices_equal_weight"), True, "statistics.weights")

    verdict = _mapping(protocol.get("objective_verdict"), "objective_verdict")
    for field, expected in {
        "aggregate_esr_relative_improvement_minimum": 0.10,
        "aggregate_esr_bootstrap_lower_95_strictly_greater_than": 0.0,
        "per_confirmation_device_esr_median_improvement_strictly_greater_than": 0.0,
        "aggregate_l1_plus_mrstft_improvement_strictly_greater_than": 0.0,
        "aggregate_l1_plus_mrstft_bootstrap_lower_95_strictly_greater_than": 0.0,
        "secondary_relative_regression_maximum": 0.05,
        "correlation_strictly_greater_than": 0.9,
        "gain_error_strictly_greater_than": -0.2,
    }.items():
        _require_equal(verdict.get(field), expected, f"objective_verdict.{field}")


def make_run_id(stage: str, device: str, family: str, seed: int) -> str:
    if stage not in STAGES or device not in DEVICES or family not in FAMILIES:
        raise QualityTeacherConfigError("run identifier component is outside registry")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise QualityTeacherConfigError("run seed must be a non-negative integer")
    return f"quality_teacher_v1_{stage}_{device}_{family}_seed{seed}_v1"


def parse_run_id(run_id: str) -> QualityTeacherRunSpec:
    match = RUN_ID_PATTERN.fullmatch(run_id)
    if match is None:
        raise QualityTeacherConfigError("invalid quality-teacher run id")
    values = match.groupdict()
    spec = QualityTeacherRunSpec(
        stage=values["stage"],
        device=values["device"],
        family=values["family"],
        seed=int(values["seed"]),
        run_id=run_id,
    )
    if (
        spec.stage not in STAGES
        or spec.device not in DEVICES
        or spec.family not in FAMILIES
    ):
        raise QualityTeacherConfigError("run id names an unregistered component")
    return spec


def validate_stage_authorization(stage: str, gates: Mapping[str, str]) -> None:
    if stage not in STAGES:
        raise QualityTeacherAuthorizationError(f"unknown stage: {stage}")
    index = STAGES.index(stage)
    for predecessor in STAGES[:index]:
        if gates.get(predecessor) != "passed":
            raise QualityTeacherAuthorizationError(
                f"{stage} requires passed gate {predecessor}"
            )
    if stage == "test_open":
        missing = [
            gate for gate in REQUIRED_LOCKS_FOR_TEST if gates.get(gate) != "passed"
        ]
        if missing:
            raise QualityTeacherAuthorizationError(
                f"test_open requires frozen locks: {missing}"
            )


def build_trajectory_matrix(
    selected_candidate: str, global_comparator: str
) -> tuple[QualityTeacherRunSpec, ...]:
    """Materialize the exact 25-trajectory ceiling after the two frozen gates."""
    if selected_candidate not in QUALITY_TEACHER_FAMILIES:
        raise QualityTeacherConfigError("selected candidate is outside registered pair")
    if global_comparator not in QUALITY_TEACHER_COMPARATORS:
        raise QualityTeacherConfigError("global comparator is outside registered set")
    specs: list[QualityTeacherRunSpec] = []

    def add(stage: str, device: str, family: str, seed: int) -> None:
        run_id = make_run_id(stage, device, family, seed)
        specs.append(QualityTeacherRunSpec(stage, device, family, seed, run_id))

    for family in QUALITY_TEACHER_FAMILIES:
        add("ampeg_slow_value", "ampeg", family, 0)
    for device in ("fulltone", "bigmuff"):
        add("development", device, selected_candidate, 0)
    for device in DEVELOPMENT_DEVICES:
        for family in QUALITY_TEACHER_COMPARATORS:
            add("development", device, family, 0)
    for device in CONFIRMATION_DEVICES:
        for seed in (0, 1, 2):
            add("confirmation_train", device, selected_candidate, seed)
            add("confirmation_train", device, global_comparator, seed)
    if len(specs) != 25 or len({spec.run_id for spec in specs}) != 25:
        raise QualityTeacherConfigError("trajectory matrix must contain 25 unique runs")
    return tuple(specs)


def validate_repository_state(root: Path) -> dict[str, Any]:
    """Validate new and superseded canonical state without observing results."""
    protocol = load_protocol(root)
    parent_state = (
        root / ".codex_campaign/amp_sota_prototype_v1_2/STATE.md"
    ).read_text(encoding="utf-8")
    if (
        "préflight passé" not in parent_state
        or "Runs scientifiques v1.2 : 0" not in parent_state
    ):
        raise QualityTeacherConfigError("parent preflight/zero-run record changed")
    if "superseded" not in parent_state:
        raise QualityTeacherConfigError("parent campaign was not superseded")
    return protocol
