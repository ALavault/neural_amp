"""Prospective contract for AMP-SOTA-PROTOTYPE-v1.2."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

CAMPAIGN_VERSION = "AMP-SOTA-PROTOTYPE-v1.2"
CAMPAIGN_KEY = "amp_sota_prototype_v1_2"
PARENT_CAMPAIGN = "AMP-SOTA-PROTOTYPE-v1.1"
PROTOCOL_PATH = Path("configs/amp_sota_prototype_v1_2/protocol.yaml")
CAMPAIGN_PATH = Path(".codex_campaign/amp_sota_prototype_v1_2")
PROTOCOL_LOCK_PATH = CAMPAIGN_PATH / "PROTOCOL_LOCK.yaml"
SYSTEMS = ("dynamic_primary", "two_clippers_primary", "static_primary")
SEEDS = (0, 1, 2)
SNAPSHOT_UPDATES = (500, 1_000, 2_000, 5_000, 10_000, 15_000)
EVALUATION_UPDATES = (10_000, 15_000)
CANDIDATE_FAMILY = "slow_long_tcn_x2"
CONTROL_FAMILY = "slow_long_tcn_x2_zero_modulation"
PROTOCOL_CANONICAL_SHA256 = (
    "47b9ee78b218ddb2e7ec2d8f86adfa0257864ccfcadb48bfd16ce2c8e4c5b7ee"
)
DECISION_SOURCE_FILES = (
    "configs/amp_sota_prototype_v1_2/protocol.yaml",
    ".codex_campaign/amp_sota_prototype_v1_2/PROTOCOL_LOCK.yaml",
    "src/fssr_nam/campaign/amp_sota_prototype_v1_2.py",
    "src/fssr_nam/campaign/amp_sota_v12_gates.py",
    "src/fssr_nam/campaign/amp_sota_v12_registry.py",
    "src/fssr_nam/campaign/quality_aa_provenance.py",
    "src/fssr_nam/data/arch_fixtures.py",
    "src/fssr_nam/data/arch_v3_fixtures.py",
    "src/fssr_nam/data/sota_v12_fixtures.py",
    "src/fssr_nam/metrics/sota_confirmation.py",
    "src/fssr_nam/metrics/spectral.py",
    "src/fssr_nam/metrics/time.py",
    "src/fssr_nam/models/approximants.py",
    "src/fssr_nam/models/arch_v1.py",
    "src/fssr_nam/models/equiripple.py",
    "src/fssr_nam/models/oversampling.py",
    "src/fssr_nam/models/prototype.py",
    "src/fssr_nam/models/r1.py",
    "src/fssr_nam/models/sota_v12.py",
    "src/fssr_nam/models/spline.py",
    "src/fssr_nam/models/structured.py",
    "src/fssr_nam/training/objectives.py",
    "src/fssr_nam/training/sota_v12.py",
)


class SotaV12ConfigError(RuntimeError):
    """Raised when the v1.2 prospective contract is incomplete or changed."""


class SotaV12AuthorizationError(RuntimeError):
    """Raised when a v1.2 stage is attempted before its prerequisite."""


def _require_equal(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise SotaV12ConfigError(
            f"{label} changed: expected {expected!r}, got {value!r}"
        )


def validate_protocol_config(protocol: Mapping[str, Any]) -> None:
    """Reject drift in every decision-bearing v1.2 field."""
    _require_equal(protocol.get("schema_version"), 1, "schema_version")
    _require_equal(protocol.get("campaign_version"), CAMPAIGN_VERSION, "campaign")
    _require_equal(protocol.get("campaign_key"), CAMPAIGN_KEY, "campaign_key")
    _require_equal(protocol.get("parent_campaign"), PARENT_CAMPAIGN, "parent")
    _require_equal(protocol.get("thresholds_changed_from_v1"), False, "thresholds")

    question = protocol.get("scientific_question", {})
    _require_equal(question.get("candidate"), CANDIDATE_FAMILY, "candidate")
    _require_equal(question.get("counterfactual_control"), CONTROL_FAMILY, "control")
    _require_equal(
        question.get("prior_numeric_outputs_used_for_threshold_or_model_selection"),
        False,
        "prior numeric selection",
    )

    boundaries = protocol.get("boundaries", {})
    for key in (
        "parent_runs_resumed_or_retuned",
        "failed_or_invalid_v1_2_run_resume_allowed",
        "result_dependent_retry_allowed",
        "quality_aa_v2_backend_requalified",
        "v1_1_approximants_or_equiripple_reused",
        "synthetic_stage_physical_audio_allowed",
        "fm9_outputs_allowed",
        "human_feedback_allowed_for_selection",
        "remote_push_allowed",
        "preflight_selection_eligible_synthetic_outputs_allowed",
        "test_only_synthetic_fixtures_eligible_for_training_or_selection",
    ):
        _require_equal(boundaries.get(key), False, f"boundaries.{key}")
    _require_equal(
        boundaries.get("physical_audio_samples_maximum_before_slow_value_pass"),
        0,
        "physical boundary",
    )
    for key in (
        "parent_protocols_and_artifacts_immutable",
        "confirmation_outputs_locked_until_candidate_and_native_lock",
        "confirmation_may_open_once",
        "external_report_only_locked",
        "test_only_synthetic_fixtures_may_be_generated",
        "scientific_execution_clean_worktree_required",
    ):
        _require_equal(boundaries.get(key), True, f"boundaries.{key}")

    architecture = protocol.get("architecture", {})
    candidate = architecture.get("candidate", {})
    expected_candidate = {
        "family": CANDIDATE_FAMILY,
        "activation": "tanh",
        "resampler": "kaiser_windowed_sinc",
        "aa_backend": "full_island_x2",
        "slow_control": "causal_zero_order_hold",
        "profile": "max",
        "initial_residual_scale": 0.5,
        "sample_rate_hz": 48_000,
        "channels": 1,
        "precision": "float32",
        "fast_path_receptive_field_samples": 12_283,
        "total_memory": "finite_fast_path_plus_recurrent_slow_state",
        "latency_samples": 32,
    }
    _require_equal(candidate, expected_candidate, "architecture.candidate")
    control = architecture.get("counterfactual_control", {})
    _require_equal(control.get("family"), CONTROL_FAMILY, "architecture.control")
    _require_equal(
        control.get("slow_modulation_forced_to_exact_zero"), True, "control zero"
    )
    _require_equal(control.get("cost_claim_from_control_allowed"), False, "cost")

    data = protocol.get("synthetic_data", {})
    _require_equal(data.get("evidence_tier"), "SYNTHETIC", "evidence tier")
    _require_equal(
        data.get("generator"),
        "fssr_nam.data.sota_v12_fixtures.build_sota_v12_system_episodes",
        "generator",
    )
    _require_equal(data.get("ordered_systems"), list(SYSTEMS), "systems")
    _require_equal(data.get("episode_samples"), 72_000, "episode samples")
    _require_equal(data.get("train_episodes"), 8, "train episodes")
    _require_equal(data.get("internal_dev_episodes"), 4, "dev episodes")
    _require_equal(data.get("train_source_seed"), 40_360_830, "train source")
    _require_equal(data.get("internal_dev_source_seed"), 40_361_830, "dev source")
    _require_equal(
        data.get("slow_value_eval_source_seed"), 40_362_830, "slow-value source"
    )
    _require_equal(data.get("slow_value_eval_episodes"), 4, "slow-value episodes")
    _require_equal(data.get("validation_or_physical_outputs_accessed"), False, "data")

    optimization = protocol.get("optimization", {})
    _require_equal(optimization.get("seeds"), list(SEEDS), "seeds")
    _require_equal(
        optimization.get("snapshot_updates"), list(SNAPSHOT_UPDATES), "snapshots"
    )
    _require_equal(
        optimization.get("evaluation_updates"),
        list(EVALUATION_UPDATES),
        "evaluations",
    )
    for key, expected in {
        "projection_gain_weight": 0.05,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "training_chunk_samples": 8_192,
        "common_preroll_samples_after_alignment": 14_400,
        "declared_latency_samples": 32,
        "scored_start_samples": 14_432,
        "scored_samples_per_episode": 57_568,
        "selected_update": 15_000,
    }.items():
        _require_equal(optimization.get(key), expected, f"optimization.{key}")

    competence = protocol.get("sequential_competence", {})
    _require_equal(competence.get("system_order"), list(SYSTEMS), "gate order")
    _require_equal(competence.get("gain_error_strictly_greater_than"), -0.2, "gain")
    _require_equal(competence.get("correlation_strictly_greater_than"), 0.9, "corr")
    _require_equal(
        competence.get("median_esr_15000_over_10000_maximum"), 1.05, "stability"
    )

    slow_value = protocol.get("slow_value", {})
    _require_equal(slow_value.get("control_family"), CONTROL_FAMILY, "slow control")
    _require_equal(slow_value.get("systems"), list(SYSTEMS), "slow systems")
    _require_equal(slow_value.get("seeds"), list(SEEDS), "slow seeds")
    _require_equal(
        slow_value.get("candidate_checkpoint_reevaluated_without_training"),
        True,
        "slow candidate reevaluation",
    )
    _require_equal(
        slow_value.get("competence_internal_dev_reused_for_slow_value"),
        False,
        "slow evaluation split",
    )
    _require_equal(
        slow_value.get("evaluation_source_seed"), 40_362_830, "slow evaluation seed"
    )
    _require_equal(
        slow_value.get("complete_candidate_seed_triplet_before_stability_decision"),
        True,
        "candidate stability triplet",
    )
    _require_equal(
        slow_value.get("nonfinite_candidate_reevaluation_verdict"),
        "NO-GO-STABILITY-v1.2",
        "candidate stability verdict",
    )
    _require_equal(
        slow_value.get("paired_esr_improvement_median_minimum"), 0.05, "slow ESR"
    )
    _require_equal(
        slow_value.get("median_relative_regression_maximum"), 0.05, "slow metrics"
    )
    _require_equal(
        slow_value.get("nonfinite_control_verdict"),
        "NO-GO-CONTROL-STABILITY-v1.2",
        "slow control stability",
    )
    bootstrap = slow_value.get("bootstrap", {})
    _require_equal(bootstrap.get("replicates"), 10_000, "bootstrap replicates")
    _require_equal(bootstrap.get("seed"), 20_260_830, "bootstrap seed")
    _require_equal(
        bootstrap.get("lower_95_bound_strictly_greater_than"), 0.0, "bootstrap"
    )
    _require_equal(
        bootstrap.get("crossed_seed_and_episode_draws_shared_across_systems"),
        True,
        "crossed bootstrap",
    )

    physical = protocol.get("physical_development", {})
    _require_equal(physical.get("devices"), ["fulltone", "bigmuff"], "dev devices")
    _require_equal(
        physical.get("candidate_median_esr_improvement_minimum"), 0.05, "dev ESR"
    )
    confirmation = protocol.get("confirmation", {})
    _require_equal(confirmation.get("devices"), ["blackstar", "ua1176"], "confirmation")
    _require_equal(confirmation.get("seeds"), [0, 1, 2, 3, 4], "confirmation seeds")
    _require_equal(
        confirmation.get("median_esr_improvement_minimum"), 0.10, "final ESR"
    )
    _require_equal(
        confirmation.get("metric_relative_regression_maximum"), 0.05, "final metrics"
    )
    native = protocol.get("native_parity_runtime", {})
    _require_equal(native.get("block_parity_max_absolute_error"), 2.0e-5, "parity")
    _require_equal(native.get("latency_samples_maximum"), 64, "latency")
    _require_equal(native.get("benchmark_block_size"), 128, "block")
    _require_equal(native.get("p95_realtime_factor_strictly_less_than"), 1.0, "RTF")

    resources = protocol.get("resource_budget", {})
    _require_equal(
        resources.get("gpu_name"),
        "NVIDIA RTX PRO 4000 Blackwell",
        "GPU name",
    )
    _require_equal(
        resources.get("gpu_total_memory_bytes_minimum"), 25_000_000_000, "GPU memory"
    )
    _require_equal(
        resources.get("gpu_hours_per_trajectory_maximum"), 6, "trajectory GPU"
    )
    _require_equal(
        resources.get("candidate_reevaluation_count_maximum"), 9, "reevaluation count"
    )
    _require_equal(
        resources.get("candidate_reevaluation_gpu_hours_per_run_maximum"),
        0.5,
        "reevaluation GPU",
    )
    _require_equal(
        resources.get(
            "slow_value_paired_candidate_eval_plus_control_gpu_hours_maximum"
        ),
        6,
        "slow paired GPU",
    )
    _require_equal(
        resources.get("per_trajectory_wall_clock_limit_enforced"), True, "time limit"
    )
    _require_equal(
        resources.get("budget_rechecked_before_each_trajectory"), True, "budget check"
    )
    _require_equal(
        resources.get("artifact_finalization_reserve_seconds"),
        300,
        "artifact reserve",
    )
    _require_equal(resources.get("v1_2_gpu_hours_maximum"), 108, "v1.2 GPU")
    _require_equal(resources.get("active_goal_gpu_hours_maximum"), 324, "goal GPU")
    _require_equal(resources.get("v1_2_disk_gib_maximum"), 20, "v1.2 disk")
    _require_equal(resources.get("run_disk_gib_maximum"), 0.05, "run disk")
    _require_equal(resources.get("active_goal_disk_gib_maximum"), 50, "goal disk")
    canonical = json.dumps(
        dict(protocol), allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    _require_equal(fingerprint, PROTOCOL_CANONICAL_SHA256, "protocol fingerprint")


def validate_clean_worktree(root: Path) -> None:
    """Require a committed execution snapshot before opening a scientific stage."""
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise SotaV12AuthorizationError(
            "v1.2 scientific execution requires a clean committed worktree"
        )


def validate_cuda_device_properties(
    protocol: Mapping[str, Any], *, name: str, total_memory_bytes: int
) -> None:
    """Reject execution on hardware outside the preregistered CUDA envelope."""
    if not isinstance(name, str) or not name:
        raise SotaV12AuthorizationError("CUDA device name is unavailable")
    if isinstance(total_memory_bytes, bool) or not isinstance(total_memory_bytes, int):
        raise SotaV12AuthorizationError("CUDA device memory is unavailable")
    resources = protocol["resource_budget"]
    if name != resources["gpu_name"]:
        raise SotaV12AuthorizationError(
            f"CUDA device changed: expected {resources['gpu_name']!r}, got {name!r}"
        )
    if total_memory_bytes < int(resources["gpu_total_memory_bytes_minimum"]):
        raise SotaV12AuthorizationError(
            "CUDA device memory is below the frozen minimum"
        )


def load_protocol(root: Path) -> dict[str, Any]:
    path = root / PROTOCOL_PATH
    try:
        protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise SotaV12ConfigError(f"missing v1.2 protocol: {path}") from error
    if not isinstance(protocol, dict):
        raise SotaV12ConfigError("v1.2 protocol must be a mapping")
    validate_protocol_config(protocol)
    return protocol


def _json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SotaV12ConfigError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise SotaV12ConfigError(f"{label} must be a JSON object")
    return value


def confirmation_was_opened(root: Path) -> bool:
    """Detect any prior SOTA confirmation access before v1.2 can proceed."""
    for campaign_dir in sorted((root / ".codex_campaign").glob("amp_sota_prototype*")):
        maturity_path = campaign_dir / "MATURITY.json"
        if maturity_path.is_file():
            maturity = _json_mapping(maturity_path, str(maturity_path))
            if maturity.get("physical_confirmation_accessed") is True:
                return True
        ledger = campaign_dir / "GATE_LEDGER.jsonl"
        if ledger.is_file():
            for line in ledger.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SotaV12ConfigError(
                        f"malformed prior gate ledger {ledger}: {error}"
                    ) from error
                if isinstance(event, dict) and event.get("stage") == "confirmation":
                    return True
    return False


def validate_repository_state(
    root: Path, *, require_frozen: bool = True
) -> dict[str, Any]:
    """Validate v1.2 and every immutable predecessor without reading audio."""
    protocol = load_protocol(root)
    if require_frozen:
        lock = yaml.safe_load((root / PROTOCOL_LOCK_PATH).read_text(encoding="utf-8"))
        _require_equal(lock, protocol, "v1.2 protocol lock")
    for config, lock in (
        (
            "configs/amp_sota_prototype_v1/protocol.yaml",
            ".codex_campaign/amp_sota_prototype_v1/PROTOCOL_LOCK.yaml",
        ),
        (
            "configs/amp_sota_prototype_v1_1/protocol.yaml",
            ".codex_campaign/amp_sota_prototype_v1_1/PROTOCOL_LOCK.yaml",
        ),
    ):
        current = yaml.safe_load((root / config).read_text(encoding="utf-8"))
        frozen = yaml.safe_load((root / lock).read_text(encoding="utf-8"))
        _require_equal(frozen, current, f"immutable predecessor {config}")
    parent = _json_mapping(
        root / ".codex_campaign/amp_sota_prototype_v1_1/VERDICT.json",
        "v1.1 verdict",
    )
    _require_equal(parent.get("verdict"), "NO-GO-MECHANISM", "v1.1 verdict")
    aa = _json_mapping(
        root / ".codex_campaign/quality_aa_v2/VERDICT.json", "AA v2 verdict"
    )
    _require_equal(aa.get("selected_backend"), "full_island_x2", "AA backend")
    if confirmation_was_opened(root):
        raise SotaV12AuthorizationError(
            "a SOTA confirmation has already been opened; user direction is required"
        )
    return protocol


def validate_stage_authorization(stage: str, decisions: Mapping[str, str]) -> None:
    prerequisites = {
        "preflight": (),
        "competence_dynamic": (("preflight", "passed"),),
        "competence_two_clippers": (("competence_dynamic", "passed"),),
        "competence_static": (("competence_two_clippers", "passed"),),
        "slow_value": (("competence_static", "passed"),),
        "physical_development": (("slow_value", "passed"),),
        "candidate_lock": (("physical_development", "passed"),),
        "native_parity_runtime": (("candidate_lock", "passed"),),
        "confirmation": (("native_parity_runtime", "passed"),),
        "prototype_audit": (("confirmation", "passed"),),
    }
    if stage not in prerequisites:
        raise SotaV12AuthorizationError(f"unknown v1.2 stage: {stage}")
    for required_stage, required_status in prerequisites[stage]:
        if decisions.get(required_stage) != required_status:
            raise SotaV12AuthorizationError(
                f"{stage} requires {required_stage}={required_status}"
            )


def make_run_id(*, stage: str, system: str, family: str, seed: int) -> str:
    if stage not in {"competence", "slow_value", "slow_value_eval"}:
        raise ValueError("unknown v1.2 synthetic run stage")
    expected_family = {
        "competence": CANDIDATE_FAMILY,
        "slow_value": CONTROL_FAMILY,
        "slow_value_eval": CANDIDATE_FAMILY,
    }[stage]
    if system not in SYSTEMS or family != expected_family:
        raise ValueError("unknown v1.2 system or family")
    if seed not in SEEDS or isinstance(seed, bool):
        raise ValueError("unknown v1.2 seed")
    return f"sota_v1_2_{stage}_{system}_{family}_seed{seed}_v1"
