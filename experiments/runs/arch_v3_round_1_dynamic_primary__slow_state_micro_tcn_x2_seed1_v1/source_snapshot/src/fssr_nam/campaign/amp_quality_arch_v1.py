"""Prospective contract and immutable identifiers for AMP-QUALITY-ARCH-v1."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CAMPAIGN_VERSION = "AMP-QUALITY-ARCH-v1"
CAMPAIGN_KEY = "amp_quality_arch_v1"
PARENT_CAMPAIGN = "FSSR-QUALITY-AA-v2"
PROTOCOL_PATH = Path("configs/amp_quality_arch_v1/protocol.yaml")
CAMPAIGN_PATH = Path(".codex_campaign/amp_quality_arch_v1")

STAGES = (
    "preflight",
    "mechanism",
    "loss_qualify",
    "screen",
    "teacher",
    "distill",
    "robustness",
    "lock",
    "native",
    "confirm_train",
    "confirm_test",
    "listen",
    "audit",
)
DEVICES = ("synthetic", "host_cpu", "fulltone", "bigmuff", "blackstar", "ua1176", "all")
COMPARATORS = (
    "nam_a2_full",
    "wright_lstm64",
    "nablafx_tcn_tfilm",
    "nablafx_s4_tfilm",
)
CANDIDATES = (
    "selective_s6_x2",
    "micro_tcn_x2",
    "phys_s6_tcn_x2",
    "phys_det_tcn_x2",
    "rf2047_tfilm_x2",
    "cascade_rf2047_tfilm_x2",
)
FAMILIES = (*COMPARATORS, *CANDIDATES)
TRAINING_CANDIDATES = CANDIDATES[1:]
TRAINING_FAMILIES = (*COMPARATORS, *TRAINING_CANDIDATES)
TEACHER_CANDIDATES = ("micro_tcn_x2", "phys_s6_tcn_x2")
LOSSES = ("m4", "wright", "nablafx")
_META_FAMILIES = ("protocol", "all")
_META_LOSSES = ("none",)


def _alternatives(values: tuple[str, ...]) -> str:
    return "|".join(re.escape(value) for value in sorted(values, key=len, reverse=True))


RUN_ID_PATTERN = re.compile(
    rf"^arch_v1_(?P<stage>{_alternatives(STAGES)})_"
    rf"(?P<device>{_alternatives(DEVICES)})_"
    rf"(?P<family>{_alternatives((*FAMILIES, *_META_FAMILIES))})_"
    rf"(?P<loss>{_alternatives((*LOSSES, *_META_LOSSES))})_"
    r"seed(?P<seed>0|[1-9][0-9]*)_v1$"
)


class ArchConfigError(RuntimeError):
    """Raised when the prospective architecture contract drifts."""


class ArchAuthorizationError(RuntimeError):
    """Raised when a sequential stage or sealed boundary is not authorized."""


@dataclass(frozen=True)
class ArchRunSpec:
    stage: str
    device: str
    family: str
    loss: str
    seed: int
    run_id: str


def _require_equal(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ArchConfigError(f"{label} must equal {expected!r}, got {value!r}")


def _require_text_keys(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                location = "/".join(path) or "<root>"
                raise ArchConfigError(f"mapping key at {location} must be text")
            _require_text_keys(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_text_keys(item, (*path, str(index)))


def load_protocol(root: Path) -> dict[str, Any]:
    path = root / PROTOCOL_PATH
    if not path.is_file():
        raise ArchConfigError(f"missing architecture protocol: {path}")
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ArchConfigError("architecture protocol must be a mapping")
    _require_text_keys(protocol)
    validate_protocol_config(protocol)
    return protocol


def validate_protocol_config(protocol: Mapping[str, Any]) -> None:
    """Fail closed on the decision-bearing architecture protocol fields."""
    for key, expected in {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "campaign_key": CAMPAIGN_KEY,
        "parent_campaign": PARENT_CAMPAIGN,
        "sample_rate_hz": 48_000,
        "channels": 1,
        "precision": "float32",
        "run_id_format": "arch_v1_<stage>_<device>_<family>_<loss>_seed<n>_v1",
        "valid_terminal_verdicts": ["GO-ARCH", "NO-GO-ARCH", "INVALID"],
    }.items():
        _require_equal(protocol.get(key), expected, key)

    claim = protocol.get("claim_boundary", {})
    _require_equal(
        claim.get("physical_aliasing_claim_allowed"), False, "claim.physical_aliasing"
    )
    _require_equal(
        claim.get("proprietary_global_sota_claim_allowed"), False, "claim.global_sota"
    )
    _require_equal(
        claim.get("external_report_only_locked"), True, "claim.external_lock"
    )
    _require_equal(claim.get("physical_192khz_dataset_assumed"), False, "claim.192khz")
    _require_equal(claim.get("fm9_capture_allowed"), False, "claim.fm9")

    data = protocol.get("data", {})
    _require_equal(
        data.get("internal_dev_devices"), ["fulltone", "bigmuff"], "data.internal_dev"
    )
    _require_equal(
        data.get("prospective_devices"), ["blackstar", "ua1176"], "data.prospective"
    )
    _require_equal(
        data.get("historical_test_exposure_disclosed"),
        ["fulltone", "bigmuff"],
        "data.historical",
    )
    _require_equal(
        data.get("sealed_test_outputs"), ["blackstar", "ua1176"], "data.sealed"
    )
    _require_equal(data.get("sealed_openings_maximum"), 1, "data.openings")

    source_pins = protocol.get("source_pins", {})
    _require_equal(
        source_pins,
        {
            "nam_trainer": {
                "path": "third_party/neural-amp-modeler",
                "commit": "f26112906de06ec6b796ad6d1982e29eed83144e",
            },
            "nam_core": {
                "path": "third_party/NeuralAmpModelerCore",
                "commit": "1f42f88535884450104b8711d7595019afa0495b",
            },
            "wright": {
                "path": "third_party/Automated-GuitarAmpModelling",
                "commit": "e3146386b0fd0b562bc393231be3a5938cf9feac",
            },
            "nablafx": {
                "path": "third_party/nablafx",
                "commit": "045db6e7d6087151c7e3a264844bd8c4eafc885c",
            },
            "auraloss": {
                "path": "third_party/auraloss",
                "url": "https://github.com/csteinmetz1/auraloss.git",
                "tag": "v0.4.0",
                "commit": "1576b0cd6e927abc002b23cf3bfc455b660f663c",
            },
            "selective_s6": {
                "path": "third_party/optical-selective-ssm",
                "url": "https://github.com/RiccardoVib/Optical-DRC-with-Selective-SSMs.git",
                "commit": "bd787ecc60024cf364efd4cef07f692af05277e6",
                "local_implementation_required": False,
                "audited_rejected_before_freeze": True,
            },
        },
        "source_pins",
    )

    comparators = protocol.get("comparators", {})
    _require_equal(comparators.get("maximum_families"), 4, "comparators.maximum")
    _require_equal(
        comparators.get("families"), list(COMPARATORS), "comparators.families"
    )
    _require_equal(
        comparators.get("technical_rejections_before_freeze"),
        {
            "selective_s6_glu": (
                "pinned_primary_source_is_not_executable_or_reproducible"
            )
        },
        "comparators.technical_rejections",
    )
    _require_equal(
        comparators.get("per_device_test_selection_forbidden"),
        True,
        "comparators.test_selection",
    )
    _require_equal(
        comparators.get("variants"),
        {
            "nam_a2_full": ["official"],
            "wright_lstm64": ["official"],
            "nablafx_tcn_tfilm": ["small", "large"],
            "nablafx_s4_tfilm": ["small", "large"],
        },
        "comparators.variants",
    )
    _require_equal(
        comparators.get("nablafx_variant_selection"),
        "pooled_fulltone_bigmuff_validation_never_per_device",
        "comparators.variant_selection",
    )
    _require_equal(
        comparators.get("nablafx_model_trajectory_count_at_1000_updates"),
        16,
        "comparators.variant_trajectories",
    )
    _require_equal(
        comparators.get("nablafx_top_level_run_id_count"),
        8,
        "comparators.variant_run_ids",
    )
    _require_equal(
        comparators.get("nablafx_variant_artifacts_are_ledgered_subtrajectories"),
        True,
        "comparators.variant_ledger",
    )
    _require_equal(
        comparators.get("conservative_causal_latency_samples"),
        {
            "nam_a2_full": {"official": 0},
            "wright_lstm64": {"official": 0},
            "nablafx_tcn_tfilm": {"small": 635, "large": 1_270},
            "nablafx_s4_tfilm": {"small": 508, "large": 1_016},
        },
        "comparators.latency",
    )
    _require_equal(
        comparators.get("comparator_latency_is_aligned_in_training_and_evaluation"),
        True,
        "comparators.latency_alignment",
    )
    _require_equal(
        comparators.get("deployable_runtime_gate_applies_to_comparators"),
        False,
        "comparators.runtime_scope",
    )

    candidates = protocol.get("candidates", {})
    _require_equal(candidates.get("maximum_families"), 12, "candidates.maximum")
    frozen = candidates.get("frozen_families")
    _require_equal(frozen, list(CANDIDATES), "candidates.frozen")
    if len(frozen) > candidates["maximum_families"]:
        raise ArchConfigError("candidate family cap exceeded")
    _require_equal(
        candidates.get("primary_candidate_has_no_selection_privilege"),
        True,
        "candidates.no_privilege",
    )
    _require_equal(
        candidates.get("teacher_maximum_finalists"), 3, "candidates.teachers"
    )
    _require_equal(
        candidates.get("deployable_finalists_maximum"), 1, "candidates.deployable"
    )
    _require_equal(
        candidates.get("deployable_width_profiles"),
        {
            "slim": {"channels": 12, "observer_state_dim": 32},
            "balanced": {"channels": 16, "observer_state_dim": 48},
            "full": {"channels": 24, "observer_state_dim": 64},
            "max": {"channels": 32, "observer_state_dim": 96},
        },
        "candidates.width_profiles",
    )
    _require_equal(
        candidates.get("deployable_profile_selection"),
        "largest_profile_passing_blind_native_skeleton_gate",
        "candidates.profile_selection",
    )
    _require_equal(
        candidates.get("deployable_profile_selection_uses_audio"),
        False,
        "candidates.profile_audio",
    )
    _require_equal(
        candidates.get("deployable_profile_selection_evidence"),
        ".codex_campaign/amp_quality_arch_v1/NATIVE_SKELETON.json",
        "candidates.profile_evidence",
    )
    _require_equal(
        candidates.get("selected_deployable_profiles"),
        {family: "max" for family in CANDIDATES},
        "candidates.selected_profiles",
    )
    _require_equal(
        candidates.get("training_feasibility_evidence"),
        ".codex_campaign/amp_quality_arch_v1/TRAINING_FEASIBILITY.json",
        "candidates.training_evidence",
    )
    _require_equal(
        candidates.get("training_eligible_families"),
        list(TRAINING_CANDIDATES),
        "candidates.training_eligible",
    )
    _require_equal(
        candidates.get("technical_rejections_before_freeze"),
        {"selective_s6_x2": ("projected_training_time_exceeds_per_trajectory_cap")},
        "candidates.technical_rejections",
    )
    _require_equal(
        candidates.get("result_dependent_replacement_family_allowed"),
        False,
        "candidates.replacement",
    )
    _require_equal(
        candidates.get("teacher_eligible"),
        list(TEACHER_CANDIDATES),
        "candidates.teacher_eligible",
    )
    _require_equal(
        candidates.get("observer_outside_x2_island"), True, "candidates.observer_island"
    )
    _require_equal(
        candidates.get("observer_cadence_samples"), 64, "candidates.observer_cadence"
    )

    bus = protocol.get("physical_state_bus", {})
    _require_equal(
        bus.get("wet_descriptors_as_inference_inputs"),
        False,
        "physical_bus.no_target_leak",
    )
    _require_equal(
        bus.get("auxiliary_weight_candidates"),
        [0.01, 0.05, 0.10],
        "physical_bus.weights",
    )
    if any(str(name).startswith("wet_") for name in bus.get("inference_inputs", [])):
        raise ArchConfigError("wet descriptors cannot be inference inputs")

    losses = protocol.get("losses", {})
    pairs = losses.get("per_family_allowed_pairs", {})
    _require_equal(set(pairs), set(TRAINING_FAMILIES), "losses.family_keys")
    for family, allowed in pairs.items():
        if (
            len(allowed) != 2
            or len(set(allowed)) != 2
            or not set(allowed) <= set(LOSSES)
        ):
            raise ArchConfigError(
                f"{family} must declare exactly two distinct allowed losses"
            )
    _require_equal(
        losses.get("selection_checkpoint_updates"), 1_000, "losses.selection_checkpoint"
    )
    _require_equal(
        losses.get("nonselected_loss_stop_updates"), 1_000, "losses.nonselected_stop"
    )
    _require_equal(
        losses.get("selected_loss_stop_updates"), 5_000, "losses.selected_stop"
    )
    _require_equal(
        losses.get("qualification_top_level_run_id_count_maximum"),
        36,
        "losses.trajectory_count_maximum",
    )
    _require_equal(
        losses.get("qualification_model_subtrajectory_count_maximum"),
        44,
        "losses.model_subtrajectory_count_maximum",
    )
    _require_equal(
        losses.get("actual_top_level_count_formula"),
        "4_times_4_comparators_plus_mechanism_eligible_candidates",
        "losses.actual_count_formula",
    )
    _require_equal(
        losses.get("mechanism_rejected_candidates_are_not_trained"),
        True,
        "losses.mechanism_pruning",
    )

    _require_equal(
        protocol.get("physical_training"),
        {
            "evidence": (
                ".codex_campaign/amp_quality_arch_v1/PHYSICAL_TRAINING_FEASIBILITY.json"
            ),
            "output_samples_per_update": 8_192,
            "batch_size": 1,
            "context_samples": {
                "nam_a2_full": {"official": 6_346},
                "wright_lstm64": {"official": 49_152},
                "nablafx_tcn_tfilm": {"small": 133_967, "large": 119_366},
                "nablafx_s4_tfilm": {"small": 135_808, "large": 135_808},
                "candidates": 49_152,
            },
            "context_rationale": {
                "nablafx_tcn_tfilm": (
                    "receptive_field_minus_one_plus_conservative_tfilm_latency"
                ),
                "nablafx_s4_tfilm": (
                    "pinned_144000_sample_training_segment_minus_output"
                ),
                "stateful_other": (
                    "at_least_four_times_longest_240ms_fixture_time_constant"
                ),
            },
            "optimizer_by_family": {
                "nam_a2_full": {
                    "name": "Adam",
                    "learning_rate": 0.004,
                    "weight_decay": 3.17e-7,
                },
                "wright_lstm64": {
                    "name": "Adam",
                    "learning_rate": 0.005,
                    "weight_decay": 1.0e-4,
                },
                "nablafx_tcn_tfilm": {
                    "name": "Adam",
                    "learning_rate": 0.005,
                    "weight_decay": 0.0,
                },
                "nablafx_s4_tfilm": {
                    "name": "Adam",
                    "learning_rate": 0.01,
                    "weight_decay": 0.0,
                },
                "candidates": {
                    "name": "AdamW",
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                },
            },
            "gradient_clip_norm": 1.0,
            "random_window_seed": 20_260_828,
            "validation_samples": 240_000,
            "validation_source_order_fixed": True,
            "validation_checkpoint_selection_only": True,
            "comparator_latency_target_alignment": (
                "explicit_zero_padded_delay_before_tail_crop"
            ),
            "finite_output_loss_and_gradients_required": True,
            "maximum_median_seconds_per_update": 4.0,
            "maximum_projected_hours_per_5000_update_trajectory": 6.0,
            "feasibility_warmup_iterations": 1,
            "feasibility_measured_iterations": 2,
            "physical_audio_samples_read_by_feasibility": 0,
        },
        "physical_training",
    )

    stages = protocol.get("stages", {})
    _require_equal(stages.get("ordered"), list(STAGES), "stages.ordered")
    _require_equal(
        stages.get("common_checkpoints"),
        [200, 1_000, 5_000, 15_000],
        "stages.checkpoints",
    )
    _require_equal(stages.get("final_seeds"), [0, 1, 2, 3, 4], "stages.seeds")
    _require_equal(
        stages.get("final_devices"),
        ["fulltone", "bigmuff", "blackstar", "ua1176"],
        "stages.devices",
    )
    _require_equal(stages.get("final_condition_count"), 40, "stages.conditions")
    _require_equal(
        stages.get("result_dependent_retry_allowed"), False, "stages.retries"
    )

    mechanism = protocol.get("mechanism_gate", {})
    _require_equal(
        mechanism.get("native_skeleton_required_before_physical_training"),
        True,
        "mechanism.native_skeleton",
    )
    _require_equal(
        mechanism.get("native_skeleton_block64_p95_rtf_maximum"),
        0.80,
        "mechanism.native_rtf",
    )
    _require_equal(
        mechanism.get("native_skeleton_repetitions"),
        30,
        "mechanism.native_repetitions",
    )
    _require_equal(
        mechanism.get("all_deployable_width_profiles_benchmarked"),
        True,
        "mechanism.width_profiles",
    )
    _require_equal(
        mechanism.get("failure_scope"),
        "prune_failed_candidate_family_without_replacement",
        "mechanism.failure_scope",
    )
    _require_equal(
        mechanism.get("minimum_training_eligible_candidates_after_gate"),
        2,
        "mechanism.minimum_candidates",
    )
    _require_equal(
        mechanism.get("anti_collapse_gain_error_minimum"),
        -0.2,
        "mechanism.gain_guard",
    )
    _require_equal(
        mechanism.get("anti_collapse_correlation_minimum"),
        0.9,
        "mechanism.correlation_guard",
    )
    _require_equal(
        protocol.get("mechanism_training"),
        {
            "profile": "max",
            "seed": 20_260_828,
            "episode_samples": 8_192,
            "train_episodes": 16,
            "validation_episodes": 4,
            "loss_tail_samples": 2_048,
            "updates_per_trajectory": 500,
            "batch_size": 1,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "audio_loss": "esr",
            "trajectory_count": 10,
            "source_seed_train": 20_260_828,
            "source_seed_validation": 20_261_828,
            "systems": {
                "static_composite": {
                    "components": ["tanh", "asymmetric_clip"],
                    "families": [
                        "micro_tcn_x2",
                        "phys_det_tcn_x2",
                        "phys_s6_tcn_x2",
                    ],
                },
                "dynamic_composite": {
                    "components": [
                        "rf2047_memory",
                        "slow_sag",
                        "attack_release",
                        "blocking_distortion",
                        "level_transition",
                    ],
                    "families": [
                        "micro_tcn_x2",
                        "phys_det_tcn_x2",
                        "phys_s6_tcn_x2",
                    ],
                },
                "two_clippers": {
                    "components": ["two_clippers"],
                    "families": [
                        "rf2047_tfilm_x2",
                        "cascade_rf2047_tfilm_x2",
                    ],
                },
            },
            "phys_s6_auxiliary_weight_candidates": [0.01, 0.05, 0.10],
            "phys_s6_auxiliary_weight_selection": (
                "lowest_dynamic_validation_esr_then_lowest_weight"
            ),
            "phys_s6_static_uses_selected_auxiliary_weight": True,
            "no_other_hyperparameter_selection": True,
        },
        "mechanism_training",
    )
    _require_equal(
        protocol.get("native_skeleton_baseline"),
        {
            "family": "nam_a2_full",
            "model_path": "experiments/runs/m2_a2_tanh_seed0_v3/model_full.nam",
            "parameters": 12145,
            "weight_bytes": 48580,
            "host_cpu": "Intel_Xeon_E5-2630_v3",
            "required_isa": ["avx2", "fma"],
            "compiler": "GNU_15.2.0",
            "build_type": "Release",
            "compiler_flags": ["-Ofast", "-march=native", "LTO_IPO"],
        },
        "native_skeleton_baseline",
    )
    _require_equal(
        protocol.get("training_feasibility_gate"),
        {
            "input_source": "deterministic_synthetic_probe",
            "physical_audio_samples_read": 0,
            "selected_profile": "max",
            "segment_samples": 2048,
            "batch_size": 1,
            "warmup_iterations": 1,
            "measured_iterations": 2,
            "projected_updates": 5000,
            "maximum_median_seconds_per_update": 4.0,
            "maximum_projected_hours_per_trajectory": 6.0,
            "finite_output_and_gradients_required": True,
            "rejection_before_protocol_freeze_allowed": True,
            "result_dependent_replacement_family_allowed": False,
        },
        "training_feasibility_gate",
    )

    resources = protocol.get("resource_budget", {})
    _require_equal(
        resources.get("training_run_directories_maximum"), 88, "resources.run_cap"
    )
    _require_equal(resources.get("gpu_hours_total_maximum"), 240, "resources.gpu_hours")
    _require_equal(
        resources.get("gpu_hours_per_trajectory_maximum"),
        6,
        "resources.trajectory_hours",
    )
    _require_equal(
        resources.get("budget_from_failed_branch_reallocation_allowed"),
        False,
        "resources.reallocation",
    )

    runtime = protocol.get("runtime_gate", {})
    _require_equal(runtime.get("primary_block"), 64, "runtime.primary_block")
    _require_equal(runtime.get("interleaved_ab_repetitions"), 30, "runtime.repetitions")
    _require_equal(runtime.get("p95_rtf_maximum"), 0.80, "runtime.rtf")
    _require_equal(runtime.get("added_latency_samples_maximum"), 48, "runtime.latency")
    _require_equal(runtime.get("python_cpp_max_abs_error"), 2.0e-5, "runtime.parity")
    _require_equal(
        runtime.get("audio_loop_allocations_maximum"), 0, "runtime.allocations"
    )

    statistics = protocol.get("statistics", {})
    _require_equal(
        statistics.get("bootstrap_replicates"), 10_000, "statistics.replicates"
    )
    _require_equal(statistics.get("bootstrap_seed"), 20_260_828, "statistics.seed")
    _require_equal(
        statistics.get("hierarchical_units"), ["seed", "source"], "statistics.units"
    )
    _require_equal(
        statistics.get("windows_as_independent_units"), False, "statistics.windows"
    )

    final_gate = protocol.get("final_gate", {})
    _require_equal(
        final_gate.get("esr_relative_improvement_lower_95_bound_minimum"),
        0.15,
        "final.esr",
    )
    _require_equal(final_gate.get("devices_won_minimum"), 3, "final.devices")
    _require_equal(
        final_gate.get("mandatory_device_wins"),
        ["blackstar", "ua1176"],
        "final.mandatory",
    )
    _require_equal(final_gate.get("asr_gain_db_minimum"), 10.0, "final.asr")


def validate_repository_state(
    root: Path, *, require_frozen: bool = False, require_native_evidence: bool = True
) -> dict[str, Any]:
    """Validate the draft/frozen protocol and sealed administrative boundary."""
    protocol = load_protocol(root)
    active = (
        (root / ".codex_campaign/ACTIVE_CAMPAIGN").read_text(encoding="utf-8").strip()
    )
    _require_equal(active, CAMPAIGN_KEY, "active campaign")
    lineages = json.loads(
        (root / ".codex_campaign/LINEAGES.json").read_text(encoding="utf-8")
    )
    _require_equal(lineages.get("active"), CAMPAIGN_KEY, "lineage active")
    entry = lineages.get("lineages", {}).get(CAMPAIGN_KEY, {})
    _require_equal(entry.get("campaign_version"), CAMPAIGN_VERSION, "lineage version")
    _require_equal(entry.get("parent"), "quality_aa_v2", "lineage parent")

    if require_native_evidence:
        native_evidence_path = (
            root / protocol["candidates"]["deployable_profile_selection_evidence"]
        )
        if not native_evidence_path.is_file():
            raise ArchConfigError("native skeleton profile evidence is missing")
        native_evidence = json.loads(native_evidence_path.read_text(encoding="utf-8"))
        _require_equal(
            native_evidence.get("status"), "passed", "native evidence status"
        )
        _require_equal(
            native_evidence.get("physical_audio_samples_read"),
            0,
            "native evidence physical audio",
        )
        _require_equal(
            native_evidence.get("selection_uses_audio_or_esr"),
            False,
            "native evidence selection inputs",
        )
        _require_equal(
            native_evidence.get("selected_profiles"),
            protocol["candidates"]["selected_deployable_profiles"],
            "native evidence selected profiles",
        )
        training_evidence_path = (
            root / protocol["candidates"]["training_feasibility_evidence"]
        )
        if not training_evidence_path.is_file():
            raise ArchConfigError("training feasibility evidence is missing")
        training_evidence = json.loads(
            training_evidence_path.read_text(encoding="utf-8")
        )
        _require_equal(
            training_evidence.get("status"),
            "passed_with_rejections",
            "training evidence status",
        )
        _require_equal(
            training_evidence.get("physical_audio_samples_read"),
            0,
            "training evidence physical audio",
        )
        _require_equal(
            training_evidence.get("passed_families"),
            list(TRAINING_CANDIDATES),
            "training evidence passed families",
        )
        _require_equal(
            training_evidence.get("rejected_families"),
            ["selective_s6_x2"],
            "training evidence rejected families",
        )
        physical_evidence_path = root / protocol["physical_training"]["evidence"]
        if not physical_evidence_path.is_file():
            raise ArchConfigError("physical training feasibility evidence is missing")
        physical_evidence = json.loads(
            physical_evidence_path.read_text(encoding="utf-8")
        )
        _require_equal(
            physical_evidence.get("status"),
            "passed",
            "physical training evidence status",
        )
        _require_equal(
            physical_evidence.get("physical_audio_samples_read"),
            0,
            "physical training evidence audio",
        )
        _require_equal(
            physical_evidence.get("selection_uses_audio_or_esr"),
            False,
            "physical training evidence selection inputs",
        )
        _require_equal(
            physical_evidence.get("workload_count"),
            11,
            "physical training evidence workloads",
        )
        if not all(
            result.get("passed") is True
            for result in physical_evidence.get("results", {}).values()
        ):
            raise ArchConfigError("a physical training workload failed feasibility")

    parent_verdict = json.loads(
        (root / ".codex_campaign/quality_aa_v2/VERDICT.json").read_text(
            encoding="utf-8"
        )
    )
    _require_equal(
        parent_verdict.get("selected_backend"), "full_island_x2", "parent backend"
    )
    _require_equal(parent_verdict.get("verdict"), "GO-QUALITY-AA-v2", "parent verdict")

    freeze = json.loads(
        (root / CAMPAIGN_PATH / "EXTERNAL_FREEZE.json").read_text(encoding="utf-8")
    )
    _require_equal(freeze.get("external_report_only_locked"), True, "freeze.external")
    _require_equal(
        freeze.get("blackstar_test_output_accessed"), False, "freeze.blackstar"
    )
    _require_equal(freeze.get("ua1176_test_output_accessed"), False, "freeze.ua1176")
    _require_equal(
        freeze.get("fulltone_bigmuff_historical_test_exposure_disclosed"),
        True,
        "freeze.history",
    )

    lock_path = root / CAMPAIGN_PATH / "PROTOCOL_LOCK.yaml"
    if lock_path.is_file():
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        _require_equal(lock, protocol, "protocol lock")
        _require_equal(
            protocol.get("status"),
            "frozen_before_first_scientific_run",
            "protocol status",
        )
    elif require_frozen:
        raise ArchConfigError("frozen protocol lock is required")
    else:
        _require_equal(
            protocol.get("status"),
            "prospective_draft_to_freeze_before_first_scientific_run",
            "protocol status",
        )
    return protocol


def _check_stage_shape(
    stage: str, device: str, family: str, loss: str, seed: int
) -> None:
    if stage == "preflight" and (device, family, loss, seed) != (
        "all",
        "protocol",
        "none",
        0,
    ):
        raise ValueError("preflight run shape is fixed")
    if stage == "mechanism" and (
        device != "synthetic"
        or family not in TRAINING_CANDIDATES
        or loss != "none"
        or seed != 0
    ):
        raise ValueError("mechanism run shape is invalid")
    if stage in {"loss_qualify", "screen"}:
        if (
            device not in {"fulltone", "bigmuff"}
            or family not in TRAINING_FAMILIES
            or loss not in LOSSES
            or seed != 0
        ):
            raise ValueError(f"{stage} run shape is invalid")
    if stage == "teacher":
        if (
            device not in {"fulltone", "bigmuff"}
            or family not in TEACHER_CANDIDATES
            or loss not in LOSSES
            or seed != 0
        ):
            raise ValueError("teacher run shape is invalid")
    if stage == "distill":
        if (
            device not in {"fulltone", "bigmuff"}
            or family not in TEACHER_CANDIDATES
            or loss not in LOSSES
            or seed != 0
        ):
            raise ValueError("distill run shape is invalid")
    if stage == "robustness":
        if (
            device not in {"fulltone", "bigmuff"}
            or family not in TRAINING_FAMILIES
            or loss not in LOSSES
            or seed not in {1, 2, 3, 4}
        ):
            raise ValueError("robustness run shape is invalid")
    if stage == "lock" and (device, family, loss, seed) != ("all", "all", "none", 0):
        raise ValueError("lock run shape is fixed")
    if stage == "native":
        if (
            device != "host_cpu"
            or family not in TRAINING_FAMILIES
            or loss != "none"
            or seed != 0
        ):
            raise ValueError("native run shape is invalid")
    if stage == "confirm_train":
        if (
            device not in {"blackstar", "ua1176"}
            or family not in TRAINING_FAMILIES
            or loss not in LOSSES
            or seed not in {0, 1, 2, 3, 4}
        ):
            raise ValueError("confirm_train run shape is invalid")
    if stage in {"confirm_test", "listen", "audit"} and (
        device,
        family,
        loss,
        seed,
    ) != ("all", "all", "none", 0):
        raise ValueError(f"{stage} run shape is fixed")


def make_run_id(stage: str, device: str, family: str, loss: str, seed: int) -> str:
    if stage not in STAGES:
        raise ValueError(f"unknown architecture stage: {stage}")
    if (
        device not in DEVICES
        or family not in {*FAMILIES, *_META_FAMILIES}
        or loss not in {*LOSSES, *_META_LOSSES}
    ):
        raise ValueError("run identifier contains an invalid component")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    _check_stage_shape(stage, device, family, loss, seed)
    run_id = f"arch_v1_{stage}_{device}_{family}_{loss}_seed{seed}_v1"
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("invalid AMP-QUALITY-ARCH-v1 run_id")
    return run_id


def parse_run_id(run_id: str) -> ArchRunSpec:
    match = RUN_ID_PATTERN.fullmatch(run_id)
    if match is None:
        raise ValueError("invalid AMP-QUALITY-ARCH-v1 run_id")
    values = match.groupdict()
    seed = int(values["seed"])
    canonical = make_run_id(
        values["stage"], values["device"], values["family"], values["loss"], seed
    )
    if canonical != run_id:
        raise ValueError("run_id is not canonical")
    return ArchRunSpec(
        values["stage"],
        values["device"],
        values["family"],
        values["loss"],
        seed,
        run_id,
    )


def loss_qualification_specs(protocol: Mapping[str, Any]) -> tuple[ArchRunSpec, ...]:
    pairs = protocol["losses"]["per_family_allowed_pairs"]
    specs: list[ArchRunSpec] = []
    for device in ("fulltone", "bigmuff"):
        for family in TRAINING_FAMILIES:
            for loss in pairs[family]:
                run_id = make_run_id("loss_qualify", device, family, loss, 0)
                specs.append(
                    ArchRunSpec("loss_qualify", device, family, loss, 0, run_id)
                )
    if len(specs) != 36 or len({spec.run_id for spec in specs}) != 36:
        raise ArchConfigError(
            "loss qualification matrix must contain 36 unique trajectories"
        )
    return tuple(specs)


_STAGE_REQUIREMENTS = {
    "mechanism": ("preflight", "passed"),
    "loss_qualify": ("mechanism", "passed"),
    "screen": ("loss_qualify", "passed"),
    "teacher": ("screen", "passed"),
    "distill": ("teacher", "passed"),
    "robustness": ("deployable", "passed"),
    "lock": ("robustness", "passed"),
    "native": ("lock", "passed"),
    "confirm_train": ("native", "passed"),
    "confirm_test": ("confirm_train", "passed"),
    "listen": ("objective_runtime", "passed"),
    "audit": ("listen", "passed"),
}


def validate_stage_authorization(stage: str, decisions: Mapping[str, str]) -> None:
    if stage == "preflight":
        return
    if stage not in _STAGE_REQUIREMENTS:
        raise ValueError(f"unknown architecture stage: {stage}")
    key, expected = _STAGE_REQUIREMENTS[stage]
    if decisions.get(key) != expected:
        raise ArchAuthorizationError(f"{stage} requires literal {key}={expected}")


def validate_distillation_authorization(decisions: Mapping[str, str]) -> None:
    if decisions.get("teacher") != "passed":
        raise ArchAuthorizationError("distillation requires teacher=passed")
    if decisions.get("deployable") not in {
        "failed_fidelity_gate",
        "failed_runtime_gate",
    }:
        raise ArchAuthorizationError(
            "distillation requires a valid failed deployable gate"
        )


def validate_sealed_test_boundary(
    *,
    decisions: Mapping[str, str],
    blackstar_open_count: int,
    ua1176_open_count: int,
    external_report_only_locked: bool,
) -> None:
    for key in ("lock", "python_cpp_parity", "benchmark", "confirm_train"):
        if decisions.get(key) != "passed":
            raise ArchAuthorizationError(f"sealed test requires {key}=passed")
    if blackstar_open_count != 0 or ua1176_open_count != 0:
        raise ArchAuthorizationError(
            "Blackstar and UA1176 sealed tests may open only once"
        )
    if not external_report_only_locked:
        raise ArchAuthorizationError("EXTERNAL_REPORT_ONLY must remain locked")


def terminal_verdict(*, valid: bool, gates: Mapping[str, bool]) -> str:
    if not valid:
        return "INVALID"
    required = ("esr", "device_wins", "asr", "runtime", "listening", "tests", "lint")
    if set(gates) != set(required):
        raise ValueError(f"terminal gates must be exactly {required}")
    return "GO-ARCH" if all(gates[name] for name in required) else "NO-GO-ARCH"
