"""Representability and competence gates for AMP-QUALITY-ARCH-v3."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from fssr_nam.data.arch_v3_fixtures import (
    PRIMARY_SYSTEMS,
    STRESS_SYSTEMS,
    apply_arch_v3_system,
    history_collision_pair,
)

from .amp_quality_arch_v3 import INITIAL_FAMILIES


class ArchV3GateError(RuntimeError):
    """Raised when v3 gate evidence is incomplete or malformed."""


def _finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ArchV3GateError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ArchV3GateError(f"{label} must be finite")
    return converted


def analyze_train_ranges(
    episodes: Mapping[str, tuple[np.ndarray, np.ndarray]],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Measure train-only target range and required residual initialization."""
    data = protocol["synthetic_data"]
    gate = protocol["representability_gate"]
    expected = {*PRIMARY_SYSTEMS, *STRESS_SYSTEMS}
    if not expected.issubset(episodes):
        raise ArchV3GateError("v3 train range evidence is incomplete")
    preroll = int(data["preroll_samples"])
    scored = int(data["scored_samples"])
    quantile = float(gate["required_residual_absolute_quantile"])
    margin = float(gate["residual_initialization_margin"])
    near_peak_ratio = float(gate["near_peak_relative_threshold"])
    rows: list[dict[str, Any]] = []
    for system in (*PRIMARY_SYSTEMS, *STRESS_SYSTEMS):
        inputs, targets = episodes[system]
        if inputs.shape != targets.shape or inputs.ndim != 2:
            raise ArchV3GateError(f"v3 train arrays are malformed: {system}")
        if inputs.shape[-1] != preroll + scored:
            raise ArchV3GateError(f"v3 train duration changed: {system}")
        dry = np.asarray(inputs[:, preroll:], dtype=np.float64).reshape(-1)
        target = np.asarray(targets[:, preroll:], dtype=np.float64).reshape(-1)
        if not np.isfinite(dry).all() or not np.isfinite(target).all():
            raise ArchV3GateError(f"v3 train arrays are non-finite: {system}")
        peak = float(np.max(np.abs(target)))
        if peak <= np.finfo(float).eps:
            raise ArchV3GateError(f"v3 train target is silent: {system}")
        required = target - dry
        required_quantile = float(np.quantile(np.abs(required), quantile))
        rows.append(
            {
                "system": system,
                "tier": "train_only",
                "primary": system in PRIMARY_SYSTEMS,
                "target_rms": float(np.sqrt(np.mean(np.square(target)))),
                "target_absolute_peak": peak,
                "near_peak_fraction": float(
                    np.mean(np.abs(target) >= near_peak_ratio * peak)
                ),
                "required_residual_absolute_quantile": required_quantile,
                "recommended_initial_residual_scale": required_quantile * margin,
            }
        )
    return rows


def analyze_history_collisions(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expose target memory beyond each finite receptive field."""
    probe_samples = int(
        protocol["representability_gate"]["history_collision_probe_samples"]
    )
    rows: list[dict[str, Any]] = []
    cache: dict[int, float] = {}
    for family in INITIAL_FAMILIES:
        config = protocol["architectures"][family]
        receptive_field = int(config["physical_receptive_field_samples"])
        if receptive_field not in cache:
            first, second = history_collision_pair(
                receptive_field_samples=receptive_field,
                probe_samples=probe_samples,
            )
            first_target = apply_arch_v3_system("dynamic_primary", first)
            second_target = apply_arch_v3_system("dynamic_primary", second)
            difference = np.asarray(
                first_target[-probe_samples:] - second_target[-probe_samples:],
                dtype=np.float64,
            )
            cache[receptive_field] = float(np.sqrt(np.mean(np.square(difference))))
        target_collision = cache[receptive_field]
        stateful = bool(config["slow_state"])
        threshold = float(
            protocol["representability_gate"]["history_collision_target_rms_minimum"]
        )
        rows.append(
            {
                "family": family,
                "stateful": stateful,
                "physical_receptive_field_samples": receptive_field,
                "target_collision_rms": target_collision,
                "finite_rf_collision_detected": target_collision >= threshold,
                "memory_capable": stateful or target_collision < threshold,
            }
        )
    return rows


def evaluate_representability_gate(
    range_rows: Sequence[Mapping[str, Any]],
    memory_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail before training when data and architecture domains are incompatible."""
    ranges: dict[str, dict[str, Any]] = {}
    for row in range_rows:
        system = row.get("system")
        if system not in {*PRIMARY_SYSTEMS, *STRESS_SYSTEMS} or system in ranges:
            raise ArchV3GateError("v3 range row has invalid or duplicate system")
        ranges[str(system)] = dict(row)
    if set(ranges) != {*PRIMARY_SYSTEMS, *STRESS_SYSTEMS}:
        raise ArchV3GateError("v3 range rows are incomplete")
    for system, row in ranges.items():
        if row.get("tier") != "train_only":
            raise ArchV3GateError(f"v3 range evidence crossed train tier: {system}")
        for metric in (
            "target_rms",
            "target_absolute_peak",
            "near_peak_fraction",
            "required_residual_absolute_quantile",
            "recommended_initial_residual_scale",
        ):
            row[metric] = _finite(row.get(metric), f"v3 {system} {metric}")

    memories: dict[str, dict[str, Any]] = {}
    for row in memory_rows:
        family = row.get("family")
        if family not in INITIAL_FAMILIES or family in memories:
            raise ArchV3GateError("v3 memory row has invalid or duplicate family")
        normalized = dict(row)
        normalized["target_collision_rms"] = _finite(
            row.get("target_collision_rms"), f"v3 {family} collision"
        )
        if not isinstance(row.get("memory_capable"), bool):
            raise ArchV3GateError(f"v3 {family} memory capability is invalid")
        memories[str(family)] = normalized
    if set(memories) != set(INITIAL_FAMILIES):
        raise ArchV3GateError("v3 memory rows are incomplete")

    gate = protocol["representability_gate"]
    minimum_rms = float(gate["primary_target_rms_minimum"])
    maximum_rms = float(gate["primary_target_rms_maximum"])
    maximum_near_peak = float(gate["near_peak_fraction_maximum"])
    primary_checks = {
        system: {
            "target_rms": minimum_rms <= ranges[system]["target_rms"] <= maximum_rms,
            "not_rail_collapsed": ranges[system]["near_peak_fraction"]
            <= maximum_near_peak,
            "positive_residual_initialization": ranges[system][
                "recommended_initial_residual_scale"
            ]
            > 0.0,
        }
        for system in PRIMARY_SYSTEMS
    }
    training_eligible = [
        family
        for family in INITIAL_FAMILIES
        if all(all(checks.values()) for checks in primary_checks.values())
        and not bool(gate["finite_residual_scale_ceiling_allowed"])
    ]
    memory_capable = [
        family for family in INITIAL_FAMILIES if memories[family]["memory_capable"]
    ]
    checks = {
        "primary_ranges": all(all(item.values()) for item in primary_checks.values()),
        "unbounded_residual_parameterization": not bool(
            gate["finite_residual_scale_ceiling_allowed"]
        )
        and protocol["architectures"]["residual_scale_parameterization"]
        == "softplus_unbounded",
        "training_eligible_family_count": len(training_eligible)
        >= int(gate["minimum_training_eligible_families"]),
        "dynamic_has_memory_capable_family": bool(memory_capable),
        "stress_isolated": bool(
            protocol["synthetic_data"][
                "stress_systems_excluded_from_primary_competence"
            ]
        )
        and not ranges[STRESS_SYSTEMS[0]]["primary"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "primary_checks": primary_checks,
        "training_eligible_families": training_eligible,
        "memory_capable_families": memory_capable,
        "range_rows": [
            ranges[system] for system in (*PRIMARY_SYSTEMS, *STRESS_SYSTEMS)
        ],
        "memory_rows": [memories[family] for family in INITIAL_FAMILIES],
    }


def _round_metrics(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ArchV3GateError(f"{label} INTERNAL_DEV metrics are missing")
    metrics = {
        metric: _finite(value.get(metric), f"{label} {metric}")
        for metric in ("esr", "gain_error", "correlation")
    }
    if metrics["esr"] <= 0.0:
        raise ArchV3GateError(f"{label} ESR must be positive")
    return metrics


def evaluate_round_one_gate(
    trajectories: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    round_lock: Mapping[str, Any],
    family_parameters: Mapping[str, int],
) -> dict[str, Any]:
    """Compare only families whose gain/correlation guards pass first."""
    round_one = protocol["round_1"]
    families = tuple(round_lock["matrix"]["families"])
    systems = tuple(round_lock["matrix"]["systems"])
    seeds = tuple(round_lock["matrix"]["seeds"])
    checkpoints = tuple(round_one["checkpoint_updates"])
    guard_checkpoints = tuple(round_lock["selection"]["guard_checkpoints"])
    expected = {
        (family, system, seed)
        for family in families
        for system in systems
        for seed in seeds
    }
    if set(families) != set(INITIAL_FAMILIES):
        raise ArchV3GateError("round 1 family lock changed")
    if set(systems) != set(PRIMARY_SYSTEMS):
        raise ArchV3GateError("round 1 system lock changed")
    if len(trajectories) != len(expected):
        raise ArchV3GateError("round 1 trajectory count is incomplete")
    if set(family_parameters) != set(families) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in family_parameters.values()
    ):
        raise ArchV3GateError("round 1 family parameter evidence is invalid")

    indexed: dict[tuple[str, str, int], dict[int, dict[str, float]]] = {}
    for row in trajectories:
        family = row.get("family")
        system = row.get("system")
        seed = row.get("seed")
        key = (
            str(family),
            str(system),
            int(seed) if isinstance(seed, int) and not isinstance(seed, bool) else -1,
        )
        if key not in expected:
            raise ArchV3GateError("round 1 family, system, or seed is invalid")
        if key in indexed:
            raise ArchV3GateError(f"duplicate round 1 trajectory: {key}")
        rows = row.get("checkpoints")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ArchV3GateError(f"round 1 checkpoints are missing: {key}")
        checkpoint_index: dict[int, dict[str, float]] = {}
        for checkpoint in rows:
            if not isinstance(checkpoint, Mapping):
                raise ArchV3GateError(f"malformed round 1 checkpoint: {key}")
            update = checkpoint.get("update")
            if update not in checkpoints or update in checkpoint_index:
                raise ArchV3GateError(f"unexpected round 1 checkpoint: {key}/{update}")
            checkpoint_index[int(update)] = _round_metrics(
                checkpoint.get("internal_dev"), f"round 1 {key}/{update}"
            )
        if set(checkpoint_index) != set(checkpoints):
            raise ArchV3GateError(f"incomplete round 1 checkpoints: {key}")
        indexed[key] = checkpoint_index
    if set(indexed) != expected:
        raise ArchV3GateError("round 1 trajectory matrix is incomplete")

    selection = round_lock["selection"]
    gain_threshold = float(selection["gain_error_strictly_greater_than"])
    correlation_threshold = float(selection["correlation_strictly_greater_than"])
    final_update = checkpoints[-1]
    family_results: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    for family in families:
        guard_rows = [
            indexed[(family, system, seed)][update]
            for system in systems
            for seed in seeds
            for update in guard_checkpoints
        ]
        guards_pass = all(
            row["gain_error"] > gain_threshold
            and row["correlation"] > correlation_threshold
            for row in guard_rows
        )
        dynamic_eligible = bool(
            protocol["architectures"][family]["promotion_eligible_for_dynamic"]
        )
        final_esr = {
            (system, seed): indexed[(family, system, seed)][final_update]["esr"]
            for system in systems
            for seed in seeds
        }
        family_results[family] = {
            "guards_pass": guards_pass,
            "dynamic_promotion_eligible": dynamic_eligible,
            "eligible_for_comparison": guards_pass and dynamic_eligible,
            "minimum_guard_gain_error": min(row["gain_error"] for row in guard_rows),
            "minimum_guard_correlation": min(row["correlation"] for row in guard_rows),
            "final_median_esr": float(np.median(list(final_esr.values()))),
            "final_per_system_median_esr": {
                system: float(np.median([final_esr[(system, seed)] for seed in seeds]))
                for system in systems
            },
            "parameters": int(family_parameters[family]),
        }
        if guards_pass and dynamic_eligible:
            eligible.append(family)

    selected: str | None = None
    tie_candidates: list[str] = []
    if eligible:
        condition_best = {
            (system, seed): min(
                indexed[(family, system, seed)][final_update]["esr"]
                for family in eligible
            )
            for system in systems
            for seed in seeds
        }
        for family in eligible:
            normalized = [
                indexed[(family, system, seed)][final_update]["esr"]
                / condition_best[(system, seed)]
                for system in systems
                for seed in seeds
            ]
            family_results[family]["paired_condition_normalized_esr"] = normalized
            family_results[family]["selection_score"] = float(np.median(normalized))
        best_score = min(
            float(family_results[family]["selection_score"]) for family in eligible
        )
        tolerance = float(selection["tie_relative_tolerance"])
        tie_candidates = [
            family
            for family in eligible
            if float(family_results[family]["selection_score"])
            <= best_score * (1.0 + tolerance)
        ]
        selected = min(
            tie_candidates,
            key=lambda family: (family_parameters[family], family),
        )

    return {
        "passed": selected is not None,
        "comparison_performed_after_guards": bool(eligible),
        "eligible_families": eligible,
        "tie_candidates": tie_candidates,
        "selected_family": selected,
        "selection_update": final_update,
        "family_results": family_results,
        "trajectory_count": len(trajectories),
        "previous_round_relative_improvement": None,
        "counts_toward_consecutive_no_improvement_stop": False,
    }
