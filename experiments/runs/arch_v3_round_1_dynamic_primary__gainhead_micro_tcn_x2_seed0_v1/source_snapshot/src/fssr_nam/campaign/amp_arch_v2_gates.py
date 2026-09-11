"""Fail-closed competence and comparison gates for architecture v2."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

import numpy as np

from .amp_competence_arch_v2 import (
    ALL_FAMILIES,
    CANDIDATE_FAMILIES,
    CHECKPOINTS,
    CONTROL_FAMILY,
    SEEDS,
    SYSTEMS,
)


class ArchV2GateError(RuntimeError):
    """Raised when v2 scientific evidence is incomplete or malformed."""


def _finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ArchV2GateError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ArchV2GateError(f"{label} must be finite")
    return converted


def _validation(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ArchV2GateError(f"{label} validation is missing")
    return {
        metric: _finite(value.get(metric), f"{label} {metric}")
        for metric in (
            "esr",
            "gain_error",
            "correlation",
            "prediction_rms",
            "target_rms",
        )
    }


def _guard_passes(
    metrics: Mapping[str, float], protocol: Mapping[str, Any], stage: str
) -> bool:
    config = protocol[stage]
    return metrics["gain_error"] > float(
        config["gain_error_strictly_greater_than"]
    ) and metrics["correlation"] > float(config["correlation_strictly_greater_than"])


def evaluate_competence_gate(
    trajectories: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Select the first independently confirmed convergence budget."""
    expected_pairs = {(system, seed) for system in SYSTEMS for seed in SEEDS}
    if len(trajectories) != len(expected_pairs):
        raise ArchV2GateError("competence trajectory count is incomplete")
    indexed: dict[tuple[str, int], dict[int, dict[str, float]]] = {}
    for row in trajectories:
        system = row.get("system")
        seed = row.get("seed")
        if system not in SYSTEMS or seed not in SEEDS:
            raise ArchV2GateError("competence system or seed is invalid")
        key = (str(system), int(seed))
        if key in indexed:
            raise ArchV2GateError(f"duplicate competence trajectory: {key}")
        checkpoints = row.get("checkpoints")
        if not isinstance(checkpoints, Sequence) or isinstance(
            checkpoints, (str, bytes)
        ):
            raise ArchV2GateError(f"competence checkpoints are missing: {key}")
        checkpoint_index: dict[int, dict[str, float]] = {}
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, Mapping):
                raise ArchV2GateError(f"malformed competence checkpoint: {key}")
            update = checkpoint.get("update")
            if update not in CHECKPOINTS:
                raise ArchV2GateError(f"unexpected competence checkpoint: {key}")
            if update in checkpoint_index:
                raise ArchV2GateError(
                    f"duplicate competence checkpoint: {key}/{update}"
                )
            checkpoint_index[int(update)] = _validation(
                checkpoint.get("validation"), f"competence {key}/{update}"
            )
        if set(checkpoint_index) != set(CHECKPOINTS):
            raise ArchV2GateError(f"incomplete competence checkpoints: {key}")
        indexed[key] = checkpoint_index
    if set(indexed) != expected_pairs:
        raise ArchV2GateError("competence system/seed matrix is incomplete")

    checkpoint_summary: list[dict[str, Any]] = []
    for update in CHECKPOINTS:
        metrics = [indexed[key][update] for key in sorted(indexed)]
        checkpoint_summary.append(
            {
                "update": update,
                "all_guards_pass": all(
                    _guard_passes(item, protocol, "competence") for item in metrics
                ),
                "median_esr": float(np.median([item["esr"] for item in metrics])),
                "minimum_gain_error": min(item["gain_error"] for item in metrics),
                "minimum_correlation": min(item["correlation"] for item in metrics),
            }
        )

    selected_budget: int | None = None
    confirming_checkpoint: int | None = None
    selected_plateau: float | None = None
    minimum_plateau = float(
        protocol["competence"]["plateau_median_relative_esr_improvement_minimum"]
    )
    maximum_plateau = float(
        protocol["competence"]["plateau_median_relative_esr_improvement_maximum"]
    )
    for current, following in pairwise(checkpoint_summary):
        current_esr = float(current["median_esr"])
        if current_esr <= 0.0:
            raise ArchV2GateError("competence median ESR must be positive")
        plateau = (current_esr - float(following["median_esr"])) / current_esr
        current["next_checkpoint_update"] = following["update"]
        current["next_checkpoint_relative_esr_improvement"] = plateau
        current["selection_pair_passes"] = (
            bool(current["all_guards_pass"])
            and bool(following["all_guards_pass"])
            and minimum_plateau <= plateau <= maximum_plateau
        )
        if selected_budget is None and current["selection_pair_passes"]:
            selected_budget = int(current["update"])
            confirming_checkpoint = int(following["update"])
            selected_plateau = plateau

    return {
        "passed": selected_budget is not None,
        "selected_budget_updates": selected_budget,
        "confirming_checkpoint_updates": confirming_checkpoint,
        "selected_plateau_relative_esr_improvement": selected_plateau,
        "checkpoint_summary": checkpoint_summary,
        "trajectory_count": len(trajectories),
    }


def _hierarchical_bootstrap_lower_bound(
    improvements: Mapping[tuple[str, int], float], *, replicates: int, seed: int
) -> float:
    generator = np.random.default_rng(seed)
    distribution = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled: list[float] = []
        for system_index in generator.integers(0, len(SYSTEMS), size=len(SYSTEMS)):
            system = SYSTEMS[int(system_index)]
            for seed_index in generator.integers(0, len(SEEDS), size=len(SEEDS)):
                sampled.append(improvements[(system, SEEDS[int(seed_index)])])
        distribution[replicate] = np.median(sampled)
    return float(np.quantile(distribution, 0.025, method="linear"))


def evaluate_comparison_gate(
    trajectories: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate paired candidates only after a competent fresh control."""
    expected = {
        (family, system, seed)
        for family in ALL_FAMILIES
        for system in SYSTEMS
        for seed in SEEDS
    }
    if len(trajectories) != len(expected):
        raise ArchV2GateError("comparison trajectory count is incomplete")
    indexed: dict[tuple[str, str, int], dict[str, float]] = {}
    for row in trajectories:
        family = row.get("family")
        system = row.get("system")
        seed = row.get("seed")
        key = (str(family), str(system), int(seed) if isinstance(seed, int) else -1)
        if key not in expected:
            raise ArchV2GateError("comparison family, system, or seed is invalid")
        if key in indexed:
            raise ArchV2GateError(f"duplicate comparison trajectory: {key}")
        indexed[key] = _validation(row.get("validation"), f"comparison {key}")
    if set(indexed) != expected:
        raise ArchV2GateError("comparison matrix is incomplete")

    control_guards_pass = all(
        _guard_passes(indexed[(CONTROL_FAMILY, system, seed)], protocol, "comparison")
        for system in SYSTEMS
        for seed in SEEDS
    )
    candidate_results: dict[str, Any] = {}
    eligible: list[str] = []
    for family in CANDIDATE_FAMILIES:
        improvements: dict[tuple[str, int], float] = {}
        candidate_guards_pass = True
        for system in SYSTEMS:
            for seed in SEEDS:
                baseline = indexed[(CONTROL_FAMILY, system, seed)]
                candidate = indexed[(family, system, seed)]
                if baseline["esr"] <= 0.0:
                    raise ArchV2GateError("comparison control ESR must be positive")
                improvements[(system, seed)] = (
                    baseline["esr"] - candidate["esr"]
                ) / baseline["esr"]
                candidate_guards_pass &= _guard_passes(
                    candidate, protocol, "comparison"
                )
        observed = float(np.median(list(improvements.values())))
        per_system_regression = {
            system: -float(np.median([improvements[(system, seed)] for seed in SEEDS]))
            for system in SYSTEMS
        }
        lower_bound = _hierarchical_bootstrap_lower_bound(
            improvements,
            replicates=int(protocol["comparison"]["bootstrap_replicates"]),
            seed=int(protocol["comparison"]["bootstrap_seed"]),
        )
        checks = {
            "fresh_control_guards": control_guards_pass,
            "candidate_guards": candidate_guards_pass,
            "median_improvement": observed
            >= float(
                protocol["comparison"]["paired_median_relative_esr_improvement_minimum"]
            ),
            "bootstrap_lower_bound": lower_bound
            > float(
                protocol["comparison"][
                    "lower_95_confidence_bound_strictly_greater_than"
                ]
            ),
            "per_system_regression": max(per_system_regression.values())
            <= float(
                protocol["comparison"][
                    "per_system_median_relative_esr_regression_maximum"
                ]
            ),
        }
        passed = all(checks.values())
        if passed:
            eligible.append(family)
        candidate_results[family] = {
            "passed": passed,
            "checks": checks,
            "paired_median_relative_esr_improvement": observed,
            "lower_95_confidence_bound": lower_bound,
            "per_system_median_relative_esr_regression": per_system_regression,
        }
    promoted = (
        sorted(
            eligible,
            key=lambda family: (
                -candidate_results[family]["paired_median_relative_esr_improvement"],
                family,
            ),
        )[0]
        if eligible
        else None
    )
    return {
        "passed": promoted is not None,
        "control_guards_pass": control_guards_pass,
        "eligible_candidates": eligible,
        "promoted_candidate": promoted,
        "candidate_results": candidate_results,
        "trajectory_count": len(trajectories),
    }
