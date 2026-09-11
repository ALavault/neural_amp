"""Fail-closed gates for AMP-QUALITY-ARCH-v1 evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .amp_quality_arch_v1 import TRAINING_CANDIDATES


class ArchGateError(RuntimeError):
    """Raised when architecture evidence is incomplete or malformed."""


def _finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ArchGateError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ArchGateError(f"{label} must be finite")
    return converted


def _relative_improvement(baseline: float, candidate: float) -> float:
    if baseline <= 0.0:
        raise ArchGateError("mechanism baseline ESR must be positive")
    return (baseline - candidate) / baseline


def _relative_regression(baseline: float, candidate: float) -> float:
    if baseline <= 0.0:
        raise ArchGateError("mechanism baseline ESR must be positive")
    return (candidate - baseline) / baseline


def evaluate_arch_mechanism_gate(
    trajectories: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Prune failed ideas while preserving every valid surviving candidate."""
    if len(trajectories) != protocol["mechanism_training"]["trajectory_count"]:
        raise ArchGateError("mechanism trajectory count is incomplete")
    indexed: dict[tuple[str, str, float], Mapping[str, Any]] = {}
    for trajectory in trajectories:
        family = trajectory.get("family")
        system = trajectory.get("system")
        auxiliary_weight = _finite(
            trajectory.get("auxiliary_weight"), "mechanism auxiliary weight"
        )
        key = (str(family), str(system), auxiliary_weight)
        if key in indexed:
            raise ArchGateError(f"duplicate mechanism trajectory: {key}")
        validation = trajectory.get("validation")
        if not isinstance(validation, Mapping):
            raise ArchGateError(f"missing mechanism validation: {key}")
        for metric in ("esr", "gain_error", "correlation"):
            _finite(validation.get(metric), f"{key} {metric}")
        indexed[key] = trajectory

    auxiliary_weights = tuple(
        float(value)
        for value in protocol["mechanism_training"][
            "phys_s6_auxiliary_weight_candidates"
        ]
    )
    required = {
        ("micro_tcn_x2", "static_composite", 0.0),
        ("micro_tcn_x2", "dynamic_composite", 0.0),
        ("phys_det_tcn_x2", "static_composite", 0.0),
        ("phys_det_tcn_x2", "dynamic_composite", 0.0),
        ("rf2047_tfilm_x2", "two_clippers", 0.0),
        ("cascade_rf2047_tfilm_x2", "two_clippers", 0.0),
        *(
            ("phys_s6_tcn_x2", "dynamic_composite", weight)
            for weight in auxiliary_weights
        ),
    }
    dynamic_phys = [
        indexed[("phys_s6_tcn_x2", "dynamic_composite", weight)]
        for weight in auxiliary_weights
        if ("phys_s6_tcn_x2", "dynamic_composite", weight) in indexed
    ]
    if len(dynamic_phys) != len(auxiliary_weights):
        raise ArchGateError("physical S6 auxiliary-weight matrix is incomplete")
    selected_auxiliary = min(
        (
            _finite(item["validation"]["esr"], "physical S6 dynamic ESR"),
            float(item["auxiliary_weight"]),
        )
        for item in dynamic_phys
    )[1]
    required.add(("phys_s6_tcn_x2", "static_composite", selected_auxiliary))
    if set(indexed) != required:
        missing = sorted(required - set(indexed))
        extra = sorted(set(indexed) - required)
        raise ArchGateError(
            f"mechanism trajectory matrix drifted; missing={missing}, extra={extra}"
        )

    gain_minimum = float(protocol["mechanism_gate"]["anti_collapse_gain_error_minimum"])
    correlation_minimum = float(
        protocol["mechanism_gate"]["anti_collapse_correlation_minimum"]
    )

    def values(family: str, system: str, weight: float = 0.0) -> dict[str, float]:
        validation = indexed[(family, system, weight)]["validation"]
        return {
            metric: _finite(validation[metric], f"{family}/{system}/{metric}")
            for metric in ("esr", "gain_error", "correlation")
        }

    def guard(metrics: Mapping[str, float]) -> bool:
        return (
            metrics["gain_error"] > gain_minimum
            and metrics["correlation"] > correlation_minimum
        )

    micro_dynamic = values("micro_tcn_x2", "dynamic_composite")
    micro_static = values("micro_tcn_x2", "static_composite")
    det_dynamic = values("phys_det_tcn_x2", "dynamic_composite")
    det_static = values("phys_det_tcn_x2", "static_composite")
    phys_dynamic = values("phys_s6_tcn_x2", "dynamic_composite", selected_auxiliary)
    phys_static = values("phys_s6_tcn_x2", "static_composite", selected_auxiliary)
    rf_two = values("rf2047_tfilm_x2", "two_clippers")
    cascade_two = values("cascade_rf2047_tfilm_x2", "two_clippers")

    dynamic_minimum = float(
        protocol["mechanism_gate"]["physical_bus_dynamic_improvement_minimum"]
    )
    static_maximum = float(
        protocol["mechanism_gate"]["physical_bus_static_regression_maximum"]
    )
    cascade_minimum = float(
        protocol["mechanism_gate"]["cascade_two_clipper_improvement_minimum"]
    )
    improvements = {
        "phys_det_dynamic": _relative_improvement(
            micro_dynamic["esr"], det_dynamic["esr"]
        ),
        "phys_s6_dynamic": _relative_improvement(
            micro_dynamic["esr"], phys_dynamic["esr"]
        ),
        "cascade_two_clippers": _relative_improvement(
            rf_two["esr"], cascade_two["esr"]
        ),
    }
    regressions = {
        "phys_det_static": _relative_regression(micro_static["esr"], det_static["esr"]),
        "phys_s6_static": _relative_regression(micro_static["esr"], phys_static["esr"]),
    }
    checks = {
        "micro_tcn_x2": guard(micro_dynamic) and guard(micro_static),
        "phys_det_tcn_x2": (
            guard(det_dynamic)
            and guard(det_static)
            and improvements["phys_det_dynamic"] >= dynamic_minimum
            and regressions["phys_det_static"] <= static_maximum
        ),
        "phys_s6_tcn_x2": (
            guard(phys_dynamic)
            and guard(phys_static)
            and improvements["phys_s6_dynamic"] >= dynamic_minimum
            and regressions["phys_s6_static"] <= static_maximum
        ),
        "rf2047_tfilm_x2": guard(rf_two),
        "cascade_rf2047_tfilm_x2": (
            guard(cascade_two)
            and improvements["cascade_two_clippers"] >= cascade_minimum
        ),
    }
    eligible = [family for family in TRAINING_CANDIDATES if checks[family]]
    rejected = [family for family in TRAINING_CANDIDATES if not checks[family]]
    minimum = int(
        protocol["mechanism_gate"]["minimum_training_eligible_candidates_after_gate"]
    )
    return {
        "format": "fssr-amp-arch-mechanism-gate-v1",
        "valid": True,
        "passed": len(eligible) >= minimum,
        "selected_auxiliary_weight": selected_auxiliary,
        "eligible_candidates": eligible,
        "rejected_candidates": rejected,
        "candidate_checks": checks,
        "relative_improvements": improvements,
        "relative_regressions": regressions,
        "thresholds": {
            "dynamic_improvement_minimum": dynamic_minimum,
            "static_regression_maximum": static_maximum,
            "cascade_improvement_minimum": cascade_minimum,
            "gain_error_minimum": gain_minimum,
            "correlation_minimum": correlation_minimum,
            "minimum_candidates": minimum,
        },
    }
