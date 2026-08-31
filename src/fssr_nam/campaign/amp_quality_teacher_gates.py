"""Fail-closed decision gates for AMP-QUALITY-TEACHER-v1."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from fssr_nam.statistics.quality_teacher import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIRMATION_DEVICES,
    CONFIRMATION_SEEDS,
    SECONDARY_METRICS,
    QualityTeacherStatisticsError,
    hierarchical_confirmation_bootstrap,
    l1_plus_mrstft,
)

CANDIDATE_FAMILY = "s4_tfilm_wavenet_x2_teacher"
CONTROL_FAMILY = "wavenet_x2_teacher_fast_only"
DEVELOPMENT_DEVICES = ("fulltone", "bigmuff", "ampeg")
COMPARATOR_FAMILIES = (
    "nablafx_s4_tfilm_large",
    "nam_a2_full",
    "wavenet_dense_16x18",
)


class QualityTeacherGateError(RuntimeError):
    """Protocol or decision evidence is malformed or incomplete."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualityTeacherGateError(f"{label} must be a mapping")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise QualityTeacherGateError(f"{label} must be a sequence")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualityTeacherGateError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise QualityTeacherGateError(f"{label} must be finite")
    return result


def _nonnegative(value: object, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise QualityTeacherGateError(f"{label} must be nonnegative")
    return result


def _positive(value: object, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise QualityTeacherGateError(f"{label} must be positive")
    return result


def _require_equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise QualityTeacherGateError(f"{label} must equal {expected!r}, got {value!r}")


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    slow = _mapping(protocol.get("slow_value_gate"), "slow_value_gate")
    for name, expected in {
        "device": "ampeg",
        "seed": 0,
        "candidate": CANDIDATE_FAMILY,
        "control": CONTROL_FAMILY,
        "esr_median_relative_improvement_minimum": 0.05,
        "l1_plus_mrstft_improvement_strictly_greater_than": 0.0,
        "secondary_metrics": list(SECONDARY_METRICS),
        "secondary_relative_regression_maximum": 0.05,
        "failure_promotes": CONTROL_FAMILY,
        "success_promotes": CANDIDATE_FAMILY,
        "retry_allowed": False,
    }.items():
        _require_equal(slow.get(name), expected, f"slow_value_gate.{name}")

    comparators = _mapping(protocol.get("comparators"), "comparators")
    _require_equal(
        comparators.get("families"), list(COMPARATOR_FAMILIES), "comparators.families"
    )
    selection = _mapping(
        comparators.get("global_selection"), "comparators.global_selection"
    )
    for name, expected in {
        "devices": list(DEVELOPMENT_DEVICES),
        "seed": 0,
        "primary": "equal_device_weighted_median_esr",
        "relative_tie_threshold_strictly_less_than": 0.01,
        "tie_breaker": "equal_device_weighted_median_l1_plus_mrstft",
        "final_deterministic_order": list(COMPARATOR_FAMILIES),
        "per_device_comparator_selection_allowed": False,
    }.items():
        _require_equal(selection.get(name), expected, f"global_selection.{name}")

    confirmation = _mapping(protocol.get("confirmation"), "confirmation")
    _require_equal(
        confirmation.get("devices"),
        list(CONFIRMATION_DEVICES),
        "confirmation.devices",
    )
    _require_equal(
        confirmation.get("seeds"), list(CONFIRMATION_SEEDS), "confirmation.seeds"
    )
    statistics = _mapping(protocol.get("statistics"), "statistics")
    for name, expected in {
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "interval": "paired_percentile_2.5_97.5",
        "resampling_order_within_device": ["files", "seeds"],
        "devices_equal_weight": True,
        "observation_unit": "device_file_seed",
    }.items():
        _require_equal(statistics.get(name), expected, f"statistics.{name}")

    verdict = _mapping(protocol.get("objective_verdict"), "objective_verdict")
    for name, expected in {
        "go": "GO-OBJECTIVE-SOTA",
        "no_go": "NO-GO-OBJECTIVE-SOTA",
        "invalid": "INVALID",
        "aggregate_esr_relative_improvement_minimum": 0.10,
        "aggregate_esr_bootstrap_lower_95_strictly_greater_than": 0.0,
        "per_confirmation_device_esr_median_improvement_strictly_greater_than": 0.0,
        "aggregate_l1_plus_mrstft_improvement_strictly_greater_than": 0.0,
        "aggregate_l1_plus_mrstft_bootstrap_lower_95_strictly_greater_than": 0.0,
        "secondary_metrics": list(SECONDARY_METRICS),
        "secondary_relative_regression_maximum": 0.05,
        "correlation_strictly_greater_than": 0.9,
        "gain_error_strictly_greater_than": -0.2,
        "leakage_missing_provenance_or_incomplete_execution": "INVALID",
    }.items():
        _require_equal(verdict.get(name), expected, f"objective_verdict.{name}")


def _file_id(row: Mapping[str, Any], *, required: bool) -> str:
    values = [
        row[name]
        for name in ("file", "file_id", "source")
        if name in row and row[name] is not None
    ]
    if not values and not required:
        return "__aggregate__"
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise QualityTeacherGateError("file identifier must be non-empty")
    if any(value != values[0] for value in values[1:]):
        raise QualityTeacherGateError("file identifier aliases disagree")
    return str(values[0])


def _row_metrics(row: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    if "metrics" not in row:
        return row
    return _mapping(row.get("metrics"), f"{label} metrics")


def _combined_loss(metrics: Mapping[str, Any], label: str) -> float:
    try:
        return l1_plus_mrstft(metrics, label)
    except QualityTeacherStatisticsError as error:
        raise QualityTeacherGateError(str(error)) from error


def _at_least(value: float, threshold: float) -> bool:
    return value > threshold or math.isclose(
        value, threshold, rel_tol=1.0e-12, abs_tol=1.0e-12
    )


def _at_most(value: float, threshold: float) -> bool:
    return value < threshold or math.isclose(
        value, threshold, rel_tol=1.0e-12, abs_tol=1.0e-12
    )


def _slow_rows(
    rows: Sequence[Mapping[str, Any]], family: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for item in rows:
        row = _mapping(item, "slow-value row")
        if row.get("device") != "ampeg" or row.get("seed") != 0:
            raise QualityTeacherGateError("slow-value rows require Ampeg seed 0")
        if row.get("family") != family:
            raise QualityTeacherGateError(f"slow-value family must equal {family!r}")
        file_id = _file_id(row, required=True)
        if file_id in indexed:
            raise QualityTeacherGateError("duplicate slow-value file observation")
        indexed[file_id] = _row_metrics(row, "slow-value")
    if not indexed:
        raise QualityTeacherGateError("slow-value evidence requires paired files")
    return indexed


def evaluate_slow_value_gate(
    candidate_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Promote the S4 path only when its paired Ampeg metrics add value."""
    _validate_protocol(protocol)
    candidate = _slow_rows(candidate_rows, CANDIDATE_FAMILY)
    control = _slow_rows(control_rows, CONTROL_FAMILY)
    if set(candidate) != set(control):
        raise QualityTeacherGateError(
            "slow-value candidate and control files are not paired"
        )
    diagnostics: dict[str, dict[str, Any]] = {}
    esr_improvements = []
    loss_improvements = []
    regressions: dict[str, list[float]] = {metric: [] for metric in SECONDARY_METRICS}
    for file_id in sorted(candidate):
        candidate_metrics = candidate[file_id]
        control_metrics = control[file_id]
        control_esr = _positive(control_metrics.get("esr"), "control ESR")
        candidate_esr = _nonnegative(candidate_metrics.get("esr"), "candidate ESR")
        control_loss = _combined_loss(control_metrics, "control")
        if control_loss <= 0.0:
            raise QualityTeacherGateError("control l1_plus_mrstft must be positive")
        candidate_loss = _combined_loss(candidate_metrics, "candidate")
        esr_improvement = (control_esr - candidate_esr) / control_esr
        loss_improvement = (control_loss - candidate_loss) / control_loss
        file_regressions = {}
        for metric in SECONDARY_METRICS:
            control_value = _positive(control_metrics.get(metric), f"control {metric}")
            candidate_value = _nonnegative(
                candidate_metrics.get(metric), f"candidate {metric}"
            )
            regression = (candidate_value - control_value) / control_value
            regressions[metric].append(regression)
            file_regressions[metric] = regression
        esr_improvements.append(esr_improvement)
        loss_improvements.append(loss_improvement)
        diagnostics[file_id] = {
            "esr_relative_improvement": esr_improvement,
            "l1_plus_mrstft_relative_improvement": loss_improvement,
            "secondary_relative_regressions": file_regressions,
        }

    median_esr = float(np.median(esr_improvements))
    median_loss = float(np.median(loss_improvements))
    median_regressions = {
        metric: float(np.median(values)) for metric, values in regressions.items()
    }
    config = _mapping(protocol["slow_value_gate"], "slow_value_gate")
    checks = {
        "esr_median_relative_improvement": _at_least(
            median_esr,
            float(config["esr_median_relative_improvement_minimum"]),
        ),
        "l1_plus_mrstft_improvement": median_loss
        > float(config["l1_plus_mrstft_improvement_strictly_greater_than"]),
        **{
            f"{metric}_nonregression": _at_most(
                median_regressions[metric],
                float(config["secondary_relative_regression_maximum"]),
            )
            for metric in SECONDARY_METRICS
        },
    }
    passed = all(checks.values())
    selected = str(config["success_promotes"] if passed else config["failure_promotes"])
    device_diagnostics = {
        "file_count": len(candidate),
        "esr_median_relative_improvement": median_esr,
        "l1_plus_mrstft_median_relative_improvement": median_loss,
        "secondary_median_relative_regressions": median_regressions,
    }
    return {
        "format": "fssr-amp-quality-teacher-slow-value-gate-v1",
        "passed": passed,
        "selected_candidate": selected,
        "promoted_family": selected,
        "checks": checks,
        "per_device": {"ampeg": device_diagnostics},
        "per_file": diagnostics,
        "retry_allowed": False,
    }


def _comparator_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    indexed: dict[str, dict[str, dict[str, tuple[float, float]]]] = {
        family: defaultdict(dict) for family in COMPARATOR_FAMILIES
    }
    for item in rows:
        row = _mapping(item, "comparator row")
        family = row.get("family")
        device = row.get("device")
        if family not in COMPARATOR_FAMILIES:
            raise QualityTeacherGateError(f"unknown comparator family: {family!r}")
        if device not in DEVELOPMENT_DEVICES:
            raise QualityTeacherGateError(f"unknown development device: {device!r}")
        if row.get("seed") != 0:
            raise QualityTeacherGateError("comparator selection requires seed 0")
        file_id = _file_id(row, required=False)
        family_rows = indexed[str(family)][str(device)]
        if file_id in family_rows:
            raise QualityTeacherGateError(
                "duplicate comparator family/device/file observation"
            )
        metrics = _row_metrics(row, "comparator")
        family_rows[file_id] = (
            _positive(metrics.get("esr"), "comparator ESR"),
            _combined_loss(metrics, "comparator"),
        )

    expected_devices = set(DEVELOPMENT_DEVICES)
    for family in COMPARATOR_FAMILIES:
        if set(indexed[family]) != expected_devices:
            raise QualityTeacherGateError(
                f"{family} comparator device matrix is incomplete"
            )
    for device in DEVELOPMENT_DEVICES:
        file_sets = [set(indexed[family][device]) for family in COMPARATOR_FAMILIES]
        if not file_sets[0] or any(files != file_sets[0] for files in file_sets[1:]):
            raise QualityTeacherGateError(
                f"{device} comparator files are not paired across families"
            )
    return indexed


def select_global_comparator(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Select one global comparator from seed-0 development validation metrics."""
    _validate_protocol(protocol)
    indexed = _comparator_rows(rows)
    results: dict[str, dict[str, Any]] = {}
    for family in COMPARATOR_FAMILIES:
        per_device = {}
        for device in DEVELOPMENT_DEVICES:
            values = list(indexed[family][device].values())
            per_device[device] = {
                "file_count": len(values),
                "median_esr": float(np.median([value[0] for value in values])),
                "median_l1_plus_mrstft": float(
                    np.median([value[1] for value in values])
                ),
            }
        results[family] = {
            "per_device": per_device,
            "equal_device_weighted_median_esr": float(
                np.mean(
                    [per_device[device]["median_esr"] for device in DEVELOPMENT_DEVICES]
                )
            ),
            "equal_device_weighted_median_l1_plus_mrstft": float(
                np.mean(
                    [
                        per_device[device]["median_l1_plus_mrstft"]
                        for device in DEVELOPMENT_DEVICES
                    ]
                )
            ),
        }

    primary_scores = {
        family: details["equal_device_weighted_median_esr"]
        for family, details in results.items()
    }
    best_score = min(primary_scores.values())
    selection = _mapping(
        _mapping(protocol["comparators"], "comparators")["global_selection"],
        "global_selection",
    )
    tie_threshold = float(selection["relative_tie_threshold_strictly_less_than"])
    relative_gaps = {
        family: (score - best_score) / best_score
        for family, score in primary_scores.items()
    }
    tie_candidates = [
        family
        for family in COMPARATOR_FAMILIES
        if relative_gaps[family] < tie_threshold
    ]
    order = {
        family: index
        for index, family in enumerate(selection["final_deterministic_order"])
    }
    selected = min(
        tie_candidates,
        key=lambda family: (
            results[family]["equal_device_weighted_median_l1_plus_mrstft"],
            order[family],
        ),
    )
    for family in COMPARATOR_FAMILIES:
        results[family]["primary_relative_gap_from_best"] = relative_gaps[family]
        results[family]["inside_strict_one_percent_tie"] = family in tie_candidates
    return {
        "format": "fssr-amp-quality-teacher-global-comparator-gate-v1",
        "passed": True,
        "selected_comparator": selected,
        "global_comparator": selected,
        "tie_break_applied": len(tie_candidates) > 1,
        "tie_candidates": tie_candidates,
        "families": results,
        "per_device_comparator_selection": False,
    }


def _integrity_value(
    evidence: Mapping[str, Any], names: tuple[str, ...]
) -> tuple[object, str | None]:
    integrity = evidence.get("integrity")
    containers = [evidence]
    if integrity is not None:
        if not isinstance(integrity, Mapping):
            return None, "integrity must be a mapping"
        containers.insert(0, integrity)
    found = [
        container[name]
        for container in containers
        for name in names
        if name in container
    ]
    if not found:
        return None, f"missing integrity field {names[0]}"
    if any(value != found[0] for value in found[1:]):
        return None, f"conflicting integrity field {names[0]}"
    if not isinstance(found[0], bool):
        return None, f"integrity field {names[0]} must be boolean"
    return found[0], None


def _invalid_result(reasons: list[str]) -> dict[str, Any]:
    return {
        "format": "fssr-amp-quality-teacher-objective-verdict-v1",
        "valid": False,
        "passed": False,
        "verdict": "INVALID",
        "invalid_reasons": reasons,
        "checks": {},
        "per_device": {},
        "statistics": None,
    }


def evaluate_objective_verdict(
    evidence: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Return GO, NO-GO, or INVALID from frozen public confirmation evidence."""
    _validate_protocol(protocol)
    evidence = _mapping(evidence, "objective evidence")
    reasons = []
    leakage, error = _integrity_value(
        evidence, ("leakage_detected", "test_leakage_detected")
    )
    if error is not None:
        reasons.append(error)
    elif leakage is True:
        reasons.append("test leakage detected")
    provenance, error = _integrity_value(evidence, ("provenance_complete",))
    if error is not None:
        reasons.append(error)
    elif provenance is False:
        reasons.append("provenance incomplete")
    execution, error = _integrity_value(evidence, ("execution_complete",))
    if error is not None:
        reasons.append(error)
    elif execution is False:
        reasons.append("execution incomplete")
    if reasons:
        return _invalid_result(reasons)

    try:
        rows = _sequence(evidence.get("rows"), "objective rows")
        statistics_config = _mapping(protocol["statistics"], "statistics")
        summary = hierarchical_confirmation_bootstrap(
            rows,
            replicates=int(statistics_config["bootstrap_replicates"]),
            seed=int(statistics_config["bootstrap_seed"]),
        )
    except (QualityTeacherGateError, QualityTeacherStatisticsError) as error:
        return _invalid_result([str(error)])

    config = _mapping(protocol["objective_verdict"], "objective_verdict")
    intervals = _mapping(summary["confidence_intervals_95"], "confidence intervals")
    esr_interval = _mapping(intervals["esr_relative_improvement"], "ESR interval")
    loss_interval = _mapping(
        intervals["l1_plus_mrstft_relative_improvement"], "loss interval"
    )
    maximum_regressions = _mapping(
        summary["maximum_per_device_secondary_relative_regressions"],
        "secondary regressions",
    )
    checks = {
        "aggregate_esr_relative_improvement": _at_least(
            float(summary["aggregate_esr_relative_improvement"]),
            float(config["aggregate_esr_relative_improvement_minimum"]),
        ),
        "aggregate_esr_bootstrap_lower_95": float(esr_interval["lower_95"])
        > float(config["aggregate_esr_bootstrap_lower_95_strictly_greater_than"]),
        "per_confirmation_device_esr": all(
            float(summary["per_device"][device]["esr_relative_improvement"])
            > float(
                config[
                    "per_confirmation_device_esr_median_improvement_strictly_greater_than"
                ]
            )
            for device in CONFIRMATION_DEVICES
        ),
        "aggregate_l1_plus_mrstft_improvement": float(
            summary["aggregate_l1_plus_mrstft_relative_improvement"]
        )
        > float(config["aggregate_l1_plus_mrstft_improvement_strictly_greater_than"]),
        "aggregate_l1_plus_mrstft_bootstrap_lower_95": float(loss_interval["lower_95"])
        > float(
            config["aggregate_l1_plus_mrstft_bootstrap_lower_95_strictly_greater_than"]
        ),
        **{
            f"{metric}_nonregression": _at_most(
                float(maximum_regressions[metric]),
                float(config["secondary_relative_regression_maximum"]),
            )
            for metric in SECONDARY_METRICS
        },
        "correlation": float(summary["minimum_per_device_correlation"])
        > float(config["correlation_strictly_greater_than"]),
        "gain_error": float(summary["minimum_per_device_gain_error"])
        > float(config["gain_error_strictly_greater_than"]),
    }
    passed = all(checks.values())
    return {
        "format": "fssr-amp-quality-teacher-objective-verdict-v1",
        "valid": True,
        "passed": passed,
        "verdict": str(config["go"] if passed else config["no_go"]),
        "invalid_reasons": [],
        "checks": checks,
        "per_device": summary["per_device"],
        "statistics": summary,
    }


evaluate_ampeg_slow_value_gate = evaluate_slow_value_gate
evaluate_objective_gate = evaluate_objective_verdict


__all__ = [
    "CANDIDATE_FAMILY",
    "COMPARATOR_FAMILIES",
    "CONTROL_FAMILY",
    "DEVELOPMENT_DEVICES",
    "QualityTeacherGateError",
    "evaluate_ampeg_slow_value_gate",
    "evaluate_objective_gate",
    "evaluate_objective_verdict",
    "evaluate_slow_value_gate",
    "select_global_comparator",
]
