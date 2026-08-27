"""Pure, fail-closed evaluation of the preregistered R1 diagnostic gates."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from fssr_nam.training.r1_diagnostic import (
    A2_LINEAR_MACS_PER_SAMPLE,
    LOCKED_CHECKPOINT_STEPS,
)


class R1GateEvidenceError(RuntimeError):
    """Required completed evidence is missing, malformed, or non-finite."""


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R1GateEvidenceError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise R1GateEvidenceError(f"{label} must be finite")
    return number


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R1GateEvidenceError(f"{label} must be a mapping")
    return value


def validate_counted_metrics(
    metrics: Mapping[str, Any],
    *,
    stage: str,
    device: str,
    model: str,
    loss: str,
    seed: int = 0,
) -> None:
    """Require a full 5000-update artifact without consulting sealed test data."""
    expected = {
        "stage": stage,
        "device": device,
        "model": model,
        "loss": loss,
        "seed": seed,
    }
    for field, value in expected.items():
        if metrics.get(field) != value:
            raise R1GateEvidenceError(
                f"metrics {field} mismatch: expected {value!r}, "
                f"got {metrics.get(field)!r}"
            )
    if metrics.get("preflight") is not False:
        raise R1GateEvidenceError("preflight metrics cannot unlock a scientific gate")
    if metrics.get("optimizer_steps") != 5_000:
        raise R1GateEvidenceError("gate evidence must contain exactly 5000 updates")
    if tuple(metrics.get("checkpoint_steps", ())) != LOCKED_CHECKPOINT_STEPS:
        raise R1GateEvidenceError("gate evidence lacks exact 200/1000/5000 snapshots")
    snapshots = _mapping(metrics.get("snapshots"), "snapshots")
    if set(snapshots) != {str(step) for step in LOCKED_CHECKPOINT_STEPS}:
        raise R1GateEvidenceError("snapshot set must be exactly 200/1000/5000")
    for step in LOCKED_CHECKPOINT_STEPS:
        snapshot = _mapping(snapshots[str(step)], f"snapshot {step}")
        validation = _mapping(snapshot.get("validation"), f"snapshot {step} validation")
        for field in (
            "esr",
            "gain_error",
            "output_energy",
            "target_energy",
            "residual_energy_ratio",
        ):
            _finite(validation.get(field), f"snapshot {step} validation {field}")
        if snapshot.get("gradients_finite") is not True:
            raise R1GateEvidenceError(f"snapshot {step} has non-finite gradients")
        gradients = _mapping(
            snapshot.get("gradient_norms_by_block"),
            f"snapshot {step} gradient_norms_by_block",
        )
        if not gradients:
            raise R1GateEvidenceError(f"snapshot {step} has no block gradients")
        for name, value in gradients.items():
            _finite(value, f"snapshot {step} gradient {name}")
    selected = _mapping(metrics.get("selected_checkpoint"), "selected_checkpoint")
    selected_step = selected.get("step")
    if selected_step not in LOCKED_CHECKPOINT_STEPS:
        raise R1GateEvidenceError("selected checkpoint is not a common snapshot")
    selected_esr = _finite(selected.get("validation_esr"), "selected validation ESR")
    snapshot_esr = _finite(
        _mapping(snapshots[str(selected_step)], "selected snapshot")["validation"][
            "esr"
        ],
        "selected snapshot ESR",
    )
    if selected_esr != snapshot_esr:
        raise R1GateEvidenceError("selected checkpoint ESR differs from its snapshot")
    best = _mapping(metrics.get("best_validation"), "best_validation")
    if _finite(best.get("esr"), "best validation ESR") != selected_esr:
        raise R1GateEvidenceError("best_validation does not match selected checkpoint")
    selected_validation = _mapping(
        metrics.get("selected_validation"), "selected_validation"
    )
    if (
        _finite(selected_validation.get("esr"), "selected validation ESR")
        != selected_esr
    ):
        raise R1GateEvidenceError(
            "post-reload validation differs from selected snapshot"
        )
    checks = _mapping(metrics.get("checks"), "checks")
    if checks.get("all_gradients_finite") is not True:
        raise R1GateEvidenceError("run reports non-finite gradients")
    if checks.get("finite_selected_validation_prediction") is not True:
        raise R1GateEvidenceError("selected validation prediction is non-finite")
    if checks.get("sealed_test_opened") is not False:
        raise R1GateEvidenceError("gate evidence opened the sealed test split")


def _index(
    rows: Sequence[Mapping[str, Any]],
    expected: set[tuple[str, str, str]],
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for metrics in rows:
        key = (
            str(metrics.get("device")),
            str(metrics.get("model")),
            str(metrics.get("loss")),
        )
        if key in indexed:
            raise R1GateEvidenceError(f"duplicate gate evidence: {key}")
        indexed[key] = metrics
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise R1GateEvidenceError(
            f"gate evidence matrix mismatch; missing={missing}, extra={extra}"
        )
    return indexed


def _validation(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(metrics.get("best_validation"), "best_validation")


def _relative_improvement(baseline: float, candidate: float, label: str) -> float:
    if baseline <= 0.0:
        raise R1GateEvidenceError(f"{label} baseline ESR must be positive")
    return (baseline - candidate) / baseline


def evaluate_factorial_gate(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Select M4/Wright and apply every frozen factorial continuation check."""
    expected = {
        (device, model, loss)
        for device in ("fulltone", "bigmuff")
        for model in ("a2", "s3")
        for loss in ("m4", "wright")
    }
    indexed = _index(rows, expected)
    for (device, model, loss), metrics in indexed.items():
        validate_counted_metrics(
            metrics, stage="factorial", device=device, model=model, loss=loss
        )
    medians = {
        loss: statistics.median(
            _finite(_validation(indexed[(device, "s3", loss)])["esr"], "S3 ESR")
            for device in ("fulltone", "bigmuff")
        )
        for loss in ("m4", "wright")
    }
    gate = _mapping(config.get("promotion_gate"), "factorial promotion_gate")
    tolerance = _finite(gate.get("tie_relative_tolerance"), "tie tolerance")
    denominator = min(medians.values())
    if denominator <= 0.0:
        raise R1GateEvidenceError("S3 median validation ESR must be positive")
    tied = abs(medians["m4"] - medians["wright"]) / denominator <= tolerance
    selected_loss = (
        str(gate.get("tie_winner")) if tied else min(medians, key=medians.get)
    )
    if selected_loss not in {"m4", "wright"}:
        raise R1GateEvidenceError("factorial tie winner is not a declared loss")
    bigmuff_selected = _validation(indexed[("bigmuff", "s3", selected_loss)])
    bigmuff_m4 = _validation(indexed[("bigmuff", "s3", "m4")])
    fulltone_selected = _validation(indexed[("fulltone", "s3", selected_loss)])
    fulltone_m4 = _validation(indexed[("fulltone", "s3", "m4")])
    gain_error = _finite(bigmuff_selected["gain_error"], "Big Muff S3 gain error")
    bigmuff_improvement = _relative_improvement(
        _finite(bigmuff_m4["esr"], "Big Muff S3 M4 ESR"),
        _finite(bigmuff_selected["esr"], "Big Muff selected S3 ESR"),
        "Big Muff S3 M4",
    )
    fulltone_regression = -_relative_improvement(
        _finite(fulltone_m4["esr"], "Fulltone S3 M4 ESR"),
        _finite(fulltone_selected["esr"], "Fulltone selected S3 ESR"),
        "Fulltone S3 M4",
    )
    checks = {
        "bigmuff_gain_error": {
            "value": gain_error,
            "operator": ">",
            "threshold": _finite(
                gate.get("bigmuff_gain_error_greater_than"), "Big Muff gain threshold"
            ),
        },
        "bigmuff_s3_esr_improvement_over_m4": {
            "value": bigmuff_improvement,
            "operator": ">=",
            "threshold": _finite(
                gate.get("bigmuff_s3_esr_improvement_over_m4_minimum"),
                "Big Muff improvement threshold",
            ),
        },
        "fulltone_s3_esr_regression": {
            "value": fulltone_regression,
            "operator": "<=",
            "threshold": _finite(
                gate.get("fulltone_s3_esr_regression_maximum"),
                "Fulltone regression threshold",
            ),
        },
    }
    checks["bigmuff_gain_error"]["passed"] = (
        gain_error > checks["bigmuff_gain_error"]["threshold"]
    )
    checks["bigmuff_s3_esr_improvement_over_m4"]["passed"] = (
        bigmuff_improvement >= checks["bigmuff_s3_esr_improvement_over_m4"]["threshold"]
    )
    checks["fulltone_s3_esr_regression"]["passed"] = (
        fulltone_regression <= checks["fulltone_s3_esr_regression"]["threshold"]
    )
    passed = all(bool(item["passed"]) for item in checks.values())
    return {
        "passed": passed,
        "decision": "promoted" if passed else "failed",
        "selected_loss": selected_loss,
        "selection": {
            "criterion": "lowest_median_s3_validation_esr",
            "median_s3_validation_esr": medians,
            "relative_tie": tied,
            "tie_relative_tolerance": tolerance,
        },
        "checks": checks,
        "source_run_ids": sorted(str(row["run_id"]) for row in rows),
        "sealed_test_used": False,
    }


def evaluate_horizon_gate(
    horizon_rows: Sequence[Mapping[str, Any]],
    factorial_rows: Sequence[Mapping[str, Any]],
    *,
    selected_loss: str,
    config: Mapping[str, Any],
    promoted_residuals: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Evaluate exact RF31-to-RF2047 gap closure on both devices."""
    expected_horizon = {
        (device, model, selected_loss)
        for device in ("fulltone", "bigmuff")
        for model in ("rf31", "rf2047")
    }
    horizon = _index(horizon_rows, expected_horizon)
    expected_factorial = {
        (device, model, selected_loss)
        for device in ("fulltone", "bigmuff")
        for model in ("a2", "s3")
    }
    factorial = _index(factorial_rows, expected_factorial)
    for (device, model, loss), metrics in horizon.items():
        validate_counted_metrics(
            metrics, stage="horizon", device=device, model=model, loss=loss
        )
    for (device, model, loss), metrics in factorial.items():
        validate_counted_metrics(
            metrics, stage="factorial", device=device, model=model, loss=loss
        )
    gate = _mapping(config.get("promotion_gate"), "horizon promotion_gate")
    thresholds = {
        "fulltone": _finite(
            gate.get("fulltone_gap_closure_minimum"), "Fulltone closure"
        ),
        "bigmuff": _finite(gate.get("bigmuff_gap_closure_minimum"), "Big Muff closure"),
    }
    devices: dict[str, Any] = {}
    passed = True
    for device in ("fulltone", "bigmuff"):
        a2_esr = _finite(
            _validation(factorial[(device, "a2", selected_loss)])["esr"], "A2 ESR"
        )
        rf31 = horizon[(device, "rf31", selected_loss)]
        rf2047 = horizon[(device, "rf2047", selected_loss)]
        rf31_esr = _finite(_validation(rf31)["esr"], "RF31 ESR")
        rf2047_esr = _finite(_validation(rf2047)["esr"], "RF2047 ESR")
        gap = rf31_esr - a2_esr
        if gap <= 0.0:
            raise R1GateEvidenceError(f"{device} RF31-to-A2 gap is not positive")
        closure = (rf31_esr - rf2047_esr) / gap
        residual_ratio = _finite(
            _validation(rf2047)["residual_energy_ratio"], "RF2047 residual ratio"
        )
        cost = _finite(rf2047.get("theoretical_macs_per_sample"), "RF2047 cost")
        device_checks = {
            "gap_closure": closure >= thresholds[device],
            "finite_gradients": _mapping(rf2047["checks"], "RF2047 checks").get(
                "all_gradients_finite"
            )
            is True,
            "residual_not_dominant": residual_ratio < 1.0,
            "cost_at_most_a2": cost <= A2_LINEAR_MACS_PER_SAMPLE,
        }
        passed = passed and all(device_checks.values())
        devices[device] = {
            "a2_validation_esr": a2_esr,
            "rf31_validation_esr": rf31_esr,
            "rf2047_validation_esr": rf2047_esr,
            "gap_closure": closure,
            "gap_closure_minimum": thresholds[device],
            "residual_energy_ratio": residual_ratio,
            "theoretical_macs_per_sample": cost,
            "checks": device_checks,
        }
    if set(promoted_residuals) != {"fulltone", "bigmuff"}:
        raise R1GateEvidenceError("horizon gate needs both promoted RF2047 references")
    return {
        "passed": passed,
        "decision": "promoted" if passed else "failed",
        "selected_loss": selected_loss,
        "promoted_model": "rf2047" if passed else None,
        "promoted_residuals": dict(promoted_residuals) if passed else {},
        "devices": devices,
        "source_run_ids": sorted(
            str(row["run_id"]) for row in (*horizon_rows, *factorial_rows)
        ),
        "sealed_test_used": False,
    }


def evaluate_cascade_synthetic_gate(
    rows: Sequence[Mapping[str, Any]], *, selected_loss: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    expected = {("synthetic", model, selected_loss) for model in ("mono", "cascade")}
    indexed = _index(rows, expected)
    for (_, model, loss), metrics in indexed.items():
        validate_counted_metrics(
            metrics, stage="cascade", device="synthetic", model=model, loss=loss
        )
    mono = _finite(
        _validation(indexed[("synthetic", "mono", selected_loss)])["esr"],
        "synthetic mono ESR",
    )
    cascade = _finite(
        _validation(indexed[("synthetic", "cascade", selected_loss)])["esr"],
        "synthetic cascade ESR",
    )
    reduction = _relative_improvement(mono, cascade, "synthetic mono")
    threshold = _finite(
        _mapping(config.get("promotion_gate"), "cascade promotion_gate").get(
            "synthetic_esr_reduction_minimum"
        ),
        "synthetic reduction threshold",
    )
    passed = reduction >= threshold
    return {
        "passed": passed,
        "decision": "promoted" if passed else "failed",
        "selected_loss": selected_loss,
        "mono_validation_esr": mono,
        "cascade_validation_esr": cascade,
        "esr_reduction": reduction,
        "minimum": threshold,
        "source_run_ids": sorted(str(row["run_id"]) for row in rows),
        "sealed_test_used": False,
    }


def evaluate_cascade_physical_gate(
    cascade_rows: Sequence[Mapping[str, Any]],
    horizon_rows: Sequence[Mapping[str, Any]],
    *,
    selected_loss: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_cascade = {
        (device, "cascade", selected_loss) for device in ("bigmuff", "fulltone")
    }
    cascades = _index(cascade_rows, expected_cascade)
    expected_horizon = {
        (device, "rf2047", selected_loss) for device in ("bigmuff", "fulltone")
    }
    horizons = _index(horizon_rows, expected_horizon)
    for (device, model, loss), metrics in cascades.items():
        validate_counted_metrics(
            metrics, stage="cascade", device=device, model=model, loss=loss
        )
    for (device, model, loss), metrics in horizons.items():
        validate_counted_metrics(
            metrics, stage="horizon", device=device, model=model, loss=loss
        )
    gate = _mapping(config.get("promotion_gate"), "cascade promotion_gate")
    bigmuff_mono = _validation(horizons[("bigmuff", "rf2047", selected_loss)])
    bigmuff_cascade = _validation(cascades[("bigmuff", "cascade", selected_loss)])
    bigmuff_improvement = _relative_improvement(
        _finite(bigmuff_mono["esr"], "Big Muff RF2047 ESR"),
        _finite(bigmuff_cascade["esr"], "Big Muff cascade ESR"),
        "Big Muff RF2047",
    )
    mono_gain = abs(_finite(bigmuff_mono["gain_error"], "Big Muff RF2047 gain error"))
    cascade_gain = abs(
        _finite(bigmuff_cascade["gain_error"], "Big Muff cascade gain error")
    )
    if mono_gain <= 1.0e-12:
        gain_ratio = 0.0 if cascade_gain <= 1.0e-12 else math.inf
    else:
        gain_ratio = cascade_gain / mono_gain
    fulltone_mono = _validation(horizons[("fulltone", "rf2047", selected_loss)])
    fulltone_cascade = _validation(cascades[("fulltone", "cascade", selected_loss)])
    fulltone_regression = -_relative_improvement(
        _finite(fulltone_mono["esr"], "Fulltone RF2047 ESR"),
        _finite(fulltone_cascade["esr"], "Fulltone cascade ESR"),
        "Fulltone RF2047",
    )
    thresholds = {
        "bigmuff_esr_improvement": _finite(
            gate.get("bigmuff_esr_improvement_minimum"), "Big Muff improvement"
        ),
        "bigmuff_gain_error_ratio": _finite(
            gate.get("bigmuff_absolute_gain_error_ratio_maximum"), "Big Muff gain ratio"
        ),
        "fulltone_esr_regression": _finite(
            gate.get("fulltone_esr_regression_maximum"), "Fulltone regression"
        ),
    }
    checks = {
        "bigmuff_esr_improvement": bigmuff_improvement
        >= thresholds["bigmuff_esr_improvement"],
        "bigmuff_gain_error_ratio": gain_ratio
        <= thresholds["bigmuff_gain_error_ratio"],
        "fulltone_esr_regression": fulltone_regression
        <= thresholds["fulltone_esr_regression"],
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "decision": "promoted" if passed else "failed",
        "selected_loss": selected_loss,
        "values": {
            "bigmuff_esr_improvement": bigmuff_improvement,
            "bigmuff_absolute_gain_error_ratio": gain_ratio,
            "fulltone_esr_regression": fulltone_regression,
        },
        "thresholds": thresholds,
        "checks": checks,
        "source_run_ids": sorted(
            str(row["run_id"]) for row in (*cascade_rows, *horizon_rows)
        ),
        "sealed_test_used": False,
    }
