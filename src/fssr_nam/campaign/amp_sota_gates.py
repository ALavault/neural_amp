"""Pure fail-closed gates for AMP-SOTA-PROTOTYPE-v1."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any

from .amp_sota_prototype_v1 import (
    CAMPAIGN_VERSION,
    CONFIRMATION_DEVICES,
    SotaPrototypeConfigError,
    validate_protocol_config,
)


class SotaPrototypeGateError(RuntimeError):
    """Raised when gate evidence or a decision-bearing threshold is malformed."""


def _at_least(value: float, threshold: float) -> bool:
    return value > threshold or math.isclose(
        value, threshold, rel_tol=1.0e-12, abs_tol=1.0e-12
    )


def _at_most(value: float, threshold: float) -> bool:
    return value < threshold or math.isclose(
        value, threshold, rel_tol=1.0e-12, abs_tol=1.0e-12
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SotaPrototypeGateError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SotaPrototypeGateError(f"{label} must be a sequence")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SotaPrototypeGateError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise SotaPrototypeGateError(f"{label} must be finite")
    return converted


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SotaPrototypeGateError(f"{label} must be an integer")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise SotaPrototypeGateError(f"{label} must be boolean")
    return value


def _require_equal(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise SotaPrototypeGateError(
            f"{label} changed: expected {expected!r}, got {value!r}"
        )


def _validate_gate_protocol(protocol: Mapping[str, Any]) -> None:
    try:
        validate_protocol_config(protocol)
    except SotaPrototypeConfigError as error:
        raise SotaPrototypeGateError(str(error)) from error

    axes = _mapping(
        _mapping(protocol.get("mechanism_screen"), "mechanism_screen").get("axes"),
        "mechanism_screen.axes",
    )
    expected_axes = {
        "approximant": {
            "control": "cubic_hermite_c1",
            "candidates": ["quintic_hermite_c2", "safe_rational_4_3"],
            "asr_improvement_db_minimum": 3.0,
            "approximation_error_regression_maximum": 0.01,
            "cost_reduction_minimum": 0.25,
            "asr_noninferiority_db": 0.5,
        },
        "slow_control": {
            "control": "causal_zero_order_hold",
            "candidates": [
                "causal_exponential_hold",
                "causal_slope_limited_hold",
            ],
            "dynamic_esr_improvement_minimum": 0.05,
            "parasite_regression_db_maximum": 0.5,
            "lookahead_samples_maximum": 0,
        },
        "resampler": {
            "control": "kaiser_windowed_sinc",
            "candidates": ["equiripple_halfband_polyphase"],
            "stopband_regression_db_maximum": 0.5,
            "cost_reduction_minimum": 0.25,
        },
    }
    for axis, expected in expected_axes.items():
        observed = _mapping(axes.get(axis), f"mechanism_screen.axes.{axis}")
        for key, value in expected.items():
            _require_equal(observed.get(key), value, f"{axis}.{key}")

    confirmation = _mapping(protocol.get("confirmation"), "confirmation")
    confirmation_literals = {
        "seeds": [0, 1, 2, 3, 4],
        "devices": list(CONFIRMATION_DEVICES),
        "bootstrap_replicates": 10_000,
        "bootstrap_seed": 20_260_830,
        "median_esr_improvement_minimum": 0.10,
        "bootstrap_lower_95_bound_strictly_greater_than": 0.0,
        "metric_relative_regression_maximum": 0.05,
        "noninferiority_metrics": ["mae", "log_mel", "mrstft"],
        "aliasing_regression_allowed": False,
    }
    for key, value in confirmation_literals.items():
        _require_equal(confirmation.get(key), value, f"confirmation.{key}")

    prototype = _mapping(protocol.get("prototype"), "prototype")
    prototype_literals = {
        "block_parity_max_absolute_error": 2.0e-5,
        "latency_samples_maximum": 64,
        "benchmark_block_size": 128,
        "p95_realtime_factor_strictly_less_than": 1.0,
        "finite_required": True,
        "deterministic_reset_required": True,
    }
    for key, value in prototype_literals.items():
        _require_equal(prototype.get(key), value, f"prototype.{key}")


def evaluate_preflight_gate(
    evidence: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Accept only a clean, metadata-only start of the prospective lineage."""
    _validate_gate_protocol(protocol)
    _require_equal(
        evidence.get("campaign_version"), CAMPAIGN_VERSION, "preflight campaign"
    )
    _require_equal(
        evidence.get("parent_campaign"),
        "AMP-QUALITY-ARCH-v3",
        "preflight parent campaign",
    )
    _require_equal(evidence.get("parent_verdict"), "INVALID", "parent verdict")

    checks = {
        name: _boolean(evidence.get(name), f"preflight {name}")
        for name in (
            "protocol_frozen",
            "data_audit_passed",
            "tests_passed",
            "lint_passed",
            "public_license_audit_passed",
            "source_file_disjoint_splits_verified",
        )
    }
    physical_reads = _integer(
        evidence.get("physical_audio_samples_read"), "physical audio samples read"
    )
    confirmation_reads = _integer(
        evidence.get("confirmation_audio_samples_read"),
        "confirmation audio samples read",
    )
    fm9_reads = _integer(evidence.get("fm9_audio_samples_read"), "FM9 samples read")
    resumed = _boolean(
        evidence.get("failed_or_invalid_run_resumed"), "failed run resume flag"
    )
    checks.update(
        {
            "metadata_only": physical_reads == 0,
            "confirmation_locked": confirmation_reads == 0,
            "fm9_locked": fm9_reads == 0,
            "no_failed_run_resume": resumed is False,
        }
    )
    return {
        "format": "fssr-amp-sota-preflight-gate-v1",
        "valid": True,
        "passed": all(checks.values()),
        "checks": checks,
        "physical_audio_samples_read": physical_reads,
        "confirmation_audio_samples_read": confirmation_reads,
        "fm9_audio_samples_read": fm9_reads,
    }


def _named_rows(
    rows: Any, expected: Sequence[str], label: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for item in _sequence(rows, label):
        row = _mapping(item, f"{label} row")
        name = row.get("name")
        if not isinstance(name, str) or name not in expected:
            raise SotaPrototypeGateError(f"{label} contains an unknown candidate")
        if name in indexed:
            raise SotaPrototypeGateError(f"duplicate {label} candidate: {name}")
        indexed[name] = row
    if set(indexed) != set(expected):
        raise SotaPrototypeGateError(f"{label} candidate evidence is incomplete")
    return indexed


def _metrics(row: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    return _mapping(row.get("metrics"), f"{label} metrics")


def _positive(value: Any, label: str) -> float:
    converted = _finite(value, label)
    if converted <= 0.0:
        raise SotaPrototypeGateError(f"{label} must be positive")
    return converted


def _approximant_axis(
    evidence: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    control = _mapping(evidence.get("control"), "approximant control")
    _require_equal(control.get("name"), spec["control"], "approximant control")
    control_metrics = _metrics(control, "approximant control")
    control_asr = _finite(control_metrics.get("asr_db"), "approximant control ASR")
    control_error = _positive(
        control_metrics.get("approximation_error"),
        "approximant control approximation error",
    )
    cost_available = evidence.get("cost_measurement_available", True)
    cost_available = _boolean(cost_available, "approximant cost availability")
    control_cost = (
        _positive(control_metrics.get("cost"), "approximant control cost")
        if cost_available
        else None
    )
    candidates = _named_rows(
        evidence.get("candidates"), spec["candidates"], "approximant"
    )
    results: dict[str, Any] = {}
    eligible: list[str] = []
    for name in spec["candidates"]:
        metrics = _metrics(candidates[name], f"approximant {name}")
        asr_gain = _finite(metrics.get("asr_db"), f"{name} ASR") - control_asr
        error_regression = (
            _finite(metrics.get("approximation_error"), f"{name} approximation error")
            - control_error
        ) / control_error
        cost_reduction = None
        if cost_available:
            assert control_cost is not None
            cost_reduction = (
                control_cost - _positive(metrics.get("cost"), f"{name} cost")
            ) / control_cost
        checks = {
            "quality_route": _at_least(
                asr_gain, float(spec["asr_improvement_db_minimum"])
            )
            and _at_most(
                error_regression,
                float(spec["approximation_error_regression_maximum"]),
            ),
            "cost_route": cost_reduction is not None
            and _at_least(cost_reduction, float(spec["cost_reduction_minimum"]))
            and _at_least(asr_gain, -float(spec["asr_noninferiority_db"])),
        }
        passed = any(checks.values())
        if passed:
            eligible.append(name)
        results[name] = {
            "passed": passed,
            "checks": checks,
            "asr_improvement_db": asr_gain,
            "approximation_error_relative_regression": error_regression,
            "cost_reduction": cost_reduction,
        }
    promoted = (
        sorted(
            eligible,
            key=lambda name: (
                not results[name]["checks"]["quality_route"],
                -results[name]["asr_improvement_db"],
                results[name]["approximation_error_relative_regression"],
                -(results[name]["cost_reduction"] or 0.0),
                name,
            ),
        )[0]
        if eligible
        else None
    )
    return {
        "passed": promoted is not None,
        "eligible_candidates": eligible,
        "promoted_candidate": promoted,
        "candidate_results": results,
        "cost_measurement_available": cost_available,
    }


def _slow_axis(evidence: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    control = _mapping(evidence.get("control"), "slow-control control")
    _require_equal(control.get("name"), spec["control"], "slow-control control")
    control_metrics = _metrics(control, "slow-control control")
    control_esr = _positive(
        control_metrics.get("dynamic_esr"), "slow-control dynamic ESR"
    )
    control_parasite = _finite(
        control_metrics.get("parasite_db"), "slow-control parasite"
    )
    candidates = _named_rows(
        evidence.get("candidates"), spec["candidates"], "slow-control"
    )
    results: dict[str, Any] = {}
    eligible: list[str] = []
    for name in spec["candidates"]:
        metrics = _metrics(candidates[name], f"slow-control {name}")
        improvement = (
            control_esr - _finite(metrics.get("dynamic_esr"), f"{name} dynamic ESR")
        ) / control_esr
        parasite_regression = (
            _finite(metrics.get("parasite_db"), f"{name} parasite") - control_parasite
        )
        lookahead = _integer(metrics.get("lookahead_samples"), f"{name} lookahead")
        causal = _boolean(candidates[name].get("causal"), f"{name} causality")
        checks = {
            "dynamic_esr": _at_least(
                improvement, float(spec["dynamic_esr_improvement_minimum"])
            ),
            "parasite": _at_most(
                parasite_regression,
                float(spec["parasite_regression_db_maximum"]),
            ),
            "zero_lookahead": 0 <= lookahead <= int(spec["lookahead_samples_maximum"]),
            "causal": causal,
        }
        passed = all(checks.values())
        if passed:
            eligible.append(name)
        results[name] = {
            "passed": passed,
            "checks": checks,
            "dynamic_esr_relative_improvement": improvement,
            "parasite_regression_db": parasite_regression,
            "lookahead_samples": lookahead,
        }
    promoted = (
        sorted(
            eligible,
            key=lambda name: (
                -results[name]["dynamic_esr_relative_improvement"],
                results[name]["parasite_regression_db"],
                name,
            ),
        )[0]
        if eligible
        else None
    )
    return {
        "passed": promoted is not None,
        "eligible_candidates": eligible,
        "promoted_candidate": promoted,
        "candidate_results": results,
    }


def _resampler_axis(
    evidence: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    control = _mapping(evidence.get("control"), "resampler control")
    _require_equal(control.get("name"), spec["control"], "resampler control")
    control_metrics = _metrics(control, "resampler control")
    control_stopband = _finite(
        control_metrics.get("stopband_attenuation_db"), "resampler stopband"
    )
    cost_available = evidence.get("cost_measurement_available", True)
    cost_available = _boolean(cost_available, "resampler cost availability")
    control_cost = (
        _positive(control_metrics.get("cost"), "resampler control cost")
        if cost_available
        else None
    )
    candidates = _named_rows(
        evidence.get("candidates"), spec["candidates"], "resampler"
    )
    results: dict[str, Any] = {}
    eligible: list[str] = []
    for name in spec["candidates"]:
        metrics = _metrics(candidates[name], f"resampler {name}")
        stopband_regression = control_stopband - _finite(
            metrics.get("stopband_attenuation_db"), f"{name} stopband"
        )
        cost_reduction = None
        if cost_available:
            assert control_cost is not None
            cost_reduction = (
                control_cost - _positive(metrics.get("cost"), f"{name} cost")
            ) / control_cost
        checks = {
            "stopband": _at_most(
                stopband_regression,
                float(spec["stopband_regression_db_maximum"]),
            ),
            "cost": cost_reduction is not None
            and _at_least(cost_reduction, float(spec["cost_reduction_minimum"])),
        }
        passed = all(checks.values())
        if passed:
            eligible.append(name)
        results[name] = {
            "passed": passed,
            "checks": checks,
            "stopband_regression_db": stopband_regression,
            "cost_reduction": cost_reduction,
        }
    promoted = eligible[0] if eligible else None
    return {
        "passed": promoted is not None,
        "eligible_candidates": eligible,
        "promoted_candidate": promoted,
        "candidate_results": results,
        "cost_measurement_available": cost_available,
    }


def evaluate_mechanism_promotion_gate(
    evidence: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Promote at most one independently measured candidate on each axis."""
    _validate_gate_protocol(protocol)
    _require_equal(
        evidence.get("campaign_version"), CAMPAIGN_VERSION, "mechanism campaign"
    )
    screen = _mapping(protocol["mechanism_screen"], "mechanism_screen")
    _require_equal(
        evidence.get("evidence_tier"), screen["evidence_tier"], "mechanism tier"
    )
    reads = _integer(
        evidence.get("physical_audio_samples_read"), "mechanism physical reads"
    )
    if reads != 0:
        raise SotaPrototypeGateError("mechanism screen must not read physical audio")
    _require_equal(evidence.get("seed"), screen["seed"], "mechanism synthetic seed")
    axes_evidence = _mapping(evidence.get("axes"), "mechanism axes")
    expected_axes = ("approximant", "slow_control", "resampler")
    if set(axes_evidence) != set(expected_axes):
        raise SotaPrototypeGateError("mechanism axis evidence is incomplete")
    for axis in expected_axes:
        axis_evidence = _mapping(axes_evidence[axis], f"mechanism {axis}")
        if not _boolean(
            axis_evidence.get("other_axes_at_control"),
            f"mechanism {axis} isolation",
        ):
            raise SotaPrototypeGateError(f"mechanism {axis} was not isolated")

    specs = _mapping(screen["axes"], "mechanism axis specs")
    results = {
        "approximant": _approximant_axis(
            _mapping(axes_evidence["approximant"], "approximant evidence"),
            _mapping(specs["approximant"], "approximant spec"),
        ),
        "slow_control": _slow_axis(
            _mapping(axes_evidence["slow_control"], "slow-control evidence"),
            _mapping(specs["slow_control"], "slow-control spec"),
        ),
        "resampler": _resampler_axis(
            _mapping(axes_evidence["resampler"], "resampler evidence"),
            _mapping(specs["resampler"], "resampler spec"),
        ),
    }
    promotions = {
        axis: result["promoted_candidate"]
        for axis, result in results.items()
        if result["promoted_candidate"] is not None
    }
    maximum = int(screen["maximum_promotions_per_axis"])
    if maximum != 1:
        raise SotaPrototypeGateError("mechanism promotion cap changed")
    return {
        "format": "fssr-amp-sota-mechanism-gate-v1",
        "valid": True,
        "passed": bool(promotions),
        "axis_results": results,
        "promoted_components": promotions,
        "maximum_promotions_per_axis": maximum,
        "combined_run_count_maximum": 1
        if screen["combine_promoted_components_once"] is True
        else 0,
        "physical_audio_samples_read": reads,
    }


def _confirmation_rows(
    raw_rows: Any, protocol: Mapping[str, Any]
) -> tuple[list[dict[str, float]], dict[str, set[str]]]:
    confirmation = _mapping(protocol["confirmation"], "confirmation")
    devices = tuple(confirmation["devices"])
    seeds = tuple(confirmation["seeds"])
    observed: set[tuple[str, int, str]] = set()
    source_sets: dict[tuple[str, int], set[str]] = {
        (device, seed): set() for device in devices for seed in seeds
    }
    normalized: list[dict[str, float]] = []
    for item in _sequence(raw_rows, "confirmation rows"):
        row = _mapping(item, "confirmation row")
        device = row.get("device")
        seed = row.get("seed")
        source = row.get("source")
        if device not in devices:
            raise SotaPrototypeGateError(f"unknown confirmation device: {device!r}")
        if isinstance(seed, bool) or seed not in seeds:
            raise SotaPrototypeGateError(f"invalid confirmation seed: {seed!r}")
        if not isinstance(source, str) or not source:
            raise SotaPrototypeGateError("confirmation source must be non-empty")
        key = (str(device), int(seed), source)
        if key in observed:
            raise SotaPrototypeGateError("duplicate confirmation device/seed/source")
        observed.add(key)
        source_sets[(str(device), int(seed))].add(source)

        baseline = _mapping(row.get("baseline"), "confirmation baseline metrics")
        candidate = _mapping(row.get("candidate"), "confirmation candidate metrics")
        values: dict[str, float] = {}
        for metric in ("esr", "mae", "log_mel", "mrstft"):
            baseline_value = _positive(
                baseline.get(metric), f"confirmation baseline {metric}"
            )
            candidate_value = _finite(
                candidate.get(metric), f"confirmation candidate {metric}"
            )
            if candidate_value < 0.0:
                raise SotaPrototypeGateError(
                    f"confirmation candidate {metric} must be nonnegative"
                )
            values[f"{metric}_relative_change"] = (
                candidate_value - baseline_value
            ) / baseline_value
        baseline_alias = _finite(
            baseline.get("alias_residual_db"), "confirmation baseline alias residual"
        )
        candidate_alias = _finite(
            candidate.get("alias_residual_db"),
            "confirmation candidate alias residual",
        )
        values["alias_regression_db"] = candidate_alias - baseline_alias
        normalized.append(values | {"device": str(device), "seed": float(seed)})

    by_device: dict[str, set[str]] = {}
    for device in devices:
        sets = [source_sets[(device, seed)] for seed in seeds]
        if any(not sources for sources in sets):
            raise SotaPrototypeGateError(
                f"{device} confirmation requires every frozen seed"
            )
        if any(sources != sets[0] for sources in sets[1:]):
            raise SotaPrototypeGateError(
                f"{device} confirmation source set changed across seeds"
            )
        by_device[device] = sets[0]
    return normalized, by_device


def _runtime_checks(
    evidence: Any, protocol: Mapping[str, Any]
) -> tuple[dict[str, bool], dict[str, Any]]:
    runtime = _mapping(evidence, "prototype runtime evidence")
    spec = _mapping(protocol["prototype"], "prototype")
    finite_outputs = _boolean(runtime.get("finite_outputs"), "finite outputs")
    deterministic_reset = _boolean(
        runtime.get("deterministic_reset"), "deterministic reset"
    )
    parity = _finite(
        runtime.get("block_parity_max_absolute_error"), "block parity error"
    )
    latency = _integer(runtime.get("latency_samples"), "latency samples")
    block_size = _integer(runtime.get("benchmark_block_size"), "benchmark block size")
    p95_rtf = _finite(runtime.get("p95_realtime_factor"), "p95 realtime factor")
    if parity < 0.0 or latency < 0 or p95_rtf < 0.0:
        raise SotaPrototypeGateError("prototype runtime metrics must be nonnegative")
    _require_equal(block_size, spec["benchmark_block_size"], "benchmark block size")
    checks = {
        "finite": finite_outputs is spec["finite_required"],
        "deterministic_reset": deterministic_reset
        is spec["deterministic_reset_required"],
        "block_parity": _at_most(
            parity, float(spec["block_parity_max_absolute_error"])
        ),
        "latency": latency <= int(spec["latency_samples_maximum"]),
        "p95_realtime_factor": p95_rtf
        < float(spec["p95_realtime_factor_strictly_less_than"]),
    }
    return checks, {
        "block_parity_max_absolute_error": parity,
        "latency_samples": latency,
        "benchmark_block_size": block_size,
        "p95_realtime_factor": p95_rtf,
    }


def evaluate_confirmation_gate(
    evidence: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate held-out fidelity and runnable-prototype constraints together."""
    _validate_gate_protocol(protocol)
    _require_equal(
        evidence.get("campaign_version"), CAMPAIGN_VERSION, "confirmation campaign"
    )
    for name in (
        "candidate_locked_before_confirmation",
        "baseline_selected_on_development_only",
        "same_pairs_and_splits",
        "confirmation_excluded_from_selection",
    ):
        if not _boolean(evidence.get(name), f"confirmation {name}"):
            raise SotaPrototypeGateError(f"confirmation boundary failed: {name}")
    baseline_name = evidence.get("baseline_name")
    candidate_name = evidence.get("candidate_name")
    if not isinstance(baseline_name, str) or not baseline_name:
        raise SotaPrototypeGateError("confirmation baseline_name must be non-empty")
    if not isinstance(candidate_name, str) or not candidate_name:
        raise SotaPrototypeGateError("confirmation candidate_name must be non-empty")

    rows, source_sets = _confirmation_rows(evidence.get("rows"), protocol)
    confirmation = _mapping(protocol["confirmation"], "confirmation")
    bootstrap = _mapping(evidence.get("bootstrap"), "confirmation bootstrap")
    bootstrap_literals = {
        "method": "paired-hierarchical-sota-confirmation-v1",
        "replicates": confirmation["bootstrap_replicates"],
        "seed": confirmation["bootstrap_seed"],
        "hierarchy": ["seed", "source"],
        "devices_fixed_strata": list(CONFIRMATION_DEVICES),
        "windows_resampled": False,
    }
    for key, value in bootstrap_literals.items():
        _require_equal(bootstrap.get(key), value, f"confirmation bootstrap {key}")
    lower_bound = _finite(
        bootstrap.get("esr_relative_improvement_lower_95_bound"),
        "confirmation ESR lower 95 bound",
    )

    esr_improvements = [-row["esr_relative_change"] for row in rows]
    median_esr_improvement = float(median(esr_improvements))
    regression_metrics = tuple(confirmation["noninferiority_metrics"])
    median_regressions = {
        metric: float(median(row[f"{metric}_relative_change"] for row in rows))
        for metric in regression_metrics
    }
    per_device_regressions = {
        device: {
            metric: float(
                median(
                    row[f"{metric}_relative_change"]
                    for row in rows
                    if row["device"] == device
                )
            )
            for metric in regression_metrics
        }
        for device in CONFIRMATION_DEVICES
    }
    maximum_regressions = {
        metric: max(
            per_device_regressions[device][metric] for device in CONFIRMATION_DEVICES
        )
        for metric in regression_metrics
    }
    maximum_alias_regression = max(row["alias_regression_db"] for row in rows)
    runtime_checks, runtime_values = _runtime_checks(evidence.get("runtime"), protocol)
    checks = {
        "median_esr_improvement": _at_least(
            median_esr_improvement,
            float(confirmation["median_esr_improvement_minimum"]),
        ),
        "bootstrap_lower_95_bound": lower_bound
        > float(confirmation["bootstrap_lower_95_bound_strictly_greater_than"]),
        **{
            f"{metric}_noninferiority": _at_most(
                maximum_regressions[metric],
                float(confirmation["metric_relative_regression_maximum"]),
            )
            for metric in regression_metrics
        },
        "aliasing_no_regression": maximum_alias_regression <= 0.0,
        **runtime_checks,
    }
    return {
        "format": "fssr-amp-sota-confirmation-gate-v1",
        "valid": True,
        "passed": all(checks.values()),
        "checks": checks,
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "observation_count": len(rows),
        "sources_per_device": {
            device: len(sources) for device, sources in source_sets.items()
        },
        "median_esr_relative_improvement": median_esr_improvement,
        "bootstrap_lower_95_bound": lower_bound,
        "median_metric_relative_regressions": median_regressions,
        "maximum_per_device_metric_relative_regressions": maximum_regressions,
        "maximum_alias_regression_db": maximum_alias_regression,
        "runtime": runtime_values,
    }


__all__ = [
    "SotaPrototypeGateError",
    "evaluate_confirmation_gate",
    "evaluate_mechanism_promotion_gate",
    "evaluate_preflight_gate",
]
