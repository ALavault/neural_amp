"""Paired hierarchical statistics for AMP-QUALITY-TEACHER-v1."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

CONFIRMATION_DEVICES = ("rodent", "fuzzy_logic")
CONFIRMATION_SEEDS = (0, 1, 2)
SECONDARY_METRICS = ("mae", "log_mel", "envelope_transient")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_831

_BOOTSTRAP_METRICS = (
    "esr_relative_improvement",
    "l1_plus_mrstft_relative_improvement",
)
_SUMMARY_METRICS = (
    *_BOOTSTRAP_METRICS,
    *(f"{metric}_relative_regression" for metric in SECONDARY_METRICS),
    "correlation",
    "gain_error",
)


class QualityTeacherStatisticsError(ValueError):
    """Rows do not match the frozen paired confirmation hierarchy."""


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualityTeacherStatisticsError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise QualityTeacherStatisticsError(f"{label} must be finite")
    return result


def _nonnegative(value: object, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise QualityTeacherStatisticsError(f"{label} must be nonnegative")
    return result


def _positive(value: object, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise QualityTeacherStatisticsError(f"{label} must be positive")
    return result


def _metrics(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualityTeacherStatisticsError(f"{label} must be a mapping")
    return value


def _file_id(row: Mapping[str, Any]) -> str:
    values = [
        row[name]
        for name in ("file", "file_id", "source")
        if name in row and row[name] is not None
    ]
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise QualityTeacherStatisticsError("file identifier must be non-empty")
    if any(value != values[0] for value in values[1:]):
        raise QualityTeacherStatisticsError("file identifier aliases disagree")
    return str(values[0])


def l1_plus_mrstft(metrics: Mapping[str, Any], label: str) -> float:
    """Read the combined loss, deriving it from L1 and MR-STFT when needed."""
    combined = metrics.get("l1_plus_mrstft")
    components_present = "l1" in metrics or "mrstft" in metrics
    if components_present and not ("l1" in metrics and "mrstft" in metrics):
        raise QualityTeacherStatisticsError(
            f"{label} requires both l1 and mrstft components"
        )
    derived: float | None = None
    if components_present:
        derived = _nonnegative(metrics.get("l1"), f"{label} l1") + _nonnegative(
            metrics.get("mrstft"), f"{label} mrstft"
        )
    if combined is None:
        if derived is None:
            raise QualityTeacherStatisticsError(
                f"{label} requires l1_plus_mrstft or l1 and mrstft"
            )
        return derived
    result = _nonnegative(combined, f"{label} l1_plus_mrstft")
    if derived is not None and not math.isclose(
        result, derived, rel_tol=1.0e-12, abs_tol=1.0e-12
    ):
        raise QualityTeacherStatisticsError(
            f"{label} l1_plus_mrstft disagrees with l1 + mrstft"
        )
    return result


def _paired_values(
    comparator: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
    comparator_esr = _positive(comparator.get("esr"), "comparator ESR")
    candidate_esr = _nonnegative(candidate.get("esr"), "candidate ESR")
    comparator_l1_mrstft = l1_plus_mrstft(comparator, "comparator")
    if comparator_l1_mrstft <= 0.0:
        raise QualityTeacherStatisticsError(
            "comparator l1_plus_mrstft must be positive"
        )
    candidate_l1_mrstft = l1_plus_mrstft(candidate, "candidate")
    values = {
        "esr_relative_improvement": (comparator_esr - candidate_esr) / comparator_esr,
        "l1_plus_mrstft_relative_improvement": (
            comparator_l1_mrstft - candidate_l1_mrstft
        )
        / comparator_l1_mrstft,
    }
    for metric in SECONDARY_METRICS:
        comparator_value = _positive(comparator.get(metric), f"comparator {metric}")
        candidate_value = _nonnegative(candidate.get(metric), f"candidate {metric}")
        values[f"{metric}_relative_regression"] = (
            candidate_value - comparator_value
        ) / comparator_value
    correlation = _finite(candidate.get("correlation"), "candidate correlation")
    if correlation < -1.0 or correlation > 1.0:
        raise QualityTeacherStatisticsError("candidate correlation must lie in [-1, 1]")
    values["correlation"] = correlation
    values["gain_error"] = _finite(candidate.get("gain_error"), "candidate gain_error")
    return values


def _normalize_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
    indexed: dict[str, dict[str, dict[int, dict[str, float]]]] = {
        device: defaultdict(dict) for device in CONFIRMATION_DEVICES
    }
    for item in rows:
        if not isinstance(item, Mapping):
            raise QualityTeacherStatisticsError("confirmation row must be a mapping")
        device = item.get("device")
        seed = item.get("seed")
        if device not in CONFIRMATION_DEVICES:
            raise QualityTeacherStatisticsError(
                f"unknown confirmation device: {device!r}"
            )
        if isinstance(seed, bool) or seed not in CONFIRMATION_SEEDS:
            raise QualityTeacherStatisticsError(f"invalid confirmation seed: {seed!r}")
        file_id = _file_id(item)
        device_rows = indexed[str(device)]
        if int(seed) in device_rows[file_id]:
            raise QualityTeacherStatisticsError(
                "duplicate confirmation device/file/seed observation"
            )
        device_rows[file_id][int(seed)] = _paired_values(
            _metrics(item.get("comparator"), "comparator metrics"),
            _metrics(item.get("candidate"), "candidate metrics"),
        )

    normalized: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for device in CONFIRMATION_DEVICES:
        device_rows = indexed[device]
        if not device_rows:
            raise QualityTeacherStatisticsError(
                f"{device} requires confirmation observations"
            )
        files = tuple(sorted(device_rows))
        for file_id in files:
            if set(device_rows[file_id]) != set(CONFIRMATION_SEEDS):
                raise QualityTeacherStatisticsError(
                    f"{device}/{file_id} requires every frozen seed"
                )
        values = np.empty(
            (len(files), len(CONFIRMATION_SEEDS), len(_SUMMARY_METRICS)),
            dtype=np.float64,
        )
        for file_index, file_id in enumerate(files):
            for seed_index, seed in enumerate(CONFIRMATION_SEEDS):
                metrics = device_rows[file_id][seed]
                values[file_index, seed_index] = [
                    metrics[metric] for metric in _SUMMARY_METRICS
                ]
        normalized[device] = (files, values)
    return normalized


def _device_estimand(values: np.ndarray) -> np.ndarray:
    """Median over seeds within files, then median over files."""
    return np.median(np.median(values, axis=1), axis=0)


def hierarchical_confirmation_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap paired files, then seeds, within equal-weighted device strata."""
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 1
    ):
        raise QualityTeacherStatisticsError(
            "bootstrap replicates must be a positive integer"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise QualityTeacherStatisticsError(
            "bootstrap seed must be a non-negative integer"
        )
    normalized = _normalize_rows(rows)
    metric_indices = {name: index for index, name in enumerate(_SUMMARY_METRICS)}
    per_device_vectors = {
        device: _device_estimand(values) for device, (_, values) in normalized.items()
    }
    aggregate_vector = np.mean(
        np.stack([per_device_vectors[device] for device in CONFIRMATION_DEVICES]),
        axis=0,
    )

    generator = np.random.default_rng(seed)
    distribution = np.empty((replicates, len(_BOOTSTRAP_METRICS)), dtype=np.float64)
    bootstrap_indices = [metric_indices[name] for name in _BOOTSTRAP_METRICS]
    for replicate in range(replicates):
        device_draws = []
        for device in CONFIRMATION_DEVICES:
            _, values = normalized[device]
            file_count, seed_count, _ = values.shape
            sampled_files = generator.integers(0, file_count, size=file_count)
            sampled_seeds = generator.integers(
                0, seed_count, size=(file_count, seed_count)
            )
            sampled = values[
                sampled_files[:, np.newaxis],
                sampled_seeds,
            ][:, :, bootstrap_indices]
            device_draws.append(np.median(np.median(sampled, axis=1), axis=0))
        distribution[replicate] = np.mean(np.stack(device_draws), axis=0)

    intervals = {}
    for index, metric in enumerate(_BOOTSTRAP_METRICS):
        lower, upper = np.quantile(
            distribution[:, index], [0.025, 0.975], method="linear"
        )
        intervals[metric] = {
            "lower_95": float(lower),
            "upper_95": float(upper),
        }

    per_device = {}
    for device in CONFIRMATION_DEVICES:
        files, _ = normalized[device]
        vector = per_device_vectors[device]
        per_device[device] = {
            "file_count": len(files),
            "observation_count": len(files) * len(CONFIRMATION_SEEDS),
            "esr_relative_improvement": float(
                vector[metric_indices["esr_relative_improvement"]]
            ),
            "l1_plus_mrstft_relative_improvement": float(
                vector[metric_indices["l1_plus_mrstft_relative_improvement"]]
            ),
            "secondary_relative_regressions": {
                metric: float(vector[metric_indices[f"{metric}_relative_regression"]])
                for metric in SECONDARY_METRICS
            },
            "correlation": float(vector[metric_indices["correlation"]]),
            "gain_error": float(vector[metric_indices["gain_error"]]),
        }

    aggregate_secondary = {
        metric: float(aggregate_vector[metric_indices[f"{metric}_relative_regression"]])
        for metric in SECONDARY_METRICS
    }
    maximum_secondary = {
        metric: max(
            per_device[device]["secondary_relative_regressions"][metric]
            for device in CONFIRMATION_DEVICES
        )
        for metric in SECONDARY_METRICS
    }
    return {
        "format": "fssr-amp-quality-teacher-bootstrap-v1",
        "method": "paired-hierarchical-quality-teacher-confirmation-v1",
        "replicates": replicates,
        "seed": seed,
        "resampling_order_within_device": ["files", "seeds"],
        "devices_equal_weight": True,
        "devices_fixed_strata": list(CONFIRMATION_DEVICES),
        "seeds": list(CONFIRMATION_SEEDS),
        "observation_unit": "device_file_seed",
        "observation_count": sum(
            details["observation_count"] for details in per_device.values()
        ),
        "per_device": per_device,
        "aggregate_esr_relative_improvement": float(
            aggregate_vector[metric_indices["esr_relative_improvement"]]
        ),
        "aggregate_l1_plus_mrstft_relative_improvement": float(
            aggregate_vector[metric_indices["l1_plus_mrstft_relative_improvement"]]
        ),
        "aggregate_secondary_relative_regressions": aggregate_secondary,
        "maximum_per_device_secondary_relative_regressions": maximum_secondary,
        "minimum_per_device_correlation": min(
            details["correlation"] for details in per_device.values()
        ),
        "minimum_per_device_gain_error": min(
            details["gain_error"] for details in per_device.values()
        ),
        "confidence_intervals_95": intervals,
    }


paired_hierarchical_bootstrap = hierarchical_confirmation_bootstrap


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CONFIRMATION_DEVICES",
    "CONFIRMATION_SEEDS",
    "SECONDARY_METRICS",
    "QualityTeacherStatisticsError",
    "hierarchical_confirmation_bootstrap",
    "l1_plus_mrstft",
    "paired_hierarchical_bootstrap",
]
