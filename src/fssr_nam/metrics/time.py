"""Time-domain metrics with explicit numerical conventions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _pair(prediction: ArrayLike, target: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    prediction_array = np.asarray(prediction, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if prediction_array.shape != target_array.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction_array.ndim != 1:
        raise ValueError("metrics require one-dimensional mono signals")
    if not np.all(np.isfinite(prediction_array)) or not np.all(
        np.isfinite(target_array)
    ):
        raise ValueError("prediction and target must be finite")
    return prediction_array, target_array


def error_to_signal_ratio(
    prediction: ArrayLike, target: ArrayLike, *, epsilon: float = 1.0e-12
) -> float:
    """Return sum squared error divided by target energy.

    No alignment, gain correction, or DC correction is applied implicitly.
    """
    prediction_array, target_array = _pair(prediction, target)
    numerator = np.sum(np.square(prediction_array - target_array), dtype=np.float64)
    denominator = np.sum(np.square(target_array), dtype=np.float64)
    return float(numerator / (denominator + epsilon))


def normalized_mean_squared_error(
    prediction: ArrayLike, target: ArrayLike, *, epsilon: float = 1.0e-12
) -> float:
    """Return MSE normalized by target variance, without implicit correction."""
    prediction_array, target_array = _pair(prediction, target)
    return float(
        np.mean(np.square(prediction_array - target_array))
        / (np.var(target_array) + epsilon)
    )


def mean_absolute_error(prediction: ArrayLike, target: ArrayLike) -> float:
    prediction_array, target_array = _pair(prediction, target)
    return float(np.mean(np.abs(prediction_array - target_array)))


def gain_error(prediction: ArrayLike, target: ArrayLike) -> float:
    """Return least-squares signed gain minus one."""
    prediction_array, target_array = _pair(prediction, target)
    denominator = float(np.dot(target_array, target_array))
    if denominator <= np.finfo(float).eps:
        raise ValueError("gain error is undefined for a zero-energy target")
    return float(np.dot(prediction_array, target_array) / denominator - 1.0)


def dc_error(prediction: ArrayLike, target: ArrayLike) -> float:
    prediction_array, target_array = _pair(prediction, target)
    return float(np.mean(prediction_array - target_array))


def correlation(prediction: ArrayLike, target: ArrayLike) -> float:
    prediction_array, target_array = _pair(prediction, target)
    prediction_centered = prediction_array - np.mean(prediction_array)
    target_centered = target_array - np.mean(target_array)
    denominator = np.linalg.norm(prediction_centered) * np.linalg.norm(target_centered)
    if denominator <= np.finfo(float).eps:
        raise ValueError("correlation is undefined for a constant signal")
    return float(np.dot(prediction_centered, target_centered) / denominator)


def time_metrics(prediction: ArrayLike, target: ArrayLike) -> dict[str, float]:
    """Compute the frozen M1 time-domain metric family."""
    return {
        "esr": error_to_signal_ratio(prediction, target),
        "nmse": normalized_mean_squared_error(prediction, target),
        "mae": mean_absolute_error(prediction, target),
        "gain_error": gain_error(prediction, target),
        "dc_error": dc_error(prediction, target),
        "correlation": correlation(prediction, target),
    }
