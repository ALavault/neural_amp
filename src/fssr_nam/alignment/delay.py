"""Delay utilities with a positive-target-lag convention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import correlate, correlation_lags


@dataclass(frozen=True)
class DelayEstimate:
    """Estimated target delay relative to the reference, in samples."""

    integer_samples: int
    fractional_samples: float
    total_samples: float
    normalized_peak: float


def _mono_finite(signal: ArrayLike, name: str) -> NDArray[np.float64]:
    samples = np.asarray(signal, dtype=np.float64)
    if samples.ndim != 1 or samples.size < 3:
        raise ValueError(f"{name} must be a one-dimensional signal of length >= 3")
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"{name} must be finite")
    return samples


def estimate_delay(
    reference: ArrayLike, target: ArrayLike, *, max_lag: int
) -> DelayEstimate:
    """Estimate target lag by cross-correlation and parabolic peak refinement.

    A positive result means that the target occurs later than the reference.
    Polarity inversion does not change the selected lag because peak magnitude
    is used, while the signed normalized peak remains available to the caller.
    """
    reference_samples = _mono_finite(reference, "reference")
    target_samples = _mono_finite(target, "target")
    if reference_samples.shape != target_samples.shape:
        raise ValueError("reference and target must have identical shapes")
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")

    reference_centered = reference_samples - np.mean(reference_samples)
    target_centered = target_samples - np.mean(target_samples)
    correlation = correlate(
        target_centered, reference_centered, mode="full", method="fft"
    )
    lags = correlation_lags(target_samples.size, reference_samples.size, mode="full")
    valid = np.abs(lags) <= max_lag
    valid_indices = np.flatnonzero(valid)
    peak_index = valid_indices[np.argmax(np.abs(correlation[valid]))]
    integer_lag = int(lags[peak_index])

    norm = np.linalg.norm(reference_centered) * np.linalg.norm(target_centered)
    normalized_peak = float(correlation[peak_index] / max(norm, np.finfo(float).eps))
    reference_spectrum = np.fft.rfft(reference_centered)
    target_spectrum = np.fft.rfft(target_centered)
    cross_spectrum = target_spectrum * np.conj(reference_spectrum)
    if normalized_peak < 0.0:
        cross_spectrum *= -1.0
    angular_frequency = 2.0 * np.pi * np.fft.rfftfreq(reference_centered.size)
    integer_compensated = cross_spectrum * np.exp(
        1.0j * angular_frequency * integer_lag
    )
    phase = np.angle(integer_compensated)
    weights = np.abs(cross_spectrum)
    mask = (
        (angular_frequency > 0.0)
        & (angular_frequency < 0.9 * np.pi)
        & (weights > np.quantile(weights, 0.25))
    )
    denominator = np.sum(weights[mask] * np.square(angular_frequency[mask]))
    fractional = 0.0
    if denominator > np.finfo(float).eps:
        fractional = float(
            -np.sum(weights[mask] * angular_frequency[mask] * phase[mask]) / denominator
        )
        fractional = float(np.clip(fractional, -0.5, 0.5))
    return DelayEstimate(
        integer_samples=integer_lag,
        fractional_samples=fractional,
        total_samples=integer_lag + fractional,
        normalized_peak=normalized_peak,
    )


def align_integer_delay(
    reference: ArrayLike, target: ArrayLike, delay_samples: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Crop a pair to compensate a known integer target delay."""
    reference_samples = _mono_finite(reference, "reference")
    target_samples = _mono_finite(target, "target")
    if reference_samples.shape != target_samples.shape:
        raise ValueError("reference and target must have identical shapes")
    if abs(delay_samples) >= reference_samples.size:
        raise ValueError("absolute delay must be shorter than the signals")
    if delay_samples > 0:
        return reference_samples[:-delay_samples], target_samples[delay_samples:]
    if delay_samples < 0:
        return reference_samples[-delay_samples:], target_samples[:delay_samples]
    return reference_samples.copy(), target_samples.copy()


def apply_fractional_delay(
    signal: ArrayLike, delay_samples: float, *, taps: int = 129
) -> NDArray[np.float64]:
    """Apply an offline windowed-sinc delay with zero-defined boundaries.

    The fixed integer group delay of the symmetric FIR is removed. This helper
    is for alignment validation and data preparation, never model inference.
    """
    samples = _mono_finite(signal, "signal")
    if taps < 3 or taps % 2 == 0:
        raise ValueError("taps must be an odd integer of at least three")
    integer_delay = int(np.floor(delay_samples))
    fractional_delay = float(delay_samples - integer_delay)
    center = (taps - 1) // 2
    indices = np.arange(taps, dtype=np.float64) - center
    kernel = np.sinc(indices - fractional_delay) * np.kaiser(taps, 8.6)
    kernel /= np.sum(kernel)
    full = np.convolve(samples, kernel, mode="full")
    compensated = full[center : center + samples.size]

    delayed = np.zeros_like(compensated)
    if integer_delay > 0:
        if integer_delay < samples.size:
            delayed[integer_delay:] = compensated[:-integer_delay]
    elif integer_delay < 0:
        advance = -integer_delay
        if advance < samples.size:
            delayed[:-advance] = compensated[advance:]
    else:
        delayed = compensated
    return delayed
