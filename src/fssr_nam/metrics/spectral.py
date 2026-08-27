"""Spectral metrics with fixed analysis parameters and no realignment."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import stft

DEFAULT_FFT_SIZES = (256, 1_024, 4_096)


def _pair(prediction: ArrayLike, target: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    prediction_array = np.asarray(prediction, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if prediction_array.ndim != 1 or prediction_array.shape != target_array.shape:
        raise ValueError(
            "prediction and target must be identically shaped mono signals"
        )
    if not np.all(np.isfinite(prediction_array)) or not np.all(
        np.isfinite(target_array)
    ):
        raise ValueError("prediction and target must be finite")
    return prediction_array, target_array


def _stft(signal: np.ndarray, fft_size: int) -> NDArray[np.complex128]:
    if signal.size < fft_size:
        raise ValueError("signal must be at least as long as every FFT size")
    _, _, spectrum = stft(
        signal,
        window="hann",
        nperseg=fft_size,
        noverlap=3 * fft_size // 4,
        nfft=fft_size,
        boundary=None,
        padded=False,
    )
    return spectrum


def multi_resolution_stft_error(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    fft_sizes: tuple[int, ...] = DEFAULT_FFT_SIZES,
    epsilon: float = 1.0e-7,
) -> float:
    """Average spectral-convergence plus log-magnitude error."""
    prediction_array, target_array = _pair(prediction, target)
    per_resolution = []
    for fft_size in fft_sizes:
        prediction_magnitude = np.abs(_stft(prediction_array, fft_size))
        target_magnitude = np.abs(_stft(target_array, fft_size))
        convergence = np.linalg.norm(prediction_magnitude - target_magnitude) / (
            np.linalg.norm(target_magnitude) + epsilon
        )
        log_error = np.mean(
            np.abs(
                np.log(prediction_magnitude + epsilon)
                - np.log(target_magnitude + epsilon)
            )
        )
        per_resolution.append(float(convergence + log_error))
    return float(np.mean(per_resolution))


def magnitude_error(
    prediction: ArrayLike, target: ArrayLike, *, fft_size: int = 4_096
) -> float:
    prediction_array, target_array = _pair(prediction, target)
    prediction_magnitude = np.abs(_stft(prediction_array, fft_size))
    target_magnitude = np.abs(_stft(target_array, fft_size))
    return float(
        np.linalg.norm(prediction_magnitude - target_magnitude)
        / (np.linalg.norm(target_magnitude) + 1.0e-12)
    )


def log_spectral_distance(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    fft_size: int = 4_096,
    floor_db: float = -100.0,
) -> float:
    prediction_array, target_array = _pair(prediction, target)
    prediction_db = np.maximum(
        20.0 * np.log10(np.abs(_stft(prediction_array, fft_size)) + 1.0e-12),
        floor_db,
    )
    target_db = np.maximum(
        20.0 * np.log10(np.abs(_stft(target_array, fft_size)) + 1.0e-12),
        floor_db,
    )
    return float(
        np.mean(np.sqrt(np.mean(np.square(prediction_db - target_db), axis=0)))
    )


def phase_error(
    prediction: ArrayLike, target: ArrayLike, *, fft_size: int = 4_096
) -> float:
    """Return target-magnitude-weighted wrapped phase error in radians."""
    prediction_array, target_array = _pair(prediction, target)
    prediction_spectrum = _stft(prediction_array, fft_size)
    target_spectrum = _stft(target_array, fft_size)
    wrapped = np.angle(prediction_spectrum * np.conj(target_spectrum))
    weights = np.abs(target_spectrum)
    return float(np.sum(weights * np.abs(wrapped)) / (np.sum(weights) + 1.0e-12))


def spectral_metrics(prediction: ArrayLike, target: ArrayLike) -> dict[str, float]:
    return {
        "mrstft": multi_resolution_stft_error(prediction, target),
        "magnitude_error": magnitude_error(prediction, target),
        "log_spectral_distance_db": log_spectral_distance(prediction, target),
        "phase_error_radians": phase_error(prediction, target),
    }
