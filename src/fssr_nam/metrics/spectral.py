"""Spectral metrics with fixed analysis parameters and no realignment."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import stft

DEFAULT_FFT_SIZES = (256, 1_024, 4_096)
LOG_MEL_SAMPLE_RATE = 48_000
LOG_MEL_FFT_SIZE = 2_048
LOG_MEL_HOP_SIZE = 512
LOG_MEL_BANDS = 128
LOG_MEL_MIN_HZ = 20.0
LOG_MEL_MAX_HZ = 24_000.0


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


def _mel_filterbank() -> NDArray[np.float64]:
    def hz_to_mel(frequency: np.ndarray | float) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_edges = np.linspace(
        hz_to_mel(LOG_MEL_MIN_HZ),
        hz_to_mel(LOG_MEL_MAX_HZ),
        LOG_MEL_BANDS + 2,
    )
    hz_edges = mel_to_hz(mel_edges)
    frequencies = np.fft.rfftfreq(LOG_MEL_FFT_SIZE, 1.0 / LOG_MEL_SAMPLE_RATE)
    filters = np.zeros((LOG_MEL_BANDS, frequencies.size), dtype=np.float64)
    for band in range(LOG_MEL_BANDS):
        left, center, right = hz_edges[band : band + 3]
        filters[band] = np.maximum(
            0.0,
            np.minimum(
                (frequencies - left) / (center - left),
                (right - frequencies) / (right - center),
            ),
        )
        filters[band] /= max(float(np.sum(filters[band])), np.finfo(float).eps)
    return filters


def log_mel_error(
    prediction: ArrayLike, target: ArrayLike, *, epsilon: float = 1.0e-10
) -> float:
    """Mean absolute natural-log Mel-power error on the frozen 48 kHz grid."""
    prediction_array, target_array = _pair(prediction, target)
    if epsilon <= 0.0:
        raise ValueError("log-Mel epsilon must be positive")

    def transform(signal: np.ndarray) -> NDArray[np.float64]:
        _, _, spectrum = stft(
            signal,
            fs=LOG_MEL_SAMPLE_RATE,
            window="hann",
            nperseg=LOG_MEL_FFT_SIZE,
            noverlap=LOG_MEL_FFT_SIZE - LOG_MEL_HOP_SIZE,
            nfft=LOG_MEL_FFT_SIZE,
            boundary=None,
            padded=False,
        )
        power = np.abs(spectrum) ** 2
        return np.log(_mel_filterbank() @ power + epsilon)

    return float(np.mean(np.abs(transform(prediction_array) - transform(target_array))))


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
        "log_mel": log_mel_error(prediction, target),
        "mrstft": multi_resolution_stft_error(prediction, target),
        "magnitude_error": magnitude_error(prediction, target),
        "log_spectral_distance_db": log_spectral_distance(prediction, target),
        "phase_error_radians": phase_error(prediction, target),
    }
