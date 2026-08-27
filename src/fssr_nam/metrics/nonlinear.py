"""Harmonic, intermodulation, and known-reference parasite diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _mono(signal: ArrayLike) -> NDArray[np.float64]:
    samples = np.asarray(signal, dtype=np.float64)
    if samples.ndim != 1 or not np.all(np.isfinite(samples)):
        raise ValueError("metric input must be a finite mono signal")
    return samples


def complex_coefficients(
    signal: ArrayLike, frequencies: ArrayLike, sample_rate: int
) -> NDArray[np.complex128]:
    """Project a signal onto exact positive-frequency complex sinusoids."""
    samples = _mono(signal)
    frequency_array = np.asarray(frequencies, dtype=np.float64)
    if frequency_array.ndim != 1:
        raise ValueError("frequencies must be one-dimensional")
    if np.any(frequency_array <= 0.0) or np.any(frequency_array >= sample_rate / 2.0):
        raise ValueError("frequencies must lie strictly inside the Nyquist interval")
    indices = np.arange(samples.size, dtype=np.float64)
    basis = np.exp(
        -2.0j * np.pi * frequency_array[:, None] * indices[None, :] / sample_rate
    )
    return np.asarray((2.0 / samples.size) * (basis @ samples), dtype=np.complex128)


def complex_harmonic_error(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    fundamental_hz: float,
    sample_rate: int,
    maximum_harmonic: int = 12,
    epsilon: float = 1.0e-12,
) -> float:
    frequencies = fundamental_hz * np.arange(1, maximum_harmonic + 1)
    frequencies = frequencies[frequencies < sample_rate / 2.0]
    prediction_coefficients = complex_coefficients(prediction, frequencies, sample_rate)
    target_coefficients = complex_coefficients(target, frequencies, sample_rate)
    return float(
        np.sum(np.square(np.abs(prediction_coefficients - target_coefficients)))
        / (np.sum(np.square(np.abs(target_coefficients))) + epsilon)
    )


def complex_intermodulation_error(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    frequencies_hz: ArrayLike,
    sample_rate: int,
    epsilon: float = 1.0e-12,
) -> float:
    prediction_coefficients = complex_coefficients(
        prediction, frequencies_hz, sample_rate
    )
    target_coefficients = complex_coefficients(target, frequencies_hz, sample_rate)
    return float(
        np.sum(np.square(np.abs(prediction_coefficients - target_coefficients)))
        / (np.sum(np.square(np.abs(target_coefficients))) + epsilon)
    )


def inharmonic_energy_ratio(
    signal: ArrayLike,
    *,
    permitted_frequencies_hz: ArrayLike,
    sample_rate: int,
    tolerance_hz: float = 5.0,
) -> float:
    """Return FFT energy outside explicitly permitted narrow frequency bands."""
    samples = _mono(signal)
    spectrum = np.fft.rfft(samples)
    frequencies = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
    permitted = np.zeros(frequencies.size, dtype=bool)
    for frequency in np.asarray(permitted_frequencies_hz, dtype=float):
        permitted |= np.abs(frequencies - frequency) <= tolerance_hz
    energy = np.square(np.abs(spectrum))
    return float(np.sum(energy[~permitted]) / (np.sum(energy) + 1.0e-12))


def known_reference_parasite_db(
    prediction: ArrayLike, reference: ArrayLike, *, epsilon: float = 1.0e-18
) -> float:
    """Return residual spectral energy relative to a known clean reference.

    The result is only a parasite/alias diagnostic when the reference-generating
    process isolates that cause; it is not a general alias detector.
    """
    prediction_samples = _mono(prediction)
    reference_samples = _mono(reference)
    if prediction_samples.shape != reference_samples.shape:
        raise ValueError("prediction and reference must have identical shapes")
    residual_spectrum = np.fft.rfft(prediction_samples - reference_samples)
    reference_spectrum = np.fft.rfft(reference_samples)
    ratio = np.sum(np.square(np.abs(residual_spectrum))) / (
        np.sum(np.square(np.abs(reference_spectrum))) + epsilon
    )
    return float(10.0 * np.log10(ratio + epsilon))
