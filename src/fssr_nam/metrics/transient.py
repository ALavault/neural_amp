"""Strictly causal envelope and post-silence diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def causal_envelope(
    signal: ArrayLike,
    *,
    sample_rate: int,
    attack_seconds: float = 0.002,
    release_seconds: float = 0.05,
) -> NDArray[np.float64]:
    samples = np.asarray(signal, dtype=np.float64)
    if samples.ndim != 1 or not np.all(np.isfinite(samples)):
        raise ValueError("envelope input must be a finite mono signal")
    if sample_rate <= 0 or attack_seconds <= 0.0 or release_seconds <= 0.0:
        raise ValueError("sample rate and time constants must be positive")
    attack = np.exp(-1.0 / (attack_seconds * sample_rate))
    release = np.exp(-1.0 / (release_seconds * sample_rate))
    output = np.empty_like(samples)
    state = 0.0
    for index, magnitude in enumerate(np.abs(samples)):
        coefficient = attack if magnitude > state else release
        state = coefficient * state + (1.0 - coefficient) * magnitude
        output[index] = state
    return output


def envelope_error(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    sample_rate: int,
    epsilon: float = 1.0e-12,
) -> float:
    prediction_envelope = causal_envelope(prediction, sample_rate=sample_rate)
    target_envelope = causal_envelope(target, sample_rate=sample_rate)
    if prediction_envelope.shape != target_envelope.shape:
        raise ValueError("prediction and target must have identical shapes")
    return float(
        np.mean(np.abs(prediction_envelope - target_envelope))
        / (np.mean(np.abs(target_envelope)) + epsilon)
    )


def post_silence_error(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    silence_start: int,
    epsilon: float = 1.0e-12,
) -> float:
    prediction_samples = np.asarray(prediction, dtype=np.float64)
    target_samples = np.asarray(target, dtype=np.float64)
    if prediction_samples.shape != target_samples.shape or prediction_samples.ndim != 1:
        raise ValueError(
            "prediction and target must be identically shaped mono signals"
        )
    if not 0 <= silence_start < prediction_samples.size:
        raise ValueError("silence_start must index the signals")
    residual = prediction_samples[silence_start:] - target_samples[silence_start:]
    return float(
        np.sum(np.square(residual)) / (np.sum(np.square(target_samples)) + epsilon)
    )
