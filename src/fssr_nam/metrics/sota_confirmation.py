"""Frozen source-level metrics for SOTA-PROTOTYPE-v1.1 confirmation."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .spectral import log_mel_error, multi_resolution_stft_error
from .time import error_to_signal_ratio, mean_absolute_error

COMMON_PREROLL_SAMPLES = 14_400


def aligned_source_metrics(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    latency_samples: int,
    preroll_samples: int = COMMON_PREROLL_SAMPLES,
) -> dict[str, Any]:
    """Score one source after declared latency only; no fitted correction."""
    predicted = np.asarray(prediction, dtype=np.float32)
    reference = np.asarray(target, dtype=np.float32)
    if predicted.ndim != 1 or reference.ndim != 1:
        raise ValueError("confirmation metrics require mono source arrays")
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError("confirmation source arrays must be finite")
    if latency_samples < 0 or preroll_samples < 0:
        raise ValueError("latency and preroll must be non-negative")
    samples = min(predicted.size - latency_samples, reference.size)
    if samples <= preroll_samples + 4096:
        raise ValueError("source is too short after latency and preroll")
    scored_prediction = predicted[
        latency_samples + preroll_samples : latency_samples + samples
    ]
    scored_target = reference[preroll_samples:samples]
    return {
        "esr": error_to_signal_ratio(scored_prediction, scored_target),
        "mae": mean_absolute_error(scored_prediction, scored_target),
        "log_mel": log_mel_error(scored_prediction, scored_target),
        "mrstft": multi_resolution_stft_error(scored_prediction, scored_target),
        "latency_samples": latency_samples,
        "preroll_samples": preroll_samples,
        "scored_samples": len(scored_target),
        "gain_correction_applied": False,
        "delay_fit_applied": False,
        "dc_correction_applied": False,
    }
