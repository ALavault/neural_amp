from __future__ import annotations

import numpy as np
import pytest

from fssr_nam.alignment.delay import (
    align_integer_delay,
    apply_fractional_delay,
    estimate_delay,
)


def _noise() -> np.ndarray:
    return np.random.default_rng(4).standard_normal(8_192)


def test_integer_delay_estimation_and_alignment() -> None:
    reference = _noise()
    target = np.pad(reference, (7, 0))[: reference.size]
    estimate = estimate_delay(reference, target, max_lag=32)
    assert estimate.integer_samples == 7
    aligned_reference, aligned_target = align_integer_delay(reference, target, 7)
    np.testing.assert_array_equal(aligned_reference, aligned_target)


def test_fractional_delay_estimation() -> None:
    reference = _noise()
    target = apply_fractional_delay(reference, 3.25)
    estimate = estimate_delay(reference[256:-256], target[256:-256], max_lag=16)
    assert estimate.total_samples == pytest.approx(3.25, abs=0.12)


def test_delay_estimation_preserves_polarity_information() -> None:
    reference = _noise()
    target = -np.pad(reference, (5, 0))[: reference.size]
    estimate = estimate_delay(reference, target, max_lag=16)
    assert estimate.integer_samples == 5
    assert estimate.normalized_peak < 0.0
