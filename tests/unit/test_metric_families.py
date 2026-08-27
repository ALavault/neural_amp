from __future__ import annotations

import numpy as np
import pytest

from fssr_nam.metrics.nonlinear import (
    complex_harmonic_error,
    inharmonic_energy_ratio,
    known_reference_parasite_db,
)
from fssr_nam.metrics.spectral import spectral_metrics
from fssr_nam.metrics.time import time_metrics
from fssr_nam.metrics.transient import envelope_error

SAMPLE_RATE = 48_000
DURATION = 1.0


def _sine(frequency: float) -> np.ndarray:
    time = np.arange(round(SAMPLE_RATE * DURATION)) / SAMPLE_RATE
    return np.sin(2.0 * np.pi * frequency * time)


def test_time_metrics_separate_gain_dc_and_polarity() -> None:
    target = _sine(1_000.0)
    gain = time_metrics(0.5 * target, target)
    dc = time_metrics(target + 0.1, target)
    polarity = time_metrics(-target, target)
    assert gain["gain_error"] == pytest.approx(-0.5)
    assert abs(gain["dc_error"]) < 1.0e-12
    assert dc["dc_error"] == pytest.approx(0.1)
    assert polarity["correlation"] == pytest.approx(-1.0)


def test_spectral_metrics_detect_phase_without_magnitude_change() -> None:
    target = _sine(1_000.0)
    quadrature = np.cos(2.0 * np.pi * 1_000.0 * np.arange(target.size) / SAMPLE_RATE)
    metrics = spectral_metrics(quadrature, target)
    assert metrics["phase_error_radians"] > 1.0
    assert metrics["magnitude_error"] < 0.02


def test_complex_harmonic_error_detects_removed_harmonic() -> None:
    fundamental = _sine(1_000.0)
    target = fundamental + 0.2 * _sine(3_000.0)
    prediction = fundamental
    assert (
        complex_harmonic_error(
            prediction,
            target,
            fundamental_hz=1_000.0,
            sample_rate=SAMPLE_RATE,
        )
        > 0.03
    )


def test_inharmonic_and_known_reference_metrics_detect_added_tone() -> None:
    target = _sine(1_000.0)
    prediction = target + 0.1 * _sine(1_337.0)
    ratio = inharmonic_energy_ratio(
        prediction,
        permitted_frequencies_hz=[1_000.0],
        sample_rate=SAMPLE_RATE,
    )
    assert ratio > 0.009
    assert known_reference_parasite_db(prediction, target) == pytest.approx(
        -20.0, abs=0.1
    )


def test_envelope_metric_detects_level_error() -> None:
    target = _sine(220.0)
    assert envelope_error(
        0.5 * target, target, sample_rate=SAMPLE_RATE
    ) == pytest.approx(0.5, rel=0.01)
