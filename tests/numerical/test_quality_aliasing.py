from __future__ import annotations

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from fssr_nam.metrics.quality_aliasing import (
    AMPLITUDES,
    DFT_SAMPLES,
    K0_VALUES,
    calibrate_identity_floor,
    coherent_sine_probe,
    dc_separated_asr,
    harmonic_fidelity_guard,
)


@settings(max_examples=20, deadline=None)
@example(k0=K0_VALUES[0], amplitude=AMPLITUDES[0], dc=0.0)
@example(k0=K0_VALUES[-1], amplitude=AMPLITUDES[-1], dc=0.45)
@given(
    k0=st.sampled_from(K0_VALUES),
    amplitude=st.sampled_from(AMPLITUDES),
    dc=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
)
def test_dc_is_separate_and_cannot_turn_identity_into_alias(
    k0: int, amplitude: float, dc: float
) -> None:
    floor = -120.0
    probe = coherent_sine_probe(k0, amplitude)
    baseline = dc_separated_asr(probe, k0=k0, floor_db=floor)
    shifted = dc_separated_asr(probe.astype(np.float64) + dc, k0=k0, floor_db=floor)
    assert baseline["floor_censored"] is True
    assert shifted["floor_censored"] is True
    assert shifted["asr_db"] == floor
    assert shifted["dc_energy"] >= 0.0


def test_exact_zero_alias_is_valid_and_censored_not_rejected() -> None:
    k0 = K0_VALUES[0]
    spectrum = np.zeros(DFT_SAMPLES // 2 + 1, dtype=np.complex128)
    spectrum[k0] = -1j * DFT_SAMPLES * 0.1 / 2.0
    frame = np.fft.irfft(spectrum, n=DFT_SAMPLES)
    result = dc_separated_asr(np.tile(frame, 6), k0=k0, floor_db=-130.0)
    assert result["asr_linear"] >= 0.0
    assert result["floor_censored"] is True
    assert result["asr_db"] == -130.0


def test_explicit_alias_bin_increases_asr_without_touching_dc() -> None:
    k0 = K0_VALUES[0]
    clean = coherent_sine_probe(k0, AMPLITUDES[1]).astype(np.float64)
    index = np.arange(DFT_SAMPLES, dtype=np.float64)
    parasite = 0.01 * np.sin(2.0 * np.pi * 211 * index / DFT_SAMPLES)
    aliased = clean + np.tile(parasite, 6)
    clean_result = dc_separated_asr(clean, k0=k0, floor_db=-140.0)
    alias_result = dc_separated_asr(aliased, k0=k0, floor_db=-140.0)
    assert alias_result["asr_db"] > clean_result["asr_db"] + 30.0
    assert alias_result["dc_to_harmonic_linear"] == pytest.approx(0.0, abs=1.0e-20)


def test_absolute_relative_guard_is_stable_at_zero_baseline_error() -> None:
    signal = coherent_sine_probe(K0_VALUES[0], AMPLITUDES[1])
    within_floor = harmonic_fidelity_guard(
        1.0001 * signal, signal, signal, k0=K0_VALUES[0]
    )
    assert within_floor["passed"] is True
    assert within_floor["values"]["relative_check_applicable"] is False
    outside_floor = harmonic_fidelity_guard(
        1.01 * signal, signal, signal, k0=K0_VALUES[0]
    )
    assert outside_floor["passed"] is False
    assert outside_floor["checks"]["complex_absolute"] is False


def test_identity_floor_is_candidate_independent_and_below_cap() -> None:
    calibration = calibrate_identity_floor()
    assert calibration["candidate_outputs_observed"] is False
    assert calibration["passed"] is True
    assert calibration["locked_floor_db"] <= -120.0
    assert len(calibration["conditions"]) == 9
