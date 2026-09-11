from __future__ import annotations

import numpy as np
import pytest

from fssr_nam.metrics.r2_aliasing import (
    AMPLITUDES,
    DFT_SAMPLES,
    K0_VALUES,
    ASREvidenceError,
    aggregate_asr_grid,
    anti_silence_guard,
    coherent_sine_probe,
    sato_smith_asr,
)


def test_sato_smith_asr_detects_nonharmonic_energy_without_window_or_padding() -> None:
    k0 = K0_VALUES[0]
    clean = coherent_sine_probe(k0, AMPLITUDES[0])
    clean_result = sato_smith_asr(clean, k0=k0)
    frame_index = np.arange(DFT_SAMPLES, dtype=np.float64)
    alias_frame = 0.01 * np.sin(2.0 * np.pi * 211 * frame_index / DFT_SAMPLES)
    aliased = clean.astype(np.float64) + np.tile(alias_frame, 6)
    aliased_result = sato_smith_asr(aliased, k0=k0)
    assert aliased_result["asr_db"] > clean_result["asr_db"] + 40.0
    assert aliased_result["window"] == "none"
    assert aliased_result["zero_padding"] is False
    assert aliased_result["periodicity_passed"] is True


def test_sato_smith_asr_rejects_silence_and_aperiodic_last_frame() -> None:
    with pytest.raises(ASREvidenceError, match="nonconstant"):
        sato_smith_asr(np.zeros(6 * DFT_SAMPLES), k0=K0_VALUES[0])
    signal = coherent_sine_probe(K0_VALUES[0], AMPLITUDES[0])
    signal[-1] += 0.1
    with pytest.raises(ASREvidenceError, match="periodicity failed"):
        sato_smith_asr(signal, k0=K0_VALUES[0])


def test_anti_silence_guard_passes_harmonics_and_rejects_attenuation() -> None:
    signal = coherent_sine_probe(K0_VALUES[0], AMPLITUDES[1])
    passed = anti_silence_guard(signal, signal, signal, k0=K0_VALUES[0])
    assert passed["passed"] is True
    attenuated = 0.79 * signal
    failed = anti_silence_guard(attenuated, signal, signal, k0=K0_VALUES[0])
    assert failed["passed"] is False
    assert failed["checks"]["gain_error"] is False
    assert failed["checks"]["fundamental_level"] is False


def test_asr_grid_requires_all_nine_unique_conditions() -> None:
    rows = [
        {"k0": k0, "amplitude": amplitude, "asr_db": -20.0}
        for k0 in K0_VALUES
        for amplitude in AMPLITUDES
    ]
    summary = aggregate_asr_grid(rows)
    assert summary == {
        "conditions": 9,
        "median_asr_db": -20.0,
        "minimum_asr_db": -20.0,
        "maximum_asr_db": -20.0,
    }
    with pytest.raises(ASREvidenceError, match="grid mismatch"):
        aggregate_asr_grid(rows[:-1])
