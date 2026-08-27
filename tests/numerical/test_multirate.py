from __future__ import annotations

import numpy as np
import pytest

from fssr_nam.dsp.multirate import (
    DecimationConfig,
    controlled_decimate,
    derive_reference_rates,
)


def _tone(frequency: float, sample_rate: int, duration: float = 0.1) -> np.ndarray:
    time = np.arange(round(sample_rate * duration)) / sample_rate
    return np.sin(2.0 * np.pi * frequency * time).astype(np.float32)


def _interior_rms(signal: np.ndarray, trim: int = 200) -> float:
    interior = signal[trim:-trim]
    return float(np.sqrt(np.mean(np.square(interior, dtype=np.float64))))


def test_decimation_preserves_passband_tone() -> None:
    output = controlled_decimate(_tone(1_000.0, 96_000))
    assert _interior_rms(output) == pytest.approx(1.0 / np.sqrt(2.0), rel=0.01)


def test_decimation_rejects_tone_above_target_nyquist() -> None:
    output = controlled_decimate(_tone(30_000.0, 96_000))
    assert _interior_rms(output) < 1.0e-3


def test_reference_rate_lengths() -> None:
    rates = derive_reference_rates(np.ones(19_200, dtype=np.float32))
    assert rates[192_000].shape == (19_200,)
    assert rates[96_000].shape == (9_600,)
    assert rates[48_000].shape == (4_800,)


def test_decimation_config_rejects_even_filter_length() -> None:
    with pytest.raises(ValueError, match="odd"):
        controlled_decimate(np.ones(32), DecimationConfig(taps=32))
