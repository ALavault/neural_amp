import numpy as np
import pytest

from fssr_nam.metrics.spectral import log_mel_error


def test_log_mel_is_zero_for_identity_and_detects_spectral_change() -> None:
    index = np.arange(8192, dtype=np.float64)
    target = np.sin(2.0 * np.pi * 440.0 * index / 48_000.0)
    assert log_mel_error(target, target) == 0.0
    prediction = target + 0.1 * np.sin(2.0 * np.pi * 4000.0 * index / 48_000.0)
    assert log_mel_error(prediction, target) > 0.0


def test_log_mel_is_symmetric_finite_and_rejects_invalid_pairs() -> None:
    generator = np.random.default_rng(20260830)
    first = generator.normal(size=4096)
    second = generator.normal(size=4096)
    assert log_mel_error(first, second) == pytest.approx(log_mel_error(second, first))
    with pytest.raises(ValueError, match="identically shaped"):
        log_mel_error(first[:-1], second)
    with pytest.raises(ValueError, match="finite"):
        first[0] = np.nan
        log_mel_error(first, second)
