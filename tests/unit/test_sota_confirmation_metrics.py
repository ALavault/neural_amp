import numpy as np
import pytest

from fssr_nam.metrics.sota_confirmation import aligned_source_metrics


def test_source_metrics_use_only_declared_latency_and_common_preroll() -> None:
    generator = np.random.default_rng(20260830)
    target = generator.normal(0.0, 0.2, 24_000).astype(np.float32)
    prediction = np.pad(target, (32, 0))[:-32]
    metrics = aligned_source_metrics(prediction, target, latency_samples=32)
    assert metrics["esr"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["log_mel"] == 0.0
    assert metrics["mrstft"] == 0.0
    assert metrics["gain_correction_applied"] is False
    assert metrics["delay_fit_applied"] is False


def test_source_metrics_fail_closed_on_short_or_nonfinite_audio() -> None:
    with pytest.raises(ValueError, match="too short"):
        aligned_source_metrics(np.zeros(10_000), np.zeros(10_000), latency_samples=0)
    invalid = np.zeros(20_000)
    invalid[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        aligned_source_metrics(invalid, np.zeros_like(invalid), latency_samples=0)
