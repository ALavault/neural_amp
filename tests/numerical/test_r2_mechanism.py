from __future__ import annotations

import numpy as np
import torch
from torch import nn

from fssr_nam.data.r2_fixtures import apply_r2_fixture
from fssr_nam.data.systems import slow_sag
from fssr_nam.metrics.r2_mechanism import _full_rate_island
from fssr_nam.models.oversampling import FullRateIsland


class _TanhFixtureBranch(nn.Module):
    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(2.8 * signal) / torch.tanh(signal.new_tensor(2.8))

    def stream(self, signal: torch.Tensor) -> torch.Tensor:
        return self(signal)


def test_fixture_rate_scaling_preserves_rf2047_physical_horizon() -> None:
    impulse = np.zeros(9000, dtype=np.float32)
    impulse[0] = 0.25
    result = apply_r2_fixture("rf2047_residual", impulse, 192_000)
    assert result.receptive_field == 8185
    assert result.residual[8184] != 0.0
    assert np.count_nonzero(result.residual[8185:]) == 0


def test_vectorized_slow_sag_matches_frozen_scalar_equations() -> None:
    generator = np.random.default_rng(31)
    signal = generator.normal(0.0, 0.2, 4096).astype(np.float32)
    expected = slow_sag(signal, 48_000)
    actual = apply_r2_fixture("slow_sag", signal, 48_000).output
    np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-7)


def test_fixture_adaa_limit_is_finite_deterministic_near_zero_delta() -> None:
    signal = np.full(8192, 0.125, dtype=np.float32)
    signal[::97] += 5.0e-6
    first = apply_r2_fixture("two_clippers", signal, 48_000, adaa=True).output
    second = apply_r2_fixture("two_clippers", signal, 48_000, adaa=True).output
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_offline_mechanism_island_matches_torch_full_rate_island() -> None:
    generator = np.random.default_rng(41)
    signal = generator.normal(0.0, 0.18, 4096).astype(np.float32)
    expected = (
        FullRateIsland(_TanhFixtureBranch(), factor=2)(torch.from_numpy(signal))
        .detach()
        .numpy()
    )
    actual = _full_rate_island("tanh", signal, 2)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-7)
