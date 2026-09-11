from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.signal import freqz
from torch import nn

from fssr_nam.data.r2_fixtures import apply_r2_fixture
from fssr_nam.metrics.quality_aa_mechanism import reference_pair
from fssr_nam.metrics.quality_aliasing import (
    coherent_sine_probe,
    fundamental_delay_samples,
    harmonic_fidelity_guard,
)
from fssr_nam.models import FullRateIsland
from fssr_nam.models.oversampling import design_resampling_lowpass


class _Identity(nn.Module):
    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return signal

    def stream(self, signal: torch.Tensor) -> torch.Tensor:
        return signal

    def reset_state(self) -> None:
        pass


@pytest.mark.parametrize("factor", [2, 4])
def test_latency32_filter_meets_preregistered_response(factor: int) -> None:
    coefficients = design_resampling_lowpass(factor, factor * 32 + 1).numpy()
    frequency, response = freqz(coefficients, worN=262_144, fs=48_000 * factor)

    def magnitude_db(at_hz: float) -> float:
        index = int(np.argmin(np.abs(frequency - at_hz)))
        return float(20.0 * np.log10(max(abs(response[index]), 1.0e-20)))

    assert abs(magnitude_db(20_000.0)) <= 0.01
    assert magnitude_db(30_000.0) <= -80.0


@settings(max_examples=8, deadline=None)
@given(
    factor=st.sampled_from((2, 4)),
    block_sizes=st.lists(
        st.integers(min_value=1, max_value=67), min_size=1, max_size=12
    ),
)
def test_latency32_identity_has_exact_delay_for_arbitrary_blocks(
    factor: int, block_sizes: list[int]
) -> None:
    length = sum(block_sizes)
    signal = torch.linspace(-0.7, 0.8, length).reshape(1, -1)
    model = FullRateIsland(_Identity(), factor=factor, latency_samples=32)
    expected = torch.zeros_like(signal)
    if length > 32:
        expected[:, 32:] = signal[:, :-32]
    torch.testing.assert_close(model(signal), expected, atol=0.0, rtol=0.0)
    model.reset_state()
    chunks = []
    position = 0
    for size in block_sizes:
        chunks.append(model.stream(signal[:, position : position + size]))
        position += size
    torch.testing.assert_close(torch.cat(chunks, -1), expected, atol=0.0, rtol=0.0)


def test_adaa_probe_exposes_half_sample_not_integer_delay() -> None:
    delays = []
    for k0 in (1705, 8191, 12287):
        signal = coherent_sine_probe(k0, 0.25)
        plain = apply_r2_fixture("tanh", signal, 48_000).output
        adaa = apply_r2_fixture("tanh", signal, 48_000, adaa=True).output
        delays.append(fundamental_delay_samples(adaa, plain, k0=k0))
    np.testing.assert_allclose(delays, 0.5, atol=2.0e-6, rtol=0.0)
    assert max(abs(value - round(value)) for value in delays) > 0.49


def test_direct_x8_x16_reference_converges_on_smooth_fixture() -> None:
    reference_x8, reference_x16 = reference_pair("tanh", 1705, 0.25)
    guard = harmonic_fidelity_guard(reference_x8, reference_x16, reference_x16, k0=1705)
    assert guard["values"]["correlation"] >= 0.999
    assert guard["values"]["candidate_complex_harmonic_error"] <= 1.0e-5
    assert guard["values"]["dc_complex_error"] <= 1.0e-5
