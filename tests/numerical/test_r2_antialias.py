from __future__ import annotations

import pytest
import torch
from torch import nn

from fssr_nam.models import AAFSSR, FirstOrderADAA, FullRateIsland, HermiteCustomTanh
from fssr_nam.models.r2 import scale_a2_config_for_internal_rate


class _StreamingIdentity(nn.Module):
    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return signal

    def stream(self, signal: torch.Tensor) -> torch.Tensor:
        return signal

    def reset_state(self) -> None:
        pass


def _irregular(model: nn.Module, signal: torch.Tensor) -> torch.Tensor:
    chunks = []
    position = 0
    sizes = (1, 7, 3, 16, 5, 11)
    index = 0
    while position < signal.shape[-1]:
        stop = min(signal.shape[-1], position + sizes[index % len(sizes)])
        chunks.append(model.stream(signal[..., position:stop]))
        position = stop
        index += 1
    return torch.cat(chunks, dim=-1)


def test_hermite_custom_tanh_is_c1_and_matches_alpha_1p8_initialization() -> None:
    activation = HermiteCustomTanh(alpha=1.8)
    inputs = torch.linspace(-3.5, 3.5, 4097, requires_grad=True)
    expected = torch.tanh(inputs / 1.8)
    torch.testing.assert_close(activation(inputs), expected, atol=5.0e-6, rtol=5.0e-6)
    automatic = torch.autograd.grad(activation(inputs).sum(), inputs)[0]
    torch.testing.assert_close(
        activation.derivative(inputs), automatic, atol=2.0e-5, rtol=2.0e-5
    )


def test_adaa_limit_is_finite_near_zero_delta_and_gradients_are_finite() -> None:
    activation = HermiteCustomTanh()
    adaa = FirstOrderADAA(activation, threshold=1.0e-4)
    inputs = torch.tensor(
        [[0.25, 0.25 + 1.0e-8, 0.25 - 5.0e-5, 0.4]], requires_grad=True
    )
    output = adaa(inputs)
    assert torch.isfinite(output).all()
    midpoint = 0.5 * (inputs[:, 1] + inputs[:, 0])
    torch.testing.assert_close(
        output[:, 1], activation(midpoint), atol=1.0e-7, rtol=1.0e-7
    )
    output.square().sum().backward()
    assert torch.isfinite(inputs.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in adaa.parameters()
    )


@pytest.mark.parametrize("factor", [2, 4])
def test_full_rate_filters_have_exact_delay_identity_reset_and_block_parity(
    factor: int,
) -> None:
    torch.manual_seed(20260828 + factor)
    island = FullRateIsland(_StreamingIdentity(), factor=factor)
    signal = torch.randn(2, 193)
    expected = torch.zeros_like(signal)
    expected[:, 16:] = signal[:, :-16]
    torch.testing.assert_close(island(signal), expected, atol=0.0, rtol=0.0)
    island.reset_state()
    streamed = _irregular(island, signal)
    torch.testing.assert_close(streamed, expected, atol=0.0, rtol=0.0)
    island.reset_state()
    repeated = _irregular(island, signal)
    torch.testing.assert_close(repeated, streamed, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("mode", "scale", "latency"),
    [
        ("off", 1, 0),
        ("adaa1", 1, 1),
        ("full_island_x2", 2, 16),
        ("teacher_x4", 4, 16),
    ],
)
def test_r2_fssr_rf_is_physically_exact_and_blocks_match(
    mode: str, scale: int, latency: int
) -> None:
    torch.manual_seed(17)
    model = AAFSSR(core_kind="cascade", aa_mode=mode)
    signal = 0.1 * torch.randn(1, 97)
    expected = model(signal)
    assert expected.shape == signal.shape
    assert torch.isfinite(expected).all()
    assert model.latency_samples == latency
    assert model.internal_sample_rate == 48_000 * scale
    branch = model.processor if scale == 1 else model.processor.branch
    assert (branch.receptive_field - 1) // scale + 1 == 2047
    assert branch.slow.hidden_size == 16
    assert branch.slow.decimation == 64 * scale
    assert branch.residual.channels == 16
    model.reset_state()
    actual = _irregular(model, signal)
    torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=2.0e-6)
    model.reset_state()
    repeated = _irregular(model, signal)
    torch.testing.assert_close(repeated, actual, atol=0.0, rtol=0.0)


def test_r2_fssr_is_causal_and_has_finite_gradients() -> None:
    torch.manual_seed(22)
    model = AAFSSR(core_kind="mono", aa_mode="adaa1")
    signal = (0.1 * torch.randn(1, 80)).requires_grad_()
    reference = model(signal)
    changed = signal.detach().clone()
    changed[:, 60] += 1.0
    actual = model(changed)
    torch.testing.assert_close(actual[:, :60], reference[:, :60], atol=0.0, rtol=0.0)
    reference.square().mean().backward()
    assert torch.isfinite(signal.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_a2_temporal_config_scaling_preserves_source_and_physical_horizon() -> None:
    source = {
        "layers_configs": [
            {
                "dilations": [1, 3, 7],
                "kernel_sizes": [6, 6, 6],
                "head": {"kernel_size": 16},
            }
        ]
    }
    scaled = scale_a2_config_for_internal_rate(source, 2)
    assert source["layers_configs"][0]["dilations"] == [1, 3, 7]
    layer = scaled["layers_configs"][0]
    assert layer["dilations"] == [2, 6, 14]
    assert layer["kernel_sizes"] == [6, 6, 6]
    assert layer["head"]["kernel_size"] == 31
