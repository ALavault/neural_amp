from __future__ import annotations

import math
import sys
import types
from importlib import import_module
from pathlib import Path

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from fssr_nam.models.sota_comparators import (
    S4_VARIANTS,
    TCN_VARIANTS,
    DiagonalS4,
    NablafxS4TFiLM,
    NablafxTCNTFiLM,
    TemporalFiLM,
    build_nablafx_comparator,
    build_sota_comparator,
)


@given(samples=st.integers(min_value=1, max_value=385))
@settings(max_examples=12, deadline=None)
def test_tfilm_has_finite_shape_and_gradients_for_partial_blocks(samples: int) -> None:
    torch.manual_seed(4)
    module = TemporalFiLM(3, block_size=128)
    signal = torch.randn(2, 3, samples, requires_grad=True)
    output = module(signal)
    assert output.shape == signal.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert signal.grad is not None and torch.isfinite(signal.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_diagonal_s4_fft_matches_exact_recurrence() -> None:
    torch.manual_seed(8)
    module = DiagonalS4(channels=3, state_dim=5).double()
    signal = torch.randn(2, 3, 79, dtype=torch.double)
    fft_output = module(signal)
    recurrent_output = module.recurrent(signal)
    torch.testing.assert_close(fft_output, recurrent_output, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize(
    ("family", "variant", "expected_parameters", "expected_latency"),
    [
        ("nablafx_tcn_tfilm", "small", 45_697, 635),
        ("nablafx_tcn_tfilm", "large", 75_937, 1_270),
        ("nablafx_s4_tfilm", "small", 27_953, 508),
        ("nablafx_s4_tfilm", "large", 70_193, 1_016),
    ],
)
def test_frozen_nablafx_topologies_and_lookahead_are_literal(
    family: str, variant: str, expected_parameters: int, expected_latency: int
) -> None:
    model = build_nablafx_comparator(family, variant)
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        expected_parameters
    )
    assert model.latency_samples == expected_latency


@pytest.mark.parametrize(
    "model",
    [NablafxTCNTFiLM("small"), NablafxS4TFiLM("small")],
)
def test_delayed_comparators_are_causal_resettable_and_block_exact(
    model: torch.nn.Module,
) -> None:
    torch.manual_seed(12)
    samples = model.latency_samples + 193
    signal = torch.randn(samples)
    split = model.latency_samples + 67
    changed = signal.clone()
    changed[split:] = torch.randn_like(changed[split:])
    with torch.inference_mode():
        expected = model(signal)
        alternative = model(changed)
    torch.testing.assert_close(
        expected[:split], alternative[:split], rtol=2e-6, atol=1e-7
    )

    model.reset_state()
    parts = (1, 63, 129, samples - 193)
    assert sum(parts) == samples
    streamed: list[torch.Tensor] = []
    start = 0
    with torch.inference_mode():
        for length in parts:
            streamed.append(model.stream(signal[start : start + length]))
            start += length
    torch.testing.assert_close(torch.cat(streamed), expected, rtol=2e-5, atol=2e-6)
    model.reset_state()
    with torch.inference_mode():
        reset_output = model.stream(signal)
    torch.testing.assert_close(reset_output, expected, rtol=2e-5, atol=2e-6)


def test_frozen_variant_constants_match_primary_configs() -> None:
    assert TCN_VARIANTS["small"].blocks == 5
    assert TCN_VARIANTS["small"].kernel_size == 13
    assert TCN_VARIANTS["small"].dilation_growth == 10
    assert TCN_VARIANTS["large"].blocks == 10
    assert TCN_VARIANTS["large"].kernel_size == 5
    assert TCN_VARIANTS["large"].dilation_growth == 3
    assert S4_VARIANTS["small"].blocks == 4
    assert S4_VARIANTS["small"].state_dim == 4
    assert S4_VARIANTS["large"].blocks == 8
    assert S4_VARIANTS["large"].state_dim == 32
    assert math.prod((TCN_VARIANTS["small"].channels, 1)) == 16


def test_official_nam_and_wright_comparator_factories_are_literal() -> None:
    root = Path(__file__).resolve().parents[2]
    nam = build_sota_comparator("nam_a2_full", variant="official", root=root)
    nam_lite = build_sota_comparator("nam_a2_lite", variant="official", root=root)
    wright = build_sota_comparator("wright_lstm64", variant="official")
    assert sum(parameter.numel() for parameter in nam.parameters()) == 12_145
    assert sum(parameter.numel() for parameter in nam_lite.parameters()) < 12_145
    assert nam_lite.receptive_field == nam.receptive_field == 6_347
    assert sum(parameter.numel() for parameter in wright.parameters()) == 17_217
    with pytest.raises(ValueError, match="invalid"):
        build_sota_comparator("wright_lstm64", variant="large")


def _stub_optional_upstream_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    rational = types.ModuleType("rational")
    rational_torch = types.ModuleType("rational.torch")

    class Rational(torch.nn.Tanh):
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            super().__init__()

    rational_torch.Rational = Rational
    rational.torch = rational_torch
    monkeypatch.setitem(sys.modules, "rational", rational)
    monkeypatch.setitem(sys.modules, "rational.torch", rational_torch)

    einops = types.ModuleType("einops")

    def rearrange(tensor: torch.Tensor, pattern: str) -> torch.Tensor:
        if pattern in {"B H L -> B L H", "B L H -> B H L"}:
            return tensor.transpose(1, 2)
        if pattern == "B L -> B L 1":
            return tensor.unsqueeze(-1)
        if pattern == "B H 1 -> B H":
            return tensor.squeeze(-1)
        raise AssertionError(f"unexpected pinned einops pattern: {pattern}")

    def repeat(tensor: torch.Tensor, pattern: str, *, h: int) -> torch.Tensor:
        if pattern != "n -> h n":
            raise AssertionError(f"unexpected pinned einops pattern: {pattern}")
        return tensor.repeat(h, 1)

    einops.rearrange = rearrange
    einops.repeat = repeat
    monkeypatch.setitem(sys.modules, "einops", einops)


@pytest.mark.parametrize("family", ["tcn", "s4"])
def test_local_small_models_are_exactly_equal_to_pinned_upstream_source(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_optional_upstream_modules(monkeypatch)
    root = Path(__file__).resolve().parents[2]
    processors = root / "third_party/nablafx/tests/unit/test_processors"
    monkeypatch.syspath_prepend(str(processors))
    torch.manual_seed(31)
    if family == "tcn":
        upstream_class = import_module("nablafx_tcn.tcn").TCN
        upstream = upstream_class(
            num_inputs=1,
            num_outputs=1,
            num_controls=0,
            num_blocks=5,
            kernel_size=13,
            dilation_growth=10,
            channel_width=16,
            stack_size=12,
            groups=1,
            bias=True,
            causal=True,
            batchnorm=False,
            residual=True,
            direct_path=False,
            cond_type="tfilm",
            cond_block_size=128,
            cond_num_layers=1,
            act_type="tanh",
        )
        local = NablafxTCNTFiLM("small")
    else:
        upstream_class = import_module("nablafx_s4.s4").S4
        upstream = upstream_class(
            num_inputs=1,
            num_outputs=1,
            num_controls=0,
            num_blocks=4,
            channel_width=16,
            s4_state_dim=4,
            batchnorm=False,
            residual=True,
            direct_path=False,
            cond_type="tfilm",
            cond_block_size=128,
            cond_num_layers=1,
            act_type="tanh",
            s4_learning_rate=0.01,
        )
        local = NablafxS4TFiLM("small")
    assert set(upstream.state_dict()) == set(local.state_dict())
    local.load_state_dict(upstream.state_dict())
    signal = torch.randn(1, 1, 257)
    upstream.reset_states()
    with torch.inference_mode():
        expected = upstream(signal).squeeze(1)
        observed = local._forward_raw(signal.squeeze(1))
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
