from __future__ import annotations

import pytest
import torch

from fssr_nam.models.prototype import (
    PROTOTYPE_ACTIVATIONS,
    PROTOTYPE_RESAMPLERS,
    PROTOTYPE_SLOW_CONTROLS,
    CausalSlowControl,
    build_sota_prototype_candidate,
)


def _delay(signal: torch.Tensor, samples: int) -> torch.Tensor:
    return torch.nn.functional.pad(signal, (samples, 0))[..., :-samples]


@pytest.mark.parametrize("activation", PROTOTYPE_ACTIVATIONS)
@pytest.mark.parametrize("resampler", PROTOTYPE_RESAMPLERS)
def test_prototype_is_identity_initialized_finite_shaped_and_causal(
    activation: str, resampler: str
) -> None:
    torch.manual_seed(51)
    model = build_sota_prototype_candidate(
        activation=activation, resampler=resampler, profile="slim"
    )
    signal = torch.randn(1, 97)
    output = model(signal)
    assert output.shape == signal.shape
    assert torch.isfinite(output).all()
    assert model.latency_samples <= 64
    assert model.latency_samples == (
        24 if resampler == "equiripple_halfband_polyphase" else 32
    )
    assert model.resampler_active_multiply_count_per_base_sample == (
        50 if resampler == "equiripple_halfband_polyphase" else 260
    )
    assert model.receptive_field_samples == 12_283
    torch.testing.assert_close(
        output,
        _delay(signal, model.latency_samples),
        atol=2.0e-6,
        rtol=2.0e-6,
    )

    changed = signal.clone()
    changed[:, 65:] = torch.randn_like(changed[:, 65:])
    changed_output = model(changed)
    torch.testing.assert_close(
        output[:, :65], changed_output[:, :65], atol=2.0e-6, rtol=2.0e-6
    )


def test_prototype_active_nonlinear_path_is_finite_and_causal() -> None:
    torch.manual_seed(54)
    model = build_sota_prototype_candidate(
        activation="safe_rational_4_3",
        resampler="equiripple_halfband_polyphase",
        slow_control="causal_exponential_hold",
        profile="slim",
    )
    with torch.no_grad():
        model.branch.output_projection.weight.fill_(0.002)
        model.branch.output_projection.bias.fill_(0.001)
        model.observer.modulation_projection.weight.fill_(0.001)
    signal = 0.1 * torch.randn(1, 97)
    changed = signal.clone()
    changed[:, 65:] = torch.randn_like(changed[:, 65:])
    output = model(signal)
    changed_output = model(changed)
    assert torch.isfinite(output).all()
    assert torch.isfinite(changed_output).all()
    torch.testing.assert_close(
        output[:, :65], changed_output[:, :65], atol=2.0e-6, rtol=2.0e-6
    )


@pytest.mark.parametrize("mode", PROTOTYPE_SLOW_CONTROLS)
def test_slow_control_is_neutral_until_completed_block_and_has_block_parity(
    mode: str,
) -> None:
    targets = torch.ones(2, 3, 9)
    control = CausalSlowControl(3, mode, update_samples=4, coefficient=0.25)
    expected = control(targets)
    torch.testing.assert_close(expected[..., :4], torch.zeros_like(expected[..., :4]))
    first_valid = 1.0 if mode == "causal_zero_order_hold" else 0.25
    torch.testing.assert_close(
        expected[..., 4], torch.full_like(expected[..., 4], first_valid)
    )

    actual = torch.cat(
        (
            control.stream(targets[..., :3]),
            control.stream(targets[..., 3:6]),
            control.stream(targets[..., 6:]),
        ),
        dim=-1,
    )
    assert float((actual - expected).abs().max()) <= 2.0e-5
    control.reset_state()
    torch.testing.assert_close(control.stream(targets), expected)


@pytest.mark.parametrize("mode", PROTOTYPE_SLOW_CONTROLS)
def test_model_applies_slow_control_after_observer_at_sample_64(mode: str) -> None:
    model = build_sota_prototype_candidate(
        activation="tanh",
        resampler="kaiser_windowed_sinc",
        slow_control=mode,
        profile="slim",
    )
    with torch.no_grad():
        model.observer.modulation_projection.weight.zero_()
        model.observer.modulation_projection.bias.fill_(1.0)
    signal = torch.zeros(1, 65)
    features = model.feature_bus(signal)
    raw_modulation, _ = model.observer(features)
    modulation = model.slow_control(raw_modulation)
    torch.testing.assert_close(
        modulation[..., :64], torch.zeros_like(modulation[..., :64])
    )
    first_valid = 1.0 if mode == "causal_zero_order_hold" else 0.25
    torch.testing.assert_close(
        modulation[..., 64], torch.full_like(modulation[..., 64], first_valid)
    )


@pytest.mark.parametrize("mode", PROTOTYPE_SLOW_CONTROLS)
def test_slow_control_options_preserve_full_model_block_parity(mode: str) -> None:
    torch.manual_seed(55)
    model = build_sota_prototype_candidate(
        activation="tanh",
        resampler="equiripple_halfband_polyphase",
        slow_control=mode,
        profile="slim",
    )
    with torch.no_grad():
        model.branch.output_projection.weight.fill_(0.002)
        model.branch.output_projection.bias.fill_(0.001)
        model.observer.modulation_projection.bias.fill_(0.5)
    signal = 0.1 * torch.randn(1, 129)
    expected = model(signal)
    model.reset()
    actual = torch.cat(
        (
            model.stream(signal[..., :63]),
            model.stream(signal[..., 63:65]),
            model.stream(signal[..., 65:]),
        ),
        dim=-1,
    )
    assert float((actual - expected).abs().max().detach()) <= 2.0e-5


@pytest.mark.parametrize("activation", PROTOTYPE_ACTIVATIONS)
@pytest.mark.parametrize("resampler", PROTOTYPE_RESAMPLERS)
def test_prototype_matches_irregular_blocks_and_deterministic_reset(
    activation: str, resampler: str
) -> None:
    torch.manual_seed(52)
    model = build_sota_prototype_candidate(
        activation=activation, resampler=resampler, profile="slim"
    )
    with torch.no_grad():
        model.branch.output_projection.weight.fill_(0.002)
        model.branch.output_projection.bias.fill_(0.001)
        model.observer.modulation_projection.weight.fill_(0.001)
    signal = 0.1 * torch.randn(1, 129)
    expected = model(signal)
    model.reset()
    actual = torch.cat(
        (
            model.stream(signal[..., :1]),
            model.stream(signal[..., 1:18]),
            model.stream(signal[..., 18:66]),
            model.stream(signal[..., 66:111]),
            model.stream(signal[..., 111:]),
        ),
        dim=-1,
    )
    assert float((actual - expected).abs().max().detach()) <= 2.0e-5
    torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)
    model.reset()
    repeated = model.stream(signal)
    assert float((repeated - expected).abs().max().detach()) <= 2.0e-5
    torch.testing.assert_close(repeated, expected, atol=2.0e-5, rtol=2.0e-5)


def test_prototype_detaches_stream_state_for_truncated_bptt() -> None:
    torch.manual_seed(53)
    model = build_sota_prototype_candidate(
        activation="safe_rational_4_3",
        resampler="equiripple_halfband_polyphase",
        slow_control="causal_slope_limited_hold",
        profile="slim",
    )
    with torch.no_grad():
        model.branch.output_projection.weight.fill_(0.002)
    for _ in range(2):
        output = model.stream(0.1 * torch.randn(1, 64))
        model.detach()
        output.square().mean().backward()
        model.zero_grad(set_to_none=True)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("activation", "lagrange", "activation"),
        ("resampler", "linear", "resampler"),
        ("slow_control", "lookahead", "slow control"),
        ("profile", "unregistered", "deployment profile"),
    ],
)
def test_prototype_factory_rejects_unregistered_options(
    argument: str, value: str, message: str
) -> None:
    options = {
        "activation": "tanh",
        "resampler": "kaiser_windowed_sinc",
        "slow_control": "causal_zero_order_hold",
        "profile": "slim",
    }
    options[argument] = value
    with pytest.raises(ValueError, match=message):
        build_sota_prototype_candidate(**options)
