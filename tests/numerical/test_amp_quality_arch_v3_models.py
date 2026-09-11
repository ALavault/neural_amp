from __future__ import annotations

import math

import pytest
import torch

from fssr_nam.models import V3_FAMILIES, build_arch_v3_candidate


def _delay(signal: torch.Tensor, samples: int = 32) -> torch.Tensor:
    return torch.nn.functional.pad(signal, (samples, 0))[..., :-samples]


@pytest.mark.parametrize("family", V3_FAMILIES)
def test_v3_candidates_are_identity_initialized_finite_and_causal(
    family: str,
) -> None:
    torch.manual_seed(17)
    model = build_arch_v3_candidate(family, profile="slim", initial_residual_scale=0.73)
    signal = torch.randn(2, 193)
    output = model(signal)
    assert output.shape == signal.shape
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, _delay(signal), atol=2.0e-6, rtol=2.0e-6)
    changed = signal.clone()
    changed[:, 129:] = torch.randn_like(changed[:, 129:])
    changed_output = model(changed)
    torch.testing.assert_close(
        output[:, :129], changed_output[:, :129], atol=2.0e-6, rtol=2.0e-6
    )


@pytest.mark.parametrize("family", V3_FAMILIES)
def test_v3_candidates_match_irregular_streaming_and_reset(family: str) -> None:
    torch.manual_seed(23)
    model = build_arch_v3_candidate(family, profile="slim", initial_residual_scale=0.61)
    with torch.no_grad():
        model.branch.output_projection.weight.fill_(0.02)
        model.branch.output_projection.bias.fill_(0.01)
        if model.observer is not None:
            model.observer.modulation_projection.weight.fill_(0.01)
    signal = torch.randn(1, 257)
    expected = model(signal)
    model.reset_state()
    actual = torch.cat(
        (
            model.stream(signal[..., :1]),
            model.stream(signal[..., 1:18]),
            model.stream(signal[..., 18:82]),
            model.stream(signal[..., 82:211]),
            model.stream(signal[..., 211:]),
        ),
        dim=-1,
    )
    torch.testing.assert_close(actual, expected, atol=4.0e-6, rtol=4.0e-6)
    model.reset_state()
    repeated = model.stream(signal)
    torch.testing.assert_close(repeated, expected, atol=4.0e-6, rtol=4.0e-6)


def test_v3_residual_scale_is_positive_unbounded_and_train_initialized() -> None:
    model = build_arch_v3_candidate(
        "gainhead_micro_tcn_x2", profile="slim", initial_residual_scale=0.83
    )
    assert float(model.residual_scale.detach()) == pytest.approx(0.83, abs=1.0e-6)
    with torch.no_grad():
        model.branch.residual_scale_raw.fill_(2.0)
        model.branch.output_projection.bias.fill_(3.0)
    assert float(model.residual_scale.detach()) > 2.0
    signal = torch.zeros(1, 256)
    modulation = torch.zeros(1, 2 * model.channels, 256)
    branch_output = model.branch.forward_modulated(signal, modulation)
    assert float(branch_output.detach().abs().max()) > 1.0


def test_v3_receptive_fields_and_parameter_order_are_explicit() -> None:
    models = {
        family: build_arch_v3_candidate(family, profile="slim")
        for family in V3_FAMILIES
    }
    assert models["gainhead_micro_tcn_x2"].receptive_field_samples == 2047
    assert models["slow_state_micro_tcn_x2"].receptive_field_samples == 2047
    assert models["long_rf_tcn_x2"].receptive_field_samples == 12283
    counts = {
        family: sum(parameter.numel() for parameter in model.parameters())
        for family, model in models.items()
    }
    assert counts["slow_state_micro_tcn_x2"] > counts["gainhead_micro_tcn_x2"]
    assert counts["long_rf_tcn_x2"] > counts["gainhead_micro_tcn_x2"]


@pytest.mark.parametrize("family", V3_FAMILIES)
def test_v3_candidate_gradients_are_finite(family: str) -> None:
    torch.manual_seed(31)
    model = build_arch_v3_candidate(family, profile="slim")
    with torch.no_grad():
        model.branch.output_projection.weight.fill_(0.02)
        if model.observer is not None:
            model.observer.modulation_projection.weight.fill_(0.01)
    signal = torch.randn(2, 129, requires_grad=True)
    output = model.forward_components(signal)
    loss = output.audio.square().mean() + 0.01 * output.state_prediction.square().mean()
    loss.backward()
    assert signal.grad is not None and torch.isfinite(signal.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert math.isfinite(float(model.residual_scale.detach()))


@pytest.mark.parametrize("family", V3_FAMILIES)
def test_v3_stream_state_detaches_between_tbptt_chunks(family: str) -> None:
    torch.manual_seed(37)
    model = build_arch_v3_candidate(family, profile="slim")
    with torch.no_grad():
        model.branch.output_projection.weight.fill_(0.02)
    model.reset_state()
    for _ in range(2):
        signal = torch.randn(1, 64)
        output = model.stream(signal)
        model.detach_stream_state()
        output.square().mean().backward()
        model.zero_grad(set_to_none=True)


def test_v3_factory_rejects_unregistered_family_profile_and_scale() -> None:
    with pytest.raises(ValueError, match="unknown AMP-QUALITY-ARCH-v3 family"):
        build_arch_v3_candidate("result_dependent_family")
    with pytest.raises(ValueError, match="deployment profile"):
        build_arch_v3_candidate("gainhead_micro_tcn_x2", profile="unfrozen")
    with pytest.raises(ValueError, match="residual scale"):
        build_arch_v3_candidate("gainhead_micro_tcn_x2", initial_residual_scale=1.0e-5)
