from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from fssr_nam.models.arch_v1 import (
    DEPLOYMENT_PROFILES,
    CausalBlockFeatureBus,
    CausalSelectiveObserver,
    PhysicsConditionedTCN,
    build_arch_v1_candidate,
)

CANDIDATE_FAMILIES = (
    "selective_s6_x2",
    "micro_tcn_x2",
    "phys_s6_tcn_x2",
    "phys_det_tcn_x2",
    "rf2047_tfilm_x2",
    "cascade_rf2047_tfilm_x2",
)


def _chunks(length: int, proposed: list[int]) -> list[int]:
    result: list[int] = []
    remaining = length
    for value in proposed:
        if remaining == 0:
            break
        take = min(value, remaining)
        result.append(take)
        remaining -= take
    if remaining:
        result.append(remaining)
    return result


@given(
    proposed=st.lists(st.integers(min_value=1, max_value=97), min_size=1, max_size=12)
)
@settings(max_examples=12, deadline=None)
def test_physical_feature_bus_full_stream_roundtrip(proposed: list[int]) -> None:
    torch.manual_seed(12)
    signal = torch.randn(2, 257)
    bus = CausalBlockFeatureBus(decimation=64)
    expected = bus(signal)
    bus.reset_state()
    outputs = []
    offset = 0
    for size in _chunks(signal.shape[-1], proposed):
        outputs.append(bus.stream(signal[:, offset : offset + size]))
        offset += size
    actual = torch.cat(outputs, dim=-1)
    torch.testing.assert_close(actual, expected, atol=1.0e-7, rtol=1.0e-7)


def test_physical_feature_bus_is_causal_and_resettable() -> None:
    bus = CausalBlockFeatureBus(decimation=16)
    prefix = torch.linspace(-0.5, 0.5, 41)
    first = torch.cat((prefix, torch.zeros(37)))
    second = torch.cat((prefix, torch.ones(37)))
    torch.testing.assert_close(bus(first)[..., :41], bus(second)[..., :41])
    expected = bus(first)
    bus.stream(first[:19])
    bus.reset_state()
    torch.testing.assert_close(bus.stream(first), expected)


def test_selective_observer_full_stream_gradient_and_reset() -> None:
    torch.manual_seed(23)
    features = torch.randn(2, 6, 193, requires_grad=True)
    observer = CausalSelectiveObserver(6, 9, 14, decimation=32)
    expected_modulation, expected_auxiliary = observer(features)
    loss = expected_modulation.square().mean() + expected_auxiliary.square().mean()
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in observer.parameters()
    )

    observer.reset_state()
    modulation_parts = []
    auxiliary_parts = []
    offset = 0
    for size in (1, 17, 64, 3, 108):
        modulation, auxiliary = observer.stream(
            features.detach()[..., offset : offset + size]
        )
        modulation_parts.append(modulation)
        auxiliary_parts.append(auxiliary)
        offset += size
    torch.testing.assert_close(
        torch.cat(modulation_parts, dim=-1),
        expected_modulation.detach(),
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    torch.testing.assert_close(
        torch.cat(auxiliary_parts, dim=-1),
        expected_auxiliary.detach(),
        atol=2.0e-6,
        rtol=2.0e-6,
    )


@pytest.mark.parametrize("conditioning", ["none", "deterministic", "observer"])
def test_conditioned_x2_model_is_identity_after_delay_at_initialization(
    conditioning: str,
) -> None:
    torch.manual_seed(34)
    model = PhysicsConditionedTCN(
        conditioning=conditioning, channels=4, observer_state_dim=7
    )
    signal = torch.randn(2, 257)
    output = model(signal)
    expected = torch.nn.functional.pad(signal, (model.latency_samples, 0))[
        ..., : -model.latency_samples
    ]
    assert model.receptive_field_samples == 2047
    assert output.shape == signal.shape
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, atol=2.0e-7, rtol=2.0e-7)


@pytest.mark.parametrize("conditioning", ["none", "deterministic", "observer"])
def test_conditioned_x2_model_full_stream_irregular_and_reset(
    conditioning: str,
) -> None:
    torch.manual_seed(45)
    model = PhysicsConditionedTCN(
        conditioning=conditioning, channels=3, observer_state_dim=5
    )
    with torch.no_grad():
        model.branch.output_projection.weight.fill_(0.02)
        model.branch.output_projection.bias.fill_(-0.01)
        if model.deterministic_projection is not None:
            model.deterministic_projection.weight.fill_(0.01)
        if model.observer is not None:
            model.observer.modulation_projection.weight.fill_(0.01)
    signal = torch.randn(2, 311)
    expected = model.forward_components(signal)

    model.reset_state()
    audio_parts = []
    state_parts = []
    feature_parts = []
    offset = 0
    for size in (1, 63, 5, 128, 17, 97):
        result = model.stream_components(signal[..., offset : offset + size])
        audio_parts.append(result.audio)
        state_parts.append(result.state_prediction)
        feature_parts.append(result.dry_features)
        offset += size
    torch.testing.assert_close(
        torch.cat(audio_parts, dim=-1), expected.audio, atol=3.0e-6, rtol=3.0e-6
    )
    torch.testing.assert_close(
        torch.cat(state_parts, dim=-1),
        expected.state_prediction,
        atol=3.0e-6,
        rtol=3.0e-6,
    )
    torch.testing.assert_close(
        torch.cat(feature_parts, dim=-1),
        expected.dry_features,
        atol=3.0e-6,
        rtol=3.0e-6,
    )

    model.reset_state()
    torch.testing.assert_close(
        model.stream(signal).detach(), expected.audio, atol=3.0e-6, rtol=3.0e-6
    )


def test_conditioned_x2_model_has_finite_input_and_parameter_gradients() -> None:
    torch.manual_seed(56)
    model = PhysicsConditionedTCN(
        conditioning="observer", channels=3, observer_state_dim=5
    )
    with torch.no_grad():
        model.branch.output_projection.weight.fill_(0.02)
        assert model.observer is not None
        model.observer.modulation_projection.weight.fill_(0.01)
    signal = torch.randn(2, 129, requires_grad=True)
    result = model.forward_components(signal)
    loss = result.audio.square().mean() + 0.05 * result.state_prediction.square().mean()
    loss.backward()
    assert signal.grad is not None and torch.isfinite(signal.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_architecture_interfaces_reject_invalid_shapes_and_modes() -> None:
    with pytest.raises(ValueError, match="conditioning"):
        PhysicsConditionedTCN(conditioning="wet_target")
    bus = CausalBlockFeatureBus()
    with pytest.raises(ValueError, match="shape"):
        bus(torch.zeros(1, 1, 12))
    observer = CausalSelectiveObserver(6, 4, 8)
    with pytest.raises(ValueError, match="shape"):
        observer(torch.zeros(2, 5, 12))


@pytest.mark.parametrize("family", CANDIDATE_FAMILIES)
def test_frozen_candidate_factory_is_finite_causal_and_block_invariant(
    family: str,
) -> None:
    torch.manual_seed(67)
    model = build_arch_v1_candidate(family, profile="slim")
    signal = torch.randn(1, 97)
    expected = model(signal)
    assert expected.shape == signal.shape
    assert torch.isfinite(expected).all()
    model.reset_state()
    actual = torch.cat(
        (
            model.stream(signal[..., :1]),
            model.stream(signal[..., 1:64]),
            model.stream(signal[..., 64:69]),
            model.stream(signal[..., 69:]),
        ),
        dim=-1,
    )
    torch.testing.assert_close(actual, expected, atol=3.0e-6, rtol=3.0e-6)


def test_teacher_factory_is_limited_to_two_training_eligible_families() -> None:
    for family in ("micro_tcn_x2", "phys_s6_tcn_x2"):
        assert build_arch_v1_candidate(family, teacher=True) is not None
    with pytest.raises(ValueError, match="not eligible"):
        build_arch_v1_candidate("selective_s6_x2", teacher=True)
    with pytest.raises(ValueError, match="not eligible"):
        build_arch_v1_candidate("phys_det_tcn_x2", teacher=True)
    with pytest.raises(ValueError, match="unknown"):
        build_arch_v1_candidate("new_result_dependent_family")
    with pytest.raises(ValueError, match="deployment profile"):
        build_arch_v1_candidate("micro_tcn_x2", profile="new_result_dependent_width")


@pytest.mark.parametrize("family", CANDIDATE_FAMILIES)
def test_deployment_profiles_have_strictly_increasing_parameter_counts(
    family: str,
) -> None:
    counts = [
        sum(
            parameter.numel()
            for parameter in build_arch_v1_candidate(
                family, profile=profile
            ).parameters()
        )
        for profile in DEPLOYMENT_PROFILES
    ]
    assert counts == sorted(counts)
    assert len(set(counts)) == len(counts)
