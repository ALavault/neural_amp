from itertools import pairwise

import pytest
import torch

from fssr_nam.models.fssr import S3FastSlowResidual
from fssr_nam.models.r1 import (
    CascadeSplineCore,
    R1Cascade,
    R1Mono,
    RF31Residual,
    RF2047Residual,
)


def _activate_dynamic_paths(model: R1Mono | R1Cascade) -> None:
    with torch.no_grad():
        model.residual.output_projection.weight.fill_(0.05)
        model.residual.output_projection.bias.zero_()
        model.slow.projection.weight.fill_(0.03)
        model.slow.projection.bias.fill_(0.01)


@pytest.mark.parametrize(
    ("model", "expected_rf"),
    [
        (R1Mono(receptive_field=31), 31),
        (R1Mono(receptive_field=2047), 2047),
        (R1Cascade(), 2047),
    ],
)
def test_r1_models_are_finite_identity_initialized_and_have_expected_shape(
    model: R1Mono | R1Cascade, expected_rf: int
) -> None:
    signal = torch.linspace(-0.8, 0.8, 257)
    output, core, residual = model.forward_components(signal)
    assert output.shape == core.shape == residual.shape == signal.shape
    assert torch.isfinite(output).all()
    assert model.receptive_field == expected_rf
    torch.testing.assert_close(output, signal, rtol=2.0e-6, atol=2.0e-6)
    torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0.0, atol=0.0)


def test_rf31_composition_is_state_dict_compatible_with_historical_s3() -> None:
    torch.manual_seed(30)
    historical = S3FastSlowResidual(
        taps=5,
        num_knots=9,
        hidden_size=4,
        decimation=8,
        residual_channels=3,
    )
    control = R1Mono(
        taps=5,
        num_knots=9,
        slow_hidden_size=4,
        slow_decimation=8,
        residual_channels=3,
        receptive_field=31,
    )
    control.load_state_dict(historical.state_dict(), strict=True)
    signal = 0.1 * torch.randn(2, 97)
    torch.testing.assert_close(control(signal), historical(signal), rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "model",
    [R1Mono(receptive_field=31), R1Mono(receptive_field=2047), R1Cascade()],
)
def test_r1_irregular_blocks_match_full_processing_after_reset(
    model: R1Mono | R1Cascade,
) -> None:
    torch.manual_seed(31)
    _activate_dynamic_paths(model)
    signal = 0.1 * torch.randn(2, 223)
    expected = model(signal)
    boundaries = (0, 1, 6, 37, 38, 109, 167, 223)
    streamed = torch.cat(
        [model.stream(signal[:, left:right]) for left, right in pairwise(boundaries)],
        dim=-1,
    )
    torch.testing.assert_close(streamed, expected, rtol=2.0e-6, atol=2.0e-6)
    model.reset_state()
    torch.testing.assert_close(model.stream(signal), expected, rtol=2.0e-6, atol=2.0e-6)


@pytest.mark.parametrize("model", [R1Mono(receptive_field=31), R1Cascade()])
def test_r1_models_are_strictly_causal(model: R1Mono | R1Cascade) -> None:
    torch.manual_seed(32)
    _activate_dynamic_paths(model)
    signal = 0.1 * torch.randn(211)
    changed = signal.clone()
    changed[113:] = torch.randn_like(changed[113:])
    torch.testing.assert_close(
        model(signal)[:113], model(changed)[:113], rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize(
    ("residual", "expected_rf"),
    [(RF31Residual(channels=1), 31), (RF2047Residual(channels=1), 2047)],
)
def test_residual_receptive_field_is_exact(
    residual: RF31Residual | RF2047Residual, expected_rf: int
) -> None:
    residual = residual.double()
    with torch.no_grad():
        residual.input_projection.weight.fill_(0.05)
        residual.input_projection.bias.zero_()
        for layer in residual.layers:
            convolution = getattr(layer, "conv", None)
            if convolution is not None:
                convolution.weight.fill_(0.05)
                convolution.bias.zero_()
            else:
                layer.depthwise.weight.fill_(0.05)
                layer.depthwise.bias.zero_()
                layer.pointwise.weight.fill_(0.05)
                layer.pointwise.bias.zero_()
        residual.output_projection.weight.fill_(0.05)
        residual.output_projection.bias.zero_()
    features = torch.full(
        (1, 2, expected_rf + 1), 0.01, dtype=torch.float64, requires_grad=True
    )
    residual(features)[0, -1].backward()
    gradient = features.grad[0, 0]
    assert gradient[0] == 0.0
    assert gradient[1] != 0.0
    assert torch.isfinite(gradient).all()
    assert residual.receptive_field == expected_rf


def test_cascade_core_matches_two_clipper_synthetic_topology() -> None:
    core = CascadeSplineCore(taps=3, num_knots=17)
    with torch.no_grad():
        core.spline1.values.copy_(torch.clamp(core.spline1.knots, -0.4, 0.4))
        core.spline1.slopes.zero_()
        core.spline2.values.copy_(torch.clamp(core.spline2.knots, -0.2, 0.2))
        core.spline2.slopes.zero_()
        core.h1.coefficients.copy_(torch.tensor([0.15, -0.25, 0.8]))
    signal = torch.linspace(-1.0, 1.0, 97)
    expected = (
        core.h2(
            core.spline2(
                core.drive2
                * core.h1(core.spline1(core.drive1 * core.h0(signal) + core.offset1))
                + core.offset2
            )
        )
        * core.output_gain
    )
    torch.testing.assert_close(core(signal), expected, rtol=0.0, atol=0.0)
    assert torch.isfinite(expected).all()


def test_r1_cascade_gradients_are_finite() -> None:
    torch.manual_seed(33)
    model = R1Cascade(taps=5, num_knots=9)
    _activate_dynamic_paths(model)
    signal = 0.1 * torch.randn(2, 79)
    loss = (model(signal) - torch.tanh(2.0 * signal)).square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all() for gradient in gradients if gradient is not None
    )


def test_cascade_loads_promoted_rf2047_strictly_and_atomically() -> None:
    torch.manual_seed(34)
    promoted = RF2047Residual()
    with torch.no_grad():
        for parameter in promoted.parameters():
            parameter.uniform_(-0.05, 0.05)
    cascade = R1Cascade(promoted_residual_state=promoted)
    for expected, actual in zip(
        promoted.state_dict().values(),
        cascade.residual.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    before = {
        name: value.detach().clone()
        for name, value in cascade.residual.state_dict().items()
    }
    incomplete = dict(promoted.state_dict())
    incomplete.pop("output_projection.bias")
    with pytest.raises(ValueError, match="keys"):
        cascade.load_promoted_residual_state(incomplete)
    for name, value in cascade.residual.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)

    with pytest.raises(ValueError, match="requires an RF2047"):
        R1Cascade(receptive_field=31).load_promoted_residual_state(promoted)
