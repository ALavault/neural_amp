import pytest
import torch

from fssr_nam.models.approximants import (
    CausalControlReconstructor,
    QuinticHermiteSpline,
    SafeRationalActivation,
)


def test_quintic_spline_starts_as_identity_and_has_finite_gradients() -> None:
    spline = QuinticHermiteSpline()
    inputs = torch.linspace(-3.0, 3.0, 1001, requires_grad=True)
    outputs = spline(inputs)
    torch.testing.assert_close(outputs, inputs, atol=2.0e-6, rtol=2.0e-6)
    outputs.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in spline.parameters()
    )


def test_quintic_spline_is_c2_at_internal_knots() -> None:
    torch.manual_seed(40)
    spline = QuinticHermiteSpline(num_knots=9)
    with torch.no_grad():
        spline.values.add_(0.05 * torch.randn_like(spline.values))
        spline.slopes.add_(0.05 * torch.randn_like(spline.slopes))
        spline.curvatures.add_(0.05 * torch.randn_like(spline.curvatures))
    epsilon = 1.0e-5
    for knot in spline.knots[1:-1]:
        points = torch.tensor(
            [float(knot) - epsilon, float(knot) + epsilon], requires_grad=True
        )
        values = spline(points)
        first = torch.autograd.grad(values.sum(), points, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), points)[0]
        assert abs(float((values[0] - values[1]).detach())) < 2.0e-3
        assert abs(float((first[0] - first[1]).detach())) < 2.0e-3
        assert abs(float((second[0] - second[1]).detach())) < 2.0e-3


def test_safe_rational_starts_as_identity_and_has_positive_denominator() -> None:
    activation = SafeRationalActivation()
    inputs = torch.linspace(-20.0, 20.0, 4001, requires_grad=True)
    outputs = activation(inputs)
    torch.testing.assert_close(outputs, inputs, atol=2.0e-6, rtol=2.0e-6)
    outputs.square().mean().backward()
    assert float(activation.denominator.grad[0]) != 0.0
    activation.zero_grad(set_to_none=True)
    with torch.no_grad():
        activation.denominator.copy_(torch.tensor([-4.0, 2.0, -0.5, 0.1]))
    outputs = activation(inputs)
    assert torch.isfinite(outputs).all()
    outputs.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in activation.parameters()
    )


@pytest.mark.parametrize("mode", ["exponential", "slope_limited"])
def test_control_reconstructor_is_causal_resettable_and_block_equivalent(
    mode: str,
) -> None:
    frames = torch.tensor([[[0.0, 1.0, -0.5, 0.25]]])
    whole = CausalControlReconstructor(1, mode, coefficient=0.2)
    expected = whole(frames, frame_size=7)

    streamed = CausalControlReconstructor(1, mode, coefficient=0.2)
    actual = torch.cat(
        (streamed(frames[..., :2], 7), streamed(frames[..., 2:], 7)), dim=-1
    )
    torch.testing.assert_close(actual, expected)
    assert torch.all(expected[..., :7] == 0.0)

    streamed.reset_state()
    torch.testing.assert_close(streamed(frames, 7), expected)


@pytest.mark.parametrize(
    "module",
    [QuinticHermiteSpline(), SafeRationalActivation()],
)
def test_approximants_fail_closed_on_nonfinite_input(module: torch.nn.Module) -> None:
    with pytest.raises(ValueError, match="finite"):
        module(torch.tensor([float("nan")]))
