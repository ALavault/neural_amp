import pytest
import torch

from fssr_nam.models import SmoothHermiteSpline


def test_spline_identity_initialization_and_extrapolation():
    spline = SmoothHermiteSpline(17, -2.0, 2.0)
    inputs = torch.linspace(-3.0, 3.0, 1001)
    torch.testing.assert_close(spline(inputs), inputs, atol=2.0e-6, rtol=2.0e-6)
    torch.testing.assert_close(
        spline.derivative(inputs), torch.ones_like(inputs), atol=2.0e-6, rtol=2.0e-6
    )


def test_spline_is_c1_at_knots():
    torch.manual_seed(4)
    spline = SmoothHermiteSpline(9)
    with torch.no_grad():
        spline.values.add_(0.08 * torch.randn_like(spline.values))
        spline.slopes.add_(0.08 * torch.randn_like(spline.slopes))
    epsilon = 1.0e-5
    for knot in spline.knots[1:-1]:
        left = knot - epsilon
        right = knot + epsilon
        assert abs(float((spline(left) - spline(right)).detach())) < 1.0e-4
        derivative_jump = spline.derivative(left) - spline.derivative(right)
        assert abs(float(derivative_jump.detach())) < 2.0e-4


def test_analytic_derivative_and_primitive_match_autograd():
    torch.manual_seed(5)
    spline = SmoothHermiteSpline(11)
    with torch.no_grad():
        spline.values.add_(0.1 * torch.randn_like(spline.values))
        spline.slopes.add_(0.1 * torch.randn_like(spline.slopes))
    inputs = torch.linspace(-2.5, 2.5, 257, requires_grad=True)
    automatic = torch.autograd.grad(spline(inputs).sum(), inputs)[0]
    torch.testing.assert_close(
        spline.derivative(inputs), automatic, atol=2.0e-5, rtol=2.0e-5
    )
    primitive_derivative = torch.autograd.grad(
        spline.antiderivative(inputs).sum(), inputs
    )[0]
    torch.testing.assert_close(
        primitive_derivative, spline(inputs), atol=2.0e-5, rtol=2.0e-5
    )


def test_spline_rejects_nonfinite_input_and_has_finite_gradients():
    spline = SmoothHermiteSpline()
    with pytest.raises(ValueError, match="finite"):
        spline(torch.tensor([float("nan")]))
    loss = spline(torch.linspace(-3.0, 3.0, 1024)).square().mean()
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in spline.parameters()
    )


def test_spline_learns_tanh():
    torch.manual_seed(6)
    spline = SmoothHermiteSpline(17)
    inputs = torch.linspace(-1.5, 1.5, 512)
    target = torch.tanh(2.8 * inputs) / torch.tanh(torch.tensor(2.8))
    optimizer = torch.optim.Adam(spline.parameters(), lr=0.03)
    for _ in range(250):
        optimizer.zero_grad()
        loss = (
            torch.mean((spline(inputs) - target) ** 2)
            + 1.0e-5 * spline.curvature_penalty()
        )
        loss.backward()
        optimizer.step()
    assert float(torch.mean((spline(inputs) - target) ** 2)) < 2.0e-5
