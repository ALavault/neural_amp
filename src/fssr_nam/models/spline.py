"""Smooth learnable Hermite spline with analytic derivatives and primitives."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SmoothHermiteSpline(nn.Module):
    """Uniform-knot cubic Hermite spline with linear endpoint extrapolation."""

    def __init__(
        self, num_knots: int = 17, minimum: float = -2.0, maximum: float = 2.0
    ):
        super().__init__()
        if num_knots < 4:
            raise ValueError("num_knots must be at least four")
        if not minimum < maximum:
            raise ValueError("minimum must be less than maximum")
        knots = torch.linspace(minimum, maximum, num_knots)
        self.register_buffer("knots", knots)
        self.values = nn.Parameter(knots.clone())
        self.slopes = nn.Parameter(torch.ones_like(knots))

    @property
    def spacing(self) -> float:
        return float((self.knots[-1] - self.knots[0]) / (len(self.knots) - 1))

    def _coordinates(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        spacing = self.spacing
        coordinate = (inputs - self.knots[0]) / spacing
        index = torch.floor(coordinate).to(torch.long)
        index = torch.clamp(index, 0, len(self.knots) - 2)
        local = coordinate - index
        return (
            index,
            local,
            torch.as_tensor(spacing, device=inputs.device, dtype=inputs.dtype),
        )

    def _finite_input(self, inputs: Tensor) -> None:
        if not torch.isfinite(inputs).all():
            raise ValueError("spline input must be finite")

    def forward(self, inputs: Tensor) -> Tensor:
        self._finite_input(inputs)
        index, local, spacing = self._coordinates(inputs)
        y0 = self.values[index]
        y1 = self.values[index + 1]
        m0 = self.slopes[index]
        m1 = self.slopes[index + 1]
        h00 = 2 * local**3 - 3 * local**2 + 1
        h10 = local**3 - 2 * local**2 + local
        h01 = -2 * local**3 + 3 * local**2
        h11 = local**3 - local**2
        inside = h00 * y0 + h10 * spacing * m0 + h01 * y1 + h11 * spacing * m1
        left = self.values[0] + self.slopes[0] * (inputs - self.knots[0])
        right = self.values[-1] + self.slopes[-1] * (inputs - self.knots[-1])
        return torch.where(
            inputs < self.knots[0],
            left,
            torch.where(inputs > self.knots[-1], right, inside),
        )

    def derivative(self, inputs: Tensor) -> Tensor:
        self._finite_input(inputs)
        index, local, spacing = self._coordinates(inputs)
        y0 = self.values[index]
        y1 = self.values[index + 1]
        m0 = self.slopes[index]
        m1 = self.slopes[index + 1]
        derivative = (
            (6 * local**2 - 6 * local) * y0 / spacing
            + (3 * local**2 - 4 * local + 1) * m0
            + (-6 * local**2 + 6 * local) * y1 / spacing
            + (3 * local**2 - 2 * local) * m1
        )
        return torch.where(
            inputs < self.knots[0],
            self.slopes[0],
            torch.where(inputs > self.knots[-1], self.slopes[-1], derivative),
        )

    def antiderivative(self, inputs: Tensor) -> Tensor:
        """Return an analytic primitive whose value is zero at the first knot."""
        self._finite_input(inputs)
        spacing_value = self.spacing
        spacing = torch.as_tensor(
            spacing_value, device=inputs.device, dtype=inputs.dtype
        )
        segment_integrals = spacing * (
            0.5 * self.values[:-1]
            + 0.5 * self.values[1:]
            + spacing * (self.slopes[:-1] - self.slopes[1:]) / 12.0
        )
        cumulative = torch.cat(
            (
                torch.zeros_like(segment_integrals[:1]),
                torch.cumsum(segment_integrals, 0),
            )
        )
        index, local, _ = self._coordinates(inputs)
        y0 = self.values[index]
        y1 = self.values[index + 1]
        m0 = self.slopes[index]
        m1 = self.slopes[index + 1]
        local_integral = spacing * (
            (0.5 * local**4 - local**3 + local) * y0
            + (0.25 * local**4 - 2.0 * local**3 / 3.0 + 0.5 * local**2) * spacing * m0
            + (-0.5 * local**4 + local**3) * y1
            + (0.25 * local**4 - local**3 / 3.0) * spacing * m1
        )
        inside = cumulative[index] + local_integral
        left_distance = inputs - self.knots[0]
        left = self.values[0] * left_distance + 0.5 * self.slopes[0] * left_distance**2
        right_distance = inputs - self.knots[-1]
        right = (
            cumulative[-1]
            + self.values[-1] * right_distance
            + 0.5 * self.slopes[-1] * right_distance**2
        )
        return torch.where(
            inputs < self.knots[0],
            left,
            torch.where(inputs > self.knots[-1], right, inside),
        )

    def curvature_penalty(self) -> Tensor:
        second_difference = self.values[:-2] - 2 * self.values[1:-1] + self.values[2:]
        slope_difference = self.slopes[1:] - self.slopes[:-1]
        return torch.mean(second_difference**2) + torch.mean(slope_difference**2)
