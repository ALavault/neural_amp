"""Prospective nonlinear and slow-control approximants for SOTA-PROTOTYPE-v1."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class QuinticHermiteSpline(nn.Module):
    """Uniform C2 quintic Hermite spline with quadratic endpoint tails."""

    def __init__(
        self, num_knots: int = 17, minimum: float = -2.0, maximum: float = 2.0
    ) -> None:
        super().__init__()
        if num_knots < 4:
            raise ValueError("num_knots must be at least four")
        if not minimum < maximum:
            raise ValueError("minimum must be less than maximum")
        knots = torch.linspace(minimum, maximum, num_knots)
        self.register_buffer("knots", knots)
        self.values = nn.Parameter(knots.clone())
        self.slopes = nn.Parameter(torch.ones_like(knots))
        self.curvatures = nn.Parameter(torch.zeros_like(knots))

    @property
    def spacing(self) -> float:
        return float((self.knots[-1] - self.knots[0]) / (len(self.knots) - 1))

    def _coordinates(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        coordinate = (inputs - self.knots[0]) / self.spacing
        index = torch.floor(coordinate).to(torch.long)
        index = torch.clamp(index, 0, len(self.knots) - 2)
        local = coordinate - index
        spacing = inputs.new_tensor(self.spacing)
        return index, local, spacing

    @staticmethod
    def _require_finite(inputs: Tensor) -> None:
        if not torch.isfinite(inputs).all():
            raise ValueError("quintic spline input must be finite")

    def forward(self, inputs: Tensor) -> Tensor:
        self._require_finite(inputs)
        index, t, h = self._coordinates(inputs)
        y0, y1 = self.values[index], self.values[index + 1]
        m0, m1 = self.slopes[index], self.slopes[index + 1]
        c0, c1 = self.curvatures[index], self.curvatures[index + 1]
        h00 = 1 - 10 * t**3 + 15 * t**4 - 6 * t**5
        h10 = t - 6 * t**3 + 8 * t**4 - 3 * t**5
        h20 = 0.5 * (t**2 - 3 * t**3 + 3 * t**4 - t**5)
        h01 = 10 * t**3 - 15 * t**4 + 6 * t**5
        h11 = -4 * t**3 + 7 * t**4 - 3 * t**5
        h21 = 0.5 * (t**3 - 2 * t**4 + t**5)
        inside = (
            h00 * y0
            + h10 * h * m0
            + h20 * h.square() * c0
            + h01 * y1
            + h11 * h * m1
            + h21 * h.square() * c1
        )
        left_dx = inputs - self.knots[0]
        right_dx = inputs - self.knots[-1]
        left = (
            self.values[0]
            + self.slopes[0] * left_dx
            + 0.5 * self.curvatures[0] * left_dx.square()
        )
        right = (
            self.values[-1]
            + self.slopes[-1] * right_dx
            + 0.5 * self.curvatures[-1] * right_dx.square()
        )
        return torch.where(
            inputs < self.knots[0],
            left,
            torch.where(inputs > self.knots[-1], right, inside),
        )

    def curvature_penalty(self) -> Tensor:
        return torch.mean((self.curvatures[1:] - self.curvatures[:-1]).square())


class SafeRationalActivation(nn.Module):
    """Trainable P4/(1 + |Q3|), which has no real denominator poles."""

    def __init__(self) -> None:
        super().__init__()
        self.numerator = nn.Parameter(torch.tensor([0.0, 1.001, 0.0, 0.0, 0.0]))
        # A tiny nonzero constant preserves identity to float32 tolerance while
        # avoiding the exactly-zero denominator gradient of abs(Q) at Q == 0.
        self.denominator = nn.Parameter(torch.tensor([1.0e-3, 0.0, 0.0, 0.0]))

    @staticmethod
    def _polynomial(coefficients: Tensor, inputs: Tensor) -> Tensor:
        result = torch.zeros_like(inputs)
        for coefficient in coefficients.flip(0):
            result = result * inputs + coefficient
        return result

    def forward(self, inputs: Tensor) -> Tensor:
        if not torch.isfinite(inputs).all():
            raise ValueError("rational activation input must be finite")
        numerator = self._polynomial(self.numerator, inputs)
        denominator = 1.0 + torch.abs(self._polynomial(self.denominator, inputs))
        return numerator / denominator


class CausalControlReconstructor(nn.Module):
    """Causal stateful reconstruction of reduced-rate control frames."""

    def __init__(self, channels: int, mode: str, coefficient: float = 0.25) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if mode not in {"exponential", "slope_limited"}:
            raise ValueError("unsupported causal reconstruction mode")
        if not 0.0 < coefficient <= 1.0:
            raise ValueError("coefficient must be in (0, 1]")
        self.channels = channels
        self.mode = mode
        self.coefficient = coefficient
        self._state: Tensor | None = None

    def reset_state(self) -> None:
        self._state = None

    def forward(self, frames: Tensor, frame_size: int) -> Tensor:
        """Expand ``[batch, channels, frames]`` without future-frame access."""
        if frames.ndim != 3 or frames.shape[1] != self.channels:
            raise ValueError("frames must have shape [batch, channels, time]")
        if frame_size < 1:
            raise ValueError("frame_size must be positive")
        if not torch.isfinite(frames).all():
            raise ValueError("control frames must be finite")
        state = self._state
        if state is None or state.shape != frames[:, :, 0].shape:
            state = frames[:, :, 0]
        samples: list[Tensor] = []
        for frame in frames.unbind(dim=-1):
            for _ in range(frame_size):
                delta = frame - state
                if self.mode == "exponential":
                    state = state + self.coefficient * delta
                else:
                    limit = self.coefficient * (1.0 + state.abs())
                    state = state + torch.clamp(delta, -limit, limit)
                samples.append(state)
        self._state = state.detach()
        return torch.stack(samples, dim=-1)
