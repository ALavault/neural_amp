"""Causal FIR structured core used by FSSR-NAM S0."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .spline import SmoothHermiteSpline


def _batch(signal: Tensor) -> tuple[Tensor, bool]:
    if signal.ndim == 1:
        return signal[None, :], True
    if signal.ndim != 2:
        raise ValueError("audio must have shape (samples,) or (batch,samples)")
    return signal, False


class CausalFIR(nn.Module):
    """Mono causal FIR with equivalent stateless and streaming paths."""

    def __init__(self, taps: int = 17):
        super().__init__()
        if taps < 1:
            raise ValueError("taps must be positive")
        coefficients = torch.zeros(taps)
        coefficients[-1] = 1.0
        self.coefficients = nn.Parameter(coefficients)
        self._state: Tensor | None = None

    @property
    def taps(self) -> int:
        return len(self.coefficients)

    def reset_state(self) -> None:
        self._state = None

    def forward(self, signal: Tensor) -> Tensor:
        batched, scalar = _batch(signal)
        padded = functional.pad(batched[:, None, :], (self.taps - 1, 0))
        output = functional.conv1d(padded, self.coefficients[None, None, :])[:, 0]
        return output[0] if scalar else output

    def stream(self, signal: Tensor) -> Tensor:
        batched, scalar = _batch(signal)
        if self._state is None:
            self._state = batched.new_zeros((len(batched), self.taps - 1))
        if self._state.shape[0] != batched.shape[0]:
            raise ValueError("stream batch size changed without reset")
        joined = torch.cat((self._state, batched), dim=-1)
        output = functional.conv1d(
            joined[:, None, :], self.coefficients[None, None, :]
        )[:, 0]
        self._state = joined[:, -(self.taps - 1) :] if self.taps > 1 else joined[:, :0]
        return output[0] if scalar else output


class S0Structured(nn.Module):
    """One-branch pre-FIR, smooth waveshaper, and post-FIR model."""

    code = "S0"
    latency_samples = 0

    def __init__(self, taps: int = 17, num_knots: int = 17):
        super().__init__()
        self.pre = CausalFIR(taps)
        self.shaper = SmoothHermiteSpline(num_knots)
        self.post = CausalFIR(taps)
        self.drive = nn.Parameter(torch.tensor(1.0))
        self.offset = nn.Parameter(torch.tensor(0.0))
        self.output_gain = nn.Parameter(torch.tensor(1.0))

    def reset_state(self) -> None:
        self.pre.reset_state()
        self.post.reset_state()

    def forward(self, signal: Tensor) -> Tensor:
        shaped = self.shaper(self.drive * self.pre(signal) + self.offset)
        return self.output_gain * self.post(shaped)

    def stream(self, signal: Tensor) -> Tensor:
        shaped = self.shaper(self.drive * self.pre.stream(signal) + self.offset)
        return self.output_gain * self.post.stream(shaped)

    def regularization(self) -> Tensor:
        return self.shaper.curvature_penalty()
