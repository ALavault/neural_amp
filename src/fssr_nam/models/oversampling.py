"""Causal local x2 oversampling around the learnable nonlinear residual."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .spline import SmoothHermiteSpline
from .structured import CausalDelay, _batch


def design_lowpass(taps: int = 33, beta: float = 8.6) -> Tensor:
    if taps < 3 or taps % 2 == 0:
        raise ValueError("oversampling filter taps must be odd and at least three")
    index = torch.arange(taps, dtype=torch.float64) - (taps - 1) / 2
    cutoff = 0.25
    impulse = 2.0 * cutoff * torch.sinc(2.0 * cutoff * index)
    window = torch.kaiser_window(taps, periodic=False, beta=beta, dtype=torch.float64)
    impulse = impulse * window
    return (impulse / impulse.sum()).to(torch.float32)


class FixedCausalFIR(nn.Module):
    def __init__(self, coefficients: Tensor):
        super().__init__()
        if coefficients.ndim != 1 or len(coefficients) < 1:
            raise ValueError("coefficients must be one-dimensional and non-empty")
        self.register_buffer("coefficients", coefficients)
        self._state: Tensor | None = None

    @property
    def history(self) -> int:
        return len(self.coefficients) - 1

    def reset_state(self) -> None:
        self._state = None

    def forward(self, signal: Tensor) -> Tensor:
        return functional.conv1d(
            functional.pad(signal[:, None], (self.history, 0)),
            self.coefficients[None, None],
        )[:, 0]

    def stream(self, signal: Tensor) -> Tensor:
        if self._state is None:
            self._state = signal.new_zeros((len(signal), self.history))
        if self._state.shape[0] != signal.shape[0]:
            raise ValueError("stream batch size changed without reset")
        joined = torch.cat((self._state, signal), dim=-1)
        output = functional.conv1d(joined[:, None], self.coefficients[None, None])[:, 0]
        self._state = joined[..., -self.history :] if self.history else joined[..., :0]
        return output


class LocalOversampledSpline2x(nn.Module):
    """Oversample only phi(x)-x and add it to a causally delayed linear path."""

    factor = 2

    def __init__(self, num_knots: int = 17, filter_taps: int = 33):
        super().__init__()
        lowpass = design_lowpass(filter_taps)
        self.upsample_filter = FixedCausalFIR(2.0 * lowpass)
        self.downsample_filter = FixedCausalFIR(lowpass)
        self.spline = SmoothHermiteSpline(num_knots)
        self.latency_samples = (filter_taps - 1) // 2
        self.linear_delay = CausalDelay(self.latency_samples)

    def reset_state(self) -> None:
        self.upsample_filter.reset_state()
        self.downsample_filter.reset_state()
        self.linear_delay.reset_state()

    @staticmethod
    def _zero_insert(signal: Tensor) -> Tensor:
        high_rate = signal.new_zeros((len(signal), 2 * signal.shape[-1]))
        high_rate[:, ::2] = signal
        return high_rate

    def _run(self, signal: Tensor, *, streaming: bool) -> Tensor:
        batched, scalar = _batch(signal)
        high_rate = self._zero_insert(batched)
        upsampled = (
            self.upsample_filter.stream(high_rate)
            if streaming
            else self.upsample_filter(high_rate)
        )
        nonlinear_residual = self.spline(upsampled) - upsampled
        filtered_residual = (
            self.downsample_filter.stream(nonlinear_residual)
            if streaming
            else self.downsample_filter(nonlinear_residual)
        )
        residual = filtered_residual[:, ::2]
        linear = (
            self.linear_delay.stream(batched)
            if streaming
            else self.linear_delay(batched)
        )
        output = linear + residual
        return output[0] if scalar else output

    def forward(self, signal: Tensor) -> Tensor:
        return self._run(signal, streaming=False)

    def stream(self, signal: Tensor) -> Tensor:
        return self._run(signal, streaming=True)

    def curvature_penalty(self) -> Tensor:
        return self.spline.curvature_penalty()
