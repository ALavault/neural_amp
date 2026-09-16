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


class CausalDelay(nn.Module):
    def __init__(self, samples: int):
        super().__init__()
        if samples < 0:
            raise ValueError("delay must be non-negative")
        self.samples = samples
        self._state: Tensor | None = None

    def reset_state(self) -> None:
        self._state = None

    def forward(self, signal: Tensor) -> Tensor:
        batched, scalar = _batch(signal)
        if self.samples == 0:
            output = batched
        else:
            output = functional.pad(batched, (self.samples, 0))[..., : -self.samples]
        return output[0] if scalar else output

    def stream(self, signal: Tensor) -> Tensor:
        batched, scalar = _batch(signal)
        if self.samples == 0:
            return signal
        if self._state is None:
            self._state = batched.new_zeros((len(batched), self.samples))
        if self._state.shape[0] != batched.shape[0]:
            raise ValueError("stream batch size changed without reset")
        joined = torch.cat((self._state, batched), dim=-1)
        output = joined[..., : batched.shape[-1]]
        self._state = joined[..., -self.samples :]
        return output[0] if scalar else output


class S0Structured(nn.Module):
    """One-branch pre-FIR, smooth waveshaper, and post-FIR model."""

    code = "S0"
    latency_samples = 0

    def __init__(
        self,
        taps: int = 17,
        num_knots: int = 17,
        shaper: nn.Module | None = None,
        spline_range: float = 2.0,
    ):
        super().__init__()
        self.pre = CausalFIR(taps)
        self.shaper = (
            shaper
            if shaper is not None
            else SmoothHermiteSpline(num_knots, -spline_range, spline_range)
        )
        self.post = CausalFIR(taps)
        self.latency_samples = int(getattr(self.shaper, "latency_samples", 0))
        self.gain_delay = CausalDelay(self.latency_samples)
        self.drive = nn.Parameter(torch.tensor(1.0))
        self.offset = nn.Parameter(torch.tensor(0.0))
        self.output_gain = nn.Parameter(torch.tensor(1.0))

    def reset_state(self) -> None:
        self.pre.reset_state()
        self.post.reset_state()
        self.gain_delay.reset_state()
        reset_shaper = getattr(self.shaper, "reset_state", None)
        if reset_shaper is not None:
            reset_shaper()

    def forward(self, signal: Tensor) -> Tensor:
        return self.forward_modulated(signal, None)

    def stream(self, signal: Tensor) -> Tensor:
        return self.stream_modulated(signal, None)

    def forward_modulated(self, signal: Tensor, modulation: Tensor | None) -> Tensor:
        return self._modulated(signal, modulation, streaming=False)

    def stream_modulated(self, signal: Tensor, modulation: Tensor | None) -> Tensor:
        return self._modulated(signal, modulation, streaming=True)

    def _modulated(
        self, signal: Tensor, modulation: Tensor | None, *, streaming: bool
    ) -> Tensor:
        batched, scalar = _batch(signal)
        filtered = self.pre.stream(batched) if streaming else self.pre(batched)
        if modulation is None:
            drive_factor = 1.0
            offset_delta = 0.0
            gain_factor = 1.0
        else:
            if scalar and modulation.ndim == 2:
                modulation = modulation[None, :, :]
            if modulation.shape != (len(batched), 3, batched.shape[-1]):
                raise ValueError("modulation must have shape (batch,3,samples)")
            drive_factor = modulation[:, 0]
            offset_delta = modulation[:, 1]
            gain_factor = modulation[:, 2]
            if self.latency_samples:
                gain_factor = (
                    self.gain_delay.stream(gain_factor)
                    if streaming
                    else self.gain_delay(gain_factor)
                )
        shaper = (
            self.shaper.stream
            if streaming and hasattr(self.shaper, "stream")
            else self.shaper
        )
        shaped = shaper(
            self.drive * drive_factor * filtered + self.offset + offset_delta
        )
        output = self.post.stream(shaped) if streaming else self.post(shaped)
        result = self.output_gain * gain_factor * output
        return result[0] if scalar else result

    def regularization(self) -> Tensor:
        return self.shaper.curvature_penalty()
