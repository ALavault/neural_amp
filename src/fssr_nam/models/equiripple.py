"""Sparse equiripple half-band FIR candidate for SOTA-PROTOTYPE-v1."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.signal import remez
from torch import Tensor, nn


def design_equiripple_halfband(taps: int = 49, transition_width: float = 0.1) -> Tensor:
    """Design a type-I half-band FIR and freeze its analytic zero phase."""
    if taps < 9 or taps % 4 != 1:
        raise ValueError("half-band taps must be at least nine and equal 1 mod 4")
    if not 0.0 < transition_width < 1.0:
        raise ValueError("transition width must be in (0, 1)")
    lower = 0.5 - transition_width / 2.0
    upper = 0.5 + transition_width / 2.0
    coefficients = remez(
        taps,
        [0.0, lower, upper, 1.0],
        [1.0, 0.0],
        fs=2.0,
    )
    center = taps // 2
    zero_phase = center % 2
    coefficients[zero_phase::2] = 0.0
    coefficients[center] = 0.5
    return torch.from_numpy(np.asarray(coefficients, dtype=np.float32))


class SparseHalfbandFIR(nn.Module):
    """Causal FIR evaluating only the nonzero half-band coefficients."""

    def __init__(self, coefficients: Tensor) -> None:
        super().__init__()
        if coefficients.ndim != 1 or len(coefficients) < 1:
            raise ValueError("coefficients must be one-dimensional and non-empty")
        active = torch.nonzero(coefficients != 0.0, as_tuple=False)[:, 0]
        self.register_buffer("coefficients", coefficients)
        self.register_buffer("active_indices", active)
        self._state: Tensor | None = None

    @property
    def history(self) -> int:
        return len(self.coefficients) - 1

    @property
    def multiply_count_per_sample(self) -> int:
        return len(self.active_indices)

    def reset_state(self) -> None:
        self._state = None

    def _filter(self, padded: Tensor) -> Tensor:
        windows = padded.unfold(-1, len(self.coefficients), 1)
        selected = windows.index_select(-1, self.active_indices)
        weights = self.coefficients.index_select(0, self.active_indices)
        return functional.linear(selected, weights)

    def forward(self, signal: Tensor) -> Tensor:
        if signal.ndim != 2:
            raise ValueError("half-band FIR expects [batch, time]")
        padded = functional.pad(signal, (self.history, 0))
        return self._filter(padded)

    def stream(self, signal: Tensor) -> Tensor:
        if signal.ndim != 2:
            raise ValueError("half-band FIR expects [batch, time]")
        if self._state is None:
            self._state = signal.new_zeros((len(signal), self.history))
        if len(self._state) != len(signal):
            raise ValueError("stream batch size changed without reset")
        joined = torch.cat((self._state, signal), dim=-1)
        output = self._filter(joined)
        self._state = joined[:, -self.history :]
        return output


def _validate_polyphase_coefficients(coefficients: Tensor) -> None:
    if coefficients.ndim != 1 or len(coefficients) < 3:
        raise ValueError("polyphase coefficients must be one-dimensional")
    if len(coefficients) % 2 != 1:
        raise ValueError("polyphase half-band coefficients must have odd length")
    if not torch.isfinite(coefficients).all():
        raise ValueError("polyphase coefficients must be finite")


class PolyphaseHalfbandInterpolator2x(nn.Module):
    """True two-phase x2 interpolation without materialized zero insertions."""

    factor = 2

    def __init__(self, coefficients: Tensor) -> None:
        super().__init__()
        _validate_polyphase_coefficients(coefficients)
        self.even_phase = SparseHalfbandFIR(2.0 * coefficients[::2])
        self.odd_phase = SparseHalfbandFIR(2.0 * coefficients[1::2])

    @property
    def active_multiply_count_per_input_sample(self) -> int:
        """Active phase multiplications needed for two output samples."""
        return (
            self.even_phase.multiply_count_per_sample
            + self.odd_phase.multiply_count_per_sample
        )

    def reset_state(self) -> None:
        self.even_phase.reset_state()
        self.odd_phase.reset_state()

    @staticmethod
    def _interleave(even: Tensor, odd: Tensor) -> Tensor:
        output = even.new_empty((len(even), 2 * even.shape[-1]))
        output[:, ::2] = even
        output[:, 1::2] = odd
        return output

    def forward(self, signal: Tensor) -> Tensor:
        if signal.ndim != 2 or signal.shape[-1] < 1:
            raise ValueError("polyphase interpolator expects nonempty [batch, time]")
        return self._interleave(
            self.even_phase(signal),
            self.odd_phase(signal),
        )

    def stream(self, signal: Tensor) -> Tensor:
        if signal.ndim != 2 or signal.shape[-1] < 1:
            raise ValueError("polyphase interpolator expects nonempty [batch, time]")
        return self._interleave(
            self.even_phase.stream(signal),
            self.odd_phase.stream(signal),
        )


class PolyphaseHalfbandDecimator2x(nn.Module):
    """True two-phase x2 decimation with arbitrary streaming block boundaries."""

    factor = 2

    def __init__(self, coefficients: Tensor) -> None:
        super().__init__()
        _validate_polyphase_coefficients(coefficients)
        self.even_phase = SparseHalfbandFIR(coefficients[::2])
        self.odd_phase = SparseHalfbandFIR(coefficients[1::2])
        self.register_buffer("_previous_odd", torch.empty(0), persistent=False)
        self._next_phase = 0

    @property
    def active_multiply_count_per_output_sample(self) -> int:
        """Active phase multiplications needed for one output sample."""
        return (
            self.even_phase.multiply_count_per_sample
            + self.odd_phase.multiply_count_per_sample
        )

    def reset_state(self) -> None:
        self.even_phase.reset_state()
        self.odd_phase.reset_state()
        self._previous_odd = self._previous_odd.new_empty(0)
        self._next_phase = 0

    def forward(self, signal: Tensor) -> Tensor:
        if signal.ndim != 2 or signal.shape[-1] < 1:
            raise ValueError("polyphase decimator expects nonempty [batch, time]")
        even = signal[:, ::2]
        odd = signal[:, 1::2]
        preceding_odd = torch.cat((torch.zeros_like(even[:, :1]), odd), dim=-1)
        preceding_odd = preceding_odd[:, : even.shape[-1]]
        return self.even_phase(even) + self.odd_phase(preceding_odd)

    def stream(self, signal: Tensor) -> Tensor:
        if signal.ndim != 2 or signal.shape[-1] < 1:
            raise ValueError("polyphase decimator expects nonempty [batch, time]")
        if self._previous_odd.numel() == 0:
            self._previous_odd = signal.new_zeros((len(signal), 1))
        elif len(self._previous_odd) != len(signal):
            raise ValueError("decimator stream batch size changed without reset")

        if self._next_phase == 0:
            even = signal[:, ::2]
            odd = signal[:, 1::2]
            preceding_odd = torch.cat((self._previous_odd, odd), dim=-1)
            preceding_odd = preceding_odd[:, : even.shape[-1]]
        else:
            odd = signal[:, ::2]
            even = signal[:, 1::2]
            preceding_odd = odd[:, : even.shape[-1]]

        if odd.shape[-1] > 0:
            self._previous_odd = odd[:, -1:]
        self._next_phase = (self._next_phase + signal.shape[-1]) % 2
        if even.shape[-1] == 0:
            return signal.new_empty((len(signal), 0))
        return self.even_phase.stream(even) + self.odd_phase.stream(preceding_odd)
