"""Causal recurrent baselines for cost-controlled comparisons."""

from __future__ import annotations

import torch
from torch import nn


class CausalGRUBaseline(nn.Module):
    """Single-layer mono GRU with explicit streaming state."""

    def __init__(self, hidden_size: int = 62) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)
        self.register_buffer("_stream_state", torch.empty(0), persistent=False)

    @property
    def latency_samples(self) -> int:
        return 0

    @property
    def estimated_macs_per_sample(self) -> int:
        return 3 * (self.hidden_size + self.hidden_size**2) + self.hidden_size

    def _sequence(self, signal: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if signal.ndim == 1:
            return signal[None, :, None], True
        if signal.ndim == 2:
            return signal[:, :, None], False
        raise ValueError("signal must have shape (time,) or (batch, time)")

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        sequence, squeeze = self._sequence(signal)
        hidden, _ = self.gru(sequence)
        result = self.output(hidden).squeeze(-1)
        return result.squeeze(0) if squeeze else result

    def reset_state(self) -> None:
        self._stream_state = torch.empty(
            0, device=self._stream_state.device, dtype=self._stream_state.dtype
        )

    def stream(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 1:
            raise ValueError("stream input must have shape (time,)")
        sequence = signal[None, :, None]
        state = None if self._stream_state.numel() == 0 else self._stream_state
        hidden, state = self.gru(sequence, state)
        self._stream_state = state.detach()
        return self.output(hidden).squeeze(0).squeeze(-1)
