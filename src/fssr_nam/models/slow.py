"""Causal reduced-rate controller for slow amplifier dynamics."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .structured import _batch


class SlowStateController(nn.Module):
    """Update a small GRU only after each completed causal decimation interval."""

    def __init__(self, hidden_size: int = 8, decimation: int = 64):
        super().__init__()
        if hidden_size < 1 or decimation < 1:
            raise ValueError("hidden_size and decimation must be positive")
        self.hidden_size = hidden_size
        self.decimation = decimation
        self.gru = nn.GRUCell(3, hidden_size)
        self.projection = nn.Linear(hidden_size, 3)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self._hidden: Tensor | None = None
        self._accumulator: Tensor | None = None
        self._count = 0

    def reset_state(self) -> None:
        self._hidden = None
        self._accumulator = None
        self._count = 0

    @staticmethod
    def _features(sample: Tensor) -> Tensor:
        return torch.stack((sample.abs(), sample.square(), sample), dim=-1)

    @staticmethod
    def _bounded_modulation(raw: Tensor) -> Tensor:
        return torch.stack(
            (
                torch.exp(0.25 * torch.tanh(raw[..., 0])),
                0.1 * torch.tanh(raw[..., 1]),
                torch.exp(0.25 * torch.tanh(raw[..., 2])),
            ),
            dim=-1,
        )

    def _run(
        self,
        signal: Tensor,
        hidden: Tensor,
        accumulator: Tensor,
        count: int,
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        outputs = []
        position = 0
        while position < signal.shape[-1]:
            take = min(self.decimation - count, signal.shape[-1] - position)
            raw = self.projection(hidden)
            modulation = self._bounded_modulation(raw)
            outputs.append(modulation[..., None].expand(-1, -1, take))
            chunk = signal[:, position : position + take]
            accumulator = accumulator + torch.stack(
                (chunk.abs().sum(-1), chunk.square().sum(-1), chunk.sum(-1)), dim=-1
            )
            position += take
            count += take
            if count == self.decimation:
                hidden = self.gru(accumulator / self.decimation, hidden)
                accumulator = torch.zeros_like(accumulator)
                count = 0
        return torch.cat(outputs, dim=-1), hidden, accumulator, count

    def forward(self, signal: Tensor) -> Tensor:
        batched, scalar = _batch(signal)
        hidden = batched.new_zeros((len(batched), self.hidden_size))
        accumulator = batched.new_zeros((len(batched), 3))
        output, _, _, _ = self._run(batched, hidden, accumulator, 0)
        return output[0] if scalar else output

    def stream(self, signal: Tensor) -> Tensor:
        batched, scalar = _batch(signal)
        if self._hidden is None:
            self._hidden = batched.new_zeros((len(batched), self.hidden_size))
            self._accumulator = batched.new_zeros((len(batched), 3))
        if self._hidden.shape[0] != batched.shape[0]:
            raise ValueError("stream batch size changed without reset")
        output, self._hidden, self._accumulator, self._count = self._run(
            batched, self._hidden, self._accumulator, self._count
        )
        return output[0] if scalar else output
