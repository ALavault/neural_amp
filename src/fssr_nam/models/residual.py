"""Small causal TCN used for the constrained fast residual."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class StreamingCausalConv1d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.history = (kernel_size - 1) * dilation
        self._state: Tensor | None = None

    def reset_state(self) -> None:
        self._state = None

    def forward(self, signal: Tensor) -> Tensor:
        return self.conv(functional.pad(signal, (self.history, 0)))

    def stream(self, signal: Tensor) -> Tensor:
        if self._state is None:
            self._state = signal.new_zeros(
                (signal.shape[0], signal.shape[1], self.history)
            )
        if self._state.shape[:2] != signal.shape[:2]:
            raise ValueError("stream shape changed without reset")
        joined = torch.cat((self._state, signal), dim=-1)
        output = self.conv(joined)
        self._state = joined[..., -self.history :] if self.history else joined[..., :0]
        return output


class FastResidualTCN(nn.Module):
    """Four-layer TCN with bounded residual scale and zero-output initialization."""

    def __init__(
        self,
        input_channels: int = 2,
        channels: int = 8,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(input_channels, channels, 1)
        self.layers = nn.ModuleList(
            StreamingCausalConv1d(channels, channels, kernel_size, dilation)
            for dilation in dilations
        )
        self.output_projection = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.scale_logit = nn.Parameter(torch.tensor(math.log(0.05 / 0.95)))
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)

    @property
    def residual_scale(self) -> Tensor:
        return 0.5 * torch.sigmoid(self.scale_logit)

    def reset_state(self) -> None:
        for layer in self.layers:
            layer.reset_state()

    def _run(self, features: Tensor, *, streaming: bool) -> Tensor:
        hidden = self.input_projection(features)
        for layer in self.layers:
            update = layer.stream(hidden) if streaming else layer(hidden)
            hidden = hidden + functional.leaky_relu(update, negative_slope=0.01)
        raw = self.output_projection(hidden)
        return self.residual_scale * torch.tanh(raw[:, 0])

    def forward(self, features: Tensor) -> Tensor:
        return self._run(features, streaming=False)

    def stream(self, features: Tensor) -> Tensor:
        return self._run(features, streaming=True)


def residual_energy_ratio(
    residual: Tensor, output: Tensor, epsilon: float = 1.0e-12
) -> Tensor:
    return residual.square().sum() / (output.square().sum() + epsilon)


def normalized_residual_penalty(
    residual: Tensor, target: Tensor, epsilon: float = 1.0e-12
) -> Tensor:
    return residual.square().sum() / (target.square().sum() + epsilon)
