"""Frozen open comparator adapters for AMP-QUALITY-TEACHER-v1."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from .quality_teacher import CausalDenseConv1d
from .sota_comparators import build_sota_comparator
from .structured import _batch

QUALITY_TEACHER_COMPARATORS = (
    "nablafx_s4_tfilm_large",
    "nam_a2_full",
    "wavenet_dense_16x18",
)
WAVENET_16X18_DILATIONS = tuple(2**index for _ in range(2) for index in range(9))


class DenseWaveNetComparatorBlock(nn.Module):
    """One ungated upstream residual block with a summed skip head."""

    def __init__(
        self, channels: int, head_channels: int, kernel_size: int, dilation: int
    ) -> None:
        super().__init__()
        self.dilated = CausalDenseConv1d(channels, channels, kernel_size, dilation)
        self.residual_projection = nn.Conv1d(channels, channels, 1)
        self.skip_projection = nn.Conv1d(channels, head_channels, 1)

    def reset_state(self) -> None:
        self.dilated.reset_state()

    def _run(self, signal: Tensor, *, streaming: bool) -> tuple[Tensor, Tensor]:
        hidden = self.dilated.stream(signal) if streaming else self.dilated(signal)
        hidden = torch.tanh(hidden)
        return signal + self.residual_projection(hidden), self.skip_projection(hidden)

    def forward(self, signal: Tensor) -> tuple[Tensor, Tensor]:
        return self._run(signal, streaming=False)

    def stream(self, signal: Tensor) -> tuple[Tensor, Tensor]:
        return self._run(signal, streaming=True)


class DenseWaveNetComparatorStack(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: int,
        head_channels: int,
        dilations: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(input_channels, channels, 1)
        self.blocks = nn.ModuleList(
            DenseWaveNetComparatorBlock(channels, head_channels, 3, dilation)
            for dilation in dilations
        )

    def reset_state(self) -> None:
        for block in self.blocks:
            block.reset_state()

    def _run(self, signal: Tensor, *, streaming: bool) -> tuple[Tensor, Tensor]:
        hidden = self.input_projection(signal)
        skip_total: Tensor | None = None
        for block in self.blocks:
            hidden, skip = block.stream(hidden) if streaming else block(hidden)
            skip_total = skip if skip_total is None else skip_total + skip
        if skip_total is None:
            raise RuntimeError("dense WaveNet stack has no residual blocks")
        return hidden, skip_total

    def forward(self, signal: Tensor) -> tuple[Tensor, Tensor]:
        return self._run(signal, streaming=False)

    def stream(self, signal: Tensor) -> tuple[Tensor, Tensor]:
        return self._run(signal, streaming=True)


class DenseWaveNet16x18(nn.Module):
    """Exact unpruned 21,913-parameter topology from arXiv:2607.10086."""

    sample_rate_hz = 48_000
    latency_samples = 0
    channels = 16
    kernel_size = 3
    dilations = WAVENET_16X18_DILATIONS

    def __init__(self) -> None:
        super().__init__()
        half = len(self.dilations) // 2
        self.stacks = nn.ModuleList(
            (
                DenseWaveNetComparatorStack(
                    1, self.channels, self.channels, self.dilations[:half]
                ),
                DenseWaveNetComparatorStack(
                    self.channels, self.channels, 1, self.dilations[half:]
                ),
            )
        )
        self.receptive_field = 1 + (self.kernel_size - 1) * sum(self.dilations)

    def reset_state(self) -> None:
        for stack in self.stacks:
            stack.reset_state()

    def detach_stream_state(self) -> None:
        for module in self.modules():
            state = getattr(module, "_stream_state", None)
            if isinstance(state, Tensor):
                module._stream_state = state.detach()

    def _run(self, signal: Tensor, *, streaming: bool) -> Tensor:
        batched, scalar = _batch(signal)
        hidden = batched[:, None]
        head: Tensor | None = None
        for stack in self.stacks:
            hidden, head = stack.stream(hidden) if streaming else stack(hidden)
        if head is None:
            raise RuntimeError("dense WaveNet comparator has no output head")
        output = head[:, 0]
        return output[0] if scalar else output

    def forward(self, signal: Tensor) -> Tensor:
        return self._run(signal, streaming=False)

    def stream(self, signal: Tensor) -> Tensor:
        return self._run(signal, streaming=True)


def build_quality_teacher_comparator(
    family: str, *, root: Path | None = None
) -> nn.Module:
    """Build exactly one registered open comparator."""
    if family == "nablafx_s4_tfilm_large":
        return build_sota_comparator("nablafx_s4_tfilm", variant="large")
    if family == "nam_a2_full":
        return build_sota_comparator("nam_a2_full", variant="official", root=root)
    if family == "wavenet_dense_16x18":
        return DenseWaveNet16x18()
    raise ValueError("unknown AMP-QUALITY-TEACHER-v1 comparator")
