"""Representability-first amplifier ablations for AMP-QUALITY-ARCH-v3."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .arch_v1 import (
    DEPLOYMENT_PROFILES,
    CausalBlockFeatureBus,
    CausalSelectiveObserver,
    ConditionedFullRateIsland,
    TFiLMDepthwiseBlock,
)
from .structured import _batch

V3_FAMILIES = (
    "gainhead_micro_tcn_x2",
    "slow_state_micro_tcn_x2",
    "long_rf_tcn_x2",
)
SHORT_BASE_DILATIONS = (1, 4, 16, 64, 256)
LONG_BASE_DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def _inverse_softplus(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("softplus target must be finite and positive")
    return value + math.log(-math.expm1(-value))


class StableGainHeadMicroTCN(nn.Module):
    """Causal TCN with a non-saturating learned residual amplitude."""

    residual_scale_floor = 1.0e-4

    def __init__(
        self,
        *,
        channels: int,
        base_dilations: tuple[int, ...],
        initial_residual_scale: float,
        dilation_scale: int = 2,
    ) -> None:
        super().__init__()
        if channels < 1 or dilation_scale < 1:
            raise ValueError("channels and dilation scale must be positive")
        if not base_dilations or any(dilation < 1 for dilation in base_dilations):
            raise ValueError("base dilations must be positive")
        if initial_residual_scale <= self.residual_scale_floor:
            raise ValueError("initial residual scale is too small")
        self.channels = channels
        self.base_dilations = tuple(base_dilations)
        self.dilations = tuple(
            dilation_scale * dilation for dilation in self.base_dilations
        )
        self.input_projection = nn.Conv1d(1, channels, 1)
        self.blocks = nn.ModuleList(
            TFiLMDepthwiseBlock(channels, 7, dilation) for dilation in self.dilations
        )
        self.output_projection = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.dry_gain_delta = nn.Parameter(torch.tensor(0.0))
        scale_target = initial_residual_scale - self.residual_scale_floor
        self.residual_scale_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(scale_target))
        )
        self.receptive_field = 1 + 6 * sum(self.dilations)

    @property
    def dry_gain(self) -> Tensor:
        return 1.0 + 0.5 * torch.tanh(self.dry_gain_delta)

    @property
    def residual_scale(self) -> Tensor:
        return functional.softplus(self.residual_scale_raw) + self.residual_scale_floor

    @property
    def estimated_macs_per_internal_sample(self) -> int:
        projection = self.channels
        block = self.channels * 7 + self.channels**2
        return projection + len(self.blocks) * block + 2 * self.channels

    def reset_state(self) -> None:
        for block in self.blocks:
            block.reset_state()

    def _run(self, signal: Tensor, modulation: Tensor, *, streaming: bool) -> Tensor:
        batched, scalar = _batch(signal)
        if modulation.ndim == 2:
            modulation = modulation[None]
        if modulation.shape != (
            len(batched),
            2 * self.channels,
            batched.shape[-1],
        ):
            raise ValueError("v3 micro-TCN modulation shape is invalid")
        hidden = self.input_projection(batched[:, None])
        for block in self.blocks:
            hidden = (
                block.stream(hidden, modulation)
                if streaming
                else block(hidden, modulation)
            )
        residual = self.residual_scale * torch.tanh(
            self.output_projection(hidden)[:, 0]
        )
        output = self.dry_gain * batched + residual
        return output[0] if scalar else output

    def forward_modulated(self, signal: Tensor, modulation: Tensor) -> Tensor:
        return self._run(signal, modulation, streaming=False)

    def stream_modulated(self, signal: Tensor, modulation: Tensor) -> Tensor:
        return self._run(signal, modulation, streaming=True)


@dataclass(frozen=True)
class ArchV3Output:
    audio: Tensor
    state_prediction: Tensor
    dry_features: Tensor


class ArchV3Amplifier(nn.Module):
    """One v3 gain-head, stateful, or long-receptive-field amplifier."""

    sample_rate_hz = 48_000
    internal_sample_rate = 96_000
    latency_samples = 32
    aa_mode = "full_island_x2"

    def __init__(
        self,
        *,
        family: str,
        channels: int,
        initial_residual_scale: float,
        slow_state_units: int = 16,
        slow_update_samples: int = 64,
    ) -> None:
        super().__init__()
        if family not in V3_FAMILIES:
            raise ValueError("unknown AMP-QUALITY-ARCH-v3 family")
        self.family = family
        self.channels = channels
        self.slow_update_samples = slow_update_samples
        self.uses_slow_state = family == "slow_state_micro_tcn_x2"
        base_dilations = (
            LONG_BASE_DILATIONS if family == "long_rf_tcn_x2" else SHORT_BASE_DILATIONS
        )
        self.feature_bus: CausalBlockFeatureBus | None = None
        self.observer: CausalSelectiveObserver | None = None
        if self.uses_slow_state:
            self.feature_bus = CausalBlockFeatureBus(decimation=slow_update_samples)
            self.observer = CausalSelectiveObserver(
                6,
                slow_state_units,
                2 * channels,
                decimation=slow_update_samples,
                auxiliary_dim=6,
            )
        self.branch = StableGainHeadMicroTCN(
            channels=channels,
            base_dilations=base_dilations,
            initial_residual_scale=initial_residual_scale,
        )
        self.island = ConditionedFullRateIsland(
            self.branch, latency_samples=self.latency_samples
        )

    @property
    def receptive_field_samples(self) -> int:
        return (self.branch.receptive_field + 1) // 2

    @property
    def dry_gain(self) -> Tensor:
        return self.branch.dry_gain

    @property
    def residual_scale(self) -> Tensor:
        return self.branch.residual_scale

    def reset_state(self) -> None:
        if self.feature_bus is not None:
            self.feature_bus.reset_state()
        if self.observer is not None:
            self.observer.reset_state()
        self.island.reset_state()

    def detach_stream_state(self) -> None:
        """Cut persistent streaming tensors at a truncated-BPTT boundary."""
        for module in self.modules():
            for name in (
                "_state",
                "_stream_state",
                "_hidden",
                "_modulation",
                "_auxiliary",
            ):
                value = getattr(module, name, None)
                if isinstance(value, Tensor):
                    setattr(module, name, value.detach())
                elif isinstance(value, tuple):
                    setattr(
                        module,
                        name,
                        tuple(
                            item.detach() if isinstance(item, Tensor) else item
                            for item in value
                        ),
                    )

    def _conditioning(
        self, signal: Tensor, *, streaming: bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.feature_bus is None or self.observer is None:
            modulation = signal.new_zeros(
                (len(signal), 2 * self.channels, signal.shape[-1])
            )
            features = signal.new_zeros((len(signal), 6, signal.shape[-1]))
            return modulation, features, features
        features = (
            self.feature_bus.stream(signal) if streaming else self.feature_bus(signal)
        )
        modulation, auxiliary = (
            self.observer.stream(features) if streaming else self.observer(features)
        )
        return modulation, auxiliary, features

    def _run(self, signal: Tensor, *, streaming: bool) -> ArchV3Output:
        batched, scalar = _batch(signal)
        modulation, auxiliary, features = self._conditioning(
            batched, streaming=streaming
        )
        audio = (
            self.island.stream_modulated(batched, modulation)
            if streaming
            else self.island.forward_modulated(batched, modulation)
        )
        if scalar:
            return ArchV3Output(audio[0], auxiliary[0], features[0])
        return ArchV3Output(audio, auxiliary, features)

    def forward_components(self, signal: Tensor) -> ArchV3Output:
        return self._run(signal, streaming=False)

    def stream_components(self, signal: Tensor) -> ArchV3Output:
        return self._run(signal, streaming=True)

    def forward(self, signal: Tensor) -> Tensor:
        return self.forward_components(signal).audio

    def stream(self, signal: Tensor) -> Tensor:
        return self.stream_components(signal).audio


def build_arch_v3_candidate(
    family: str,
    *,
    profile: str = "max",
    initial_residual_scale: float = 0.5,
) -> ArchV3Amplifier:
    """Build one prospectively registered v3 ablation."""
    if profile not in DEPLOYMENT_PROFILES:
        raise ValueError("unknown AMP-QUALITY-ARCH-v3 deployment profile")
    channels = DEPLOYMENT_PROFILES[profile][0]
    return ArchV3Amplifier(
        family=family,
        channels=channels,
        initial_residual_scale=initial_residual_scale,
        slow_state_units=16,
        slow_update_samples=64,
    )
