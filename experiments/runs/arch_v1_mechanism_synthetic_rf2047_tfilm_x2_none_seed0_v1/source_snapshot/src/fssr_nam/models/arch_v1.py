"""Causal multirate building blocks for AMP-QUALITY-ARCH-v1."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .oversampling import FixedCausalFIR, design_resampling_lowpass
from .r1 import CausalDepthwiseSeparableConv1d
from .structured import CausalDelay, _batch


class CausalBlockFeatureBus(nn.Module):
    """Six dry-only physical proxies updated after each completed block."""

    feature_names = (
        "fast_envelope",
        "slow_envelope",
        "dc_bias",
        "low_band_energy",
        "high_band_energy",
        "crest",
    )

    def __init__(self, decimation: int = 64, slow_decay: float = 0.95) -> None:
        super().__init__()
        if decimation < 1:
            raise ValueError("decimation must be positive")
        if not 0.0 <= slow_decay < 1.0:
            raise ValueError("slow_decay must be in [0, 1)")
        self.decimation = int(decimation)
        self.slow_decay = float(slow_decay)
        self._state: tuple[Tensor, ...] | None = None

    def reset_state(self) -> None:
        self._state = None

    @staticmethod
    def _initial(signal: Tensor) -> tuple[Tensor, ...]:
        batch = len(signal)
        scalar = signal.new_zeros(batch)
        features = signal.new_zeros(batch, 6)
        return (
            signal.new_empty((batch, 0)),
            scalar.clone(),
            features,
        )

    def _run(
        self, signal: Tensor, state: tuple[Tensor, ...], count: int
    ) -> tuple[Tensor, tuple[Tensor, ...], int]:
        partial, slow_energy, current = state
        outputs: list[Tensor] = []
        position = 0
        while position < signal.shape[-1]:
            take = min(self.decimation - count, signal.shape[-1] - position)
            outputs.append(current[..., None].expand(-1, -1, take))
            chunk = signal[:, position : position + take]
            partial = torch.cat((partial, chunk), dim=-1)
            position += take
            count += take
            if count == self.decimation:
                absolute = partial.abs()
                mean_abs = absolute.mean(dim=-1)
                energy = partial.square().mean(dim=-1)
                mean = partial.mean(dim=-1)
                peak = absolute.amax(dim=-1)
                slow_energy = (
                    self.slow_decay * slow_energy + (1.0 - self.slow_decay) * energy
                )
                low_energy = mean.square()
                high_energy = torch.clamp_min(energy - low_energy, 0.0)
                crest = peak / torch.sqrt(energy + 1.0e-8)
                current = torch.stack(
                    (
                        mean_abs,
                        torch.sqrt(slow_energy + 1.0e-8),
                        mean,
                        low_energy,
                        high_energy,
                        crest,
                    ),
                    dim=1,
                )
                partial = signal.new_empty((len(signal), 0))
                count = 0
        next_state = (partial, slow_energy, current)
        return torch.cat(outputs, dim=-1), next_state, count

    def forward(self, signal: Tensor) -> Tensor:
        batched, scalar = _batch(signal)
        output, _, _ = self._run(batched, self._initial(batched), 0)
        return output[0] if scalar else output

    def stream(self, signal: Tensor) -> Tensor:
        batched, scalar = _batch(signal)
        if self._state is None:
            self._state = (
                *self._initial(batched),
                batched.new_tensor(0, dtype=torch.int64),
            )
        state, count_tensor = self._state[:-1], self._state[-1]
        if state[0].shape[0] != len(batched):
            raise ValueError("stream batch size changed without reset")
        output, next_state, count = self._run(batched, state, int(count_tensor.item()))
        self._state = (
            *(value.detach() for value in next_state),
            batched.new_tensor(count, dtype=torch.int64),
        )
        return output[0] if scalar else output


class CausalSelectiveObserver(nn.Module):
    """Reduced-rate diagonal selective state-space observer.

    This is an independent PyTorch implementation of the selective recurrence
    used by the frozen S6 literature comparator. It deliberately runs only at
    completed causal feature blocks and exposes no target-derived input.
    """

    def __init__(
        self,
        feature_dim: int,
        state_dim: int,
        modulation_dim: int,
        *,
        decimation: int = 64,
        auxiliary_dim: int = 6,
    ) -> None:
        super().__init__()
        if min(feature_dim, state_dim, modulation_dim, decimation) < 1:
            raise ValueError("observer dimensions and decimation must be positive")
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.modulation_dim = modulation_dim
        self.decimation = decimation
        self.input_projection = nn.Linear(feature_dim, state_dim)
        self.delta_projection = nn.Linear(feature_dim, state_dim)
        self.read_projection = nn.Linear(feature_dim, state_dim)
        self.log_a = nn.Parameter(torch.linspace(-4.0, -1.0, state_dim))
        self.modulation_projection = nn.Linear(state_dim, modulation_dim)
        self.auxiliary_projection = nn.Linear(state_dim, auxiliary_dim)
        nn.init.zeros_(self.modulation_projection.weight)
        nn.init.zeros_(self.modulation_projection.bias)
        self._hidden: Tensor | None = None
        self._modulation: Tensor | None = None
        self._auxiliary: Tensor | None = None
        self._count = 0

    def reset_state(self) -> None:
        self._hidden = None
        self._modulation = None
        self._auxiliary = None
        self._count = 0

    def _update(self, features: Tensor, hidden: Tensor) -> Tensor:
        delta = functional.softplus(self.delta_projection(features)) + 1.0e-4
        transition = -torch.exp(self.log_a)[None, :]
        decay = torch.exp(delta * transition)
        candidate = torch.tanh(self.input_projection(features))
        return decay * hidden + (1.0 - decay) * candidate

    def _read(self, features: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        gate = torch.sigmoid(self.read_projection(features))
        observed = gate * hidden
        modulation = self.modulation_projection(observed)
        auxiliary = self.auxiliary_projection(observed)
        return modulation, auxiliary

    def _run(
        self,
        features: Tensor,
        hidden: Tensor,
        current_modulation: Tensor,
        current_auxiliary: Tensor,
        count: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        if features.ndim != 3 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must have shape (batch,{self.feature_dim},time)"
            )
        if features.shape[-1] < 1:
            raise ValueError("observer input must contain samples")
        modulations: list[Tensor] = []
        auxiliaries: list[Tensor] = []
        position = 0
        while position < features.shape[-1]:
            if count == 0:
                hidden = self._update(features[:, :, position], hidden)
                current_modulation, current_auxiliary = self._read(
                    features[:, :, position], hidden
                )
            take = min(self.decimation - count, features.shape[-1] - position)
            modulations.append(current_modulation[..., None].expand(-1, -1, take))
            auxiliaries.append(current_auxiliary[..., None].expand(-1, -1, take))
            position += take
            count = (count + take) % self.decimation
        return (
            torch.cat(modulations, dim=-1),
            torch.cat(auxiliaries, dim=-1),
            hidden,
            current_modulation,
            current_auxiliary,
            count,
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = features.new_zeros((len(features), self.state_dim))
        current_modulation = features.new_zeros((len(features), self.modulation_dim))
        current_auxiliary = features.new_zeros(
            (len(features), self.auxiliary_projection.out_features)
        )
        modulation, auxiliary, _, _, _, _ = self._run(
            features, hidden, current_modulation, current_auxiliary, 0
        )
        return modulation, auxiliary

    def stream(self, features: Tensor) -> tuple[Tensor, Tensor]:
        if self._hidden is None:
            self._hidden = features.new_zeros((len(features), self.state_dim))
            self._modulation = features.new_zeros((len(features), self.modulation_dim))
            self._auxiliary = features.new_zeros(
                (len(features), self.auxiliary_projection.out_features)
            )
        if self._hidden.shape[0] != len(features):
            raise ValueError("observer stream batch size changed without reset")
        assert self._modulation is not None and self._auxiliary is not None
        modulation, auxiliary, hidden, current_modulation, current_auxiliary, count = (
            self._run(
                features,
                self._hidden,
                self._modulation,
                self._auxiliary,
                self._count,
            )
        )
        self._hidden = hidden.detach()
        self._modulation = current_modulation.detach()
        self._auxiliary = current_auxiliary.detach()
        self._count = count
        return modulation, auxiliary


class CausalFeatureDelay(nn.Module):
    """Delay a `(batch, features, time)` control tensor exactly."""

    def __init__(self, samples: int) -> None:
        super().__init__()
        if samples < 0:
            raise ValueError("delay must be non-negative")
        self.samples = samples
        self.register_buffer("_state", torch.empty(0), persistent=False)

    def reset_state(self) -> None:
        self._state = self._state.new_empty(0)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape (batch,features,time)")
        if self.samples == 0:
            return features
        return functional.pad(features, (self.samples, 0))[..., : -self.samples]

    def stream(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape (batch,features,time)")
        if self.samples == 0:
            return features
        if self._state.numel() == 0:
            self._state = features.new_zeros(
                (features.shape[0], features.shape[1], self.samples)
            )
        if self._state.shape[:2] != features.shape[:2]:
            raise ValueError("feature stream shape changed without reset")
        joined = torch.cat((self._state, features), dim=-1)
        output = joined[..., : features.shape[-1]]
        self._state = joined[..., -self.samples :].detach()
        return output


class TFiLMDepthwiseBlock(nn.Module):
    """Depthwise-separable causal block modulated at every representation."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.channels = channels
        self.convolution = CausalDepthwiseSeparableConv1d(
            channels, kernel_size, dilation
        )

    def reset_state(self) -> None:
        self.convolution.reset_state()

    def _run(self, hidden: Tensor, modulation: Tensor, *, streaming: bool) -> Tensor:
        if modulation.shape != (len(hidden), 2 * self.channels, hidden.shape[-1]):
            raise ValueError("TFiLM modulation shape is invalid")
        update = (
            self.convolution.stream(hidden) if streaming else self.convolution(hidden)
        )
        gamma_raw, beta_raw = modulation.chunk(2, dim=1)
        gamma = 1.0 + 0.25 * torch.tanh(gamma_raw)
        beta = 0.10 * torch.tanh(beta_raw)
        return hidden + torch.tanh(gamma * update + beta)

    def forward(self, hidden: Tensor, modulation: Tensor) -> Tensor:
        return self._run(hidden, modulation, streaming=False)

    def stream(self, hidden: Tensor, modulation: Tensor) -> Tensor:
        return self._run(hidden, modulation, streaming=True)


class ModulatedMicroTCN(nn.Module):
    """Identity-initialized x2 nonlinear branch with physical RF 2047."""

    def __init__(self, channels: int = 8, dilation_scale: int = 2) -> None:
        super().__init__()
        if channels < 1 or dilation_scale < 1:
            raise ValueError("channels and dilation_scale must be positive")
        self.channels = channels
        base_dilations = (1, 4, 16, 64, 256)
        self.dilations = tuple(dilation_scale * value for value in base_dilations)
        self.input_projection = nn.Conv1d(1, channels, 1)
        self.blocks = nn.ModuleList(
            TFiLMDepthwiseBlock(channels, 7, dilation) for dilation in self.dilations
        )
        self.output_projection = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.residual_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        self.receptive_field = 1 + 6 * sum(self.dilations)

    @property
    def residual_scale(self) -> Tensor:
        return 0.5 * torch.sigmoid(self.residual_logit)

    @property
    def estimated_macs_per_internal_sample(self) -> int:
        projection = self.channels
        block = self.channels * 7 + self.channels**2
        return projection + len(self.blocks) * block + self.channels

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
            raise ValueError("micro-TCN modulation shape is invalid")
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
        output = batched + residual
        return output[0] if scalar else output

    def forward_modulated(self, signal: Tensor, modulation: Tensor) -> Tensor:
        return self._run(signal, modulation, streaming=False)

    def stream_modulated(self, signal: Tensor, modulation: Tensor) -> Tensor:
        return self._run(signal, modulation, streaming=True)


class ConditionedFullRateIsland(nn.Module):
    """One x2 resampler pair around a modulated nonlinear branch."""

    factor = 2

    def __init__(
        self,
        branch: nn.Module,
        *,
        latency_samples: int = 32,
        beta: float = 8.6,
    ) -> None:
        super().__init__()
        if latency_samples < 2 or latency_samples % 2:
            raise ValueError("conditioned x2 latency must be positive and even")
        lowpass = design_resampling_lowpass(
            self.factor, self.factor * latency_samples + 1, beta
        )
        self.branch = branch
        self.latency_samples = latency_samples
        self.upsample_filter = FixedCausalFIR(self.factor * lowpass)
        self.downsample_filter = FixedCausalFIR(lowpass)
        self.linear_delay = CausalDelay(latency_samples)
        self.modulation_delay = CausalFeatureDelay(latency_samples // 2)

    def reset_state(self) -> None:
        self.upsample_filter.reset_state()
        self.downsample_filter.reset_state()
        self.linear_delay.reset_state()
        self.modulation_delay.reset_state()
        self.branch.reset_state()

    @staticmethod
    def _zero_insert(signal: Tensor) -> Tensor:
        high_rate = signal.new_zeros((len(signal), 2 * signal.shape[-1]))
        high_rate[:, ::2] = signal
        return high_rate

    def _run(self, signal: Tensor, modulation: Tensor, *, streaming: bool) -> Tensor:
        batched, scalar = _batch(signal)
        if modulation.ndim == 2:
            modulation = modulation[None]
        if (
            modulation.shape[0] != len(batched)
            or modulation.shape[-1] != batched.shape[-1]
        ):
            raise ValueError("base-rate modulation shape is invalid")
        high_rate = self._zero_insert(batched)
        interpolated = (
            self.upsample_filter.stream(high_rate)
            if streaming
            else self.upsample_filter(high_rate)
        )
        aligned = (
            self.modulation_delay.stream(modulation)
            if streaming
            else self.modulation_delay(modulation)
        )
        high_modulation = aligned.repeat_interleave(2, dim=-1)
        branch_output = (
            self.branch.stream_modulated(interpolated, high_modulation)
            if streaming
            else self.branch.forward_modulated(interpolated, high_modulation)
        )
        nonlinear_residual = branch_output - interpolated
        filtered = (
            self.downsample_filter.stream(nonlinear_residual)
            if streaming
            else self.downsample_filter(nonlinear_residual)
        )
        residual = filtered[:, ::2]
        linear = (
            self.linear_delay.stream(batched)
            if streaming
            else self.linear_delay(batched)
        )
        output = linear + residual
        return output[0] if scalar else output

    def forward_modulated(self, signal: Tensor, modulation: Tensor) -> Tensor:
        return self._run(signal, modulation, streaming=False)

    def stream_modulated(self, signal: Tensor, modulation: Tensor) -> Tensor:
        return self._run(signal, modulation, streaming=True)


@dataclass(frozen=True)
class PhysicsConditionedOutput:
    audio: Tensor
    state_prediction: Tensor
    dry_features: Tensor


class PhysicsConditionedTCN(nn.Module):
    """Ablatable dry-feature/S6 observer feeding a per-block TFiLM x2 core."""

    sample_rate_hz = 48_000
    internal_sample_rate = 96_000
    dilation_scale = 2
    latency_samples = 32
    aa_mode = "full_island_x2"

    def __init__(
        self,
        *,
        conditioning: str = "observer",
        channels: int = 8,
        observer_state_dim: int = 16,
        decimation: int = 64,
        cascade: bool = False,
    ) -> None:
        super().__init__()
        if conditioning not in {"none", "deterministic", "observer"}:
            raise ValueError("conditioning must be none, deterministic or observer")
        self.conditioning = conditioning
        self.channels = channels
        self.feature_bus = CausalBlockFeatureBus(decimation=decimation)
        modulation_dim = 2 * channels
        self.deterministic_projection: nn.Conv1d | None = None
        self.observer: CausalSelectiveObserver | None = None
        if conditioning == "deterministic":
            self.deterministic_projection = nn.Conv1d(6, modulation_dim, 1)
            nn.init.zeros_(self.deterministic_projection.weight)
            nn.init.zeros_(self.deterministic_projection.bias)
        elif conditioning == "observer":
            self.observer = CausalSelectiveObserver(
                6,
                observer_state_dim,
                modulation_dim,
                decimation=decimation,
                auxiliary_dim=6,
            )
        self.branch = (
            CascadeModulatedMicroTCN(channels=channels, dilation_scale=2)
            if cascade
            else ModulatedMicroTCN(channels=channels, dilation_scale=2)
        )
        self.island = ConditionedFullRateIsland(
            self.branch, latency_samples=self.latency_samples
        )

    @property
    def receptive_field_samples(self) -> int:
        return (self.branch.receptive_field + 1) // 2

    def reset_state(self) -> None:
        self.feature_bus.reset_state()
        if self.observer is not None:
            self.observer.reset_state()
        self.island.reset_state()

    def _conditioning(
        self, features: Tensor, *, streaming: bool
    ) -> tuple[Tensor, Tensor]:
        if self.conditioning == "none":
            modulation = features.new_zeros(
                (len(features), 2 * self.channels, features.shape[-1])
            )
            auxiliary = features.new_zeros(features.shape)
            return modulation, auxiliary
        if self.conditioning == "deterministic":
            assert self.deterministic_projection is not None
            modulation = self.deterministic_projection(features)
            auxiliary = features
            return modulation, auxiliary
        assert self.observer is not None
        return self.observer.stream(features) if streaming else self.observer(features)

    def _run(self, signal: Tensor, *, streaming: bool) -> PhysicsConditionedOutput:
        batched, scalar = _batch(signal)
        features = (
            self.feature_bus.stream(batched) if streaming else self.feature_bus(batched)
        )
        modulation, auxiliary = self._conditioning(features, streaming=streaming)
        audio = (
            self.island.stream_modulated(batched, modulation)
            if streaming
            else self.island.forward_modulated(batched, modulation)
        )
        if scalar:
            return PhysicsConditionedOutput(audio[0], auxiliary[0], features[0])
        return PhysicsConditionedOutput(audio, auxiliary, features)

    def forward_components(self, signal: Tensor) -> PhysicsConditionedOutput:
        return self._run(signal, streaming=False)

    def stream_components(self, signal: Tensor) -> PhysicsConditionedOutput:
        return self._run(signal, streaming=True)

    def forward(self, signal: Tensor) -> Tensor:
        return self.forward_components(signal).audio

    def stream(self, signal: Tensor) -> Tensor:
        return self.stream_components(signal).audio


class CascadeModulatedMicroTCN(nn.Module):
    """Two identity-initialized nonlinear TCN stages in one x2 island."""

    def __init__(self, channels: int = 6, dilation_scale: int = 2) -> None:
        super().__init__()
        self.channels = channels
        self.first = ModulatedMicroTCN(channels, dilation_scale)
        self.second = ModulatedMicroTCN(channels, dilation_scale)
        self.receptive_field = (
            self.first.receptive_field + self.second.receptive_field - 1
        )

    @property
    def estimated_macs_per_internal_sample(self) -> int:
        return (
            self.first.estimated_macs_per_internal_sample
            + self.second.estimated_macs_per_internal_sample
        )

    def reset_state(self) -> None:
        self.first.reset_state()
        self.second.reset_state()

    def forward_modulated(self, signal: Tensor, modulation: Tensor) -> Tensor:
        first = self.first.forward_modulated(signal, modulation)
        return self.second.forward_modulated(first, modulation)

    def stream_modulated(self, signal: Tensor, modulation: Tensor) -> Tensor:
        first = self.first.stream_modulated(signal, modulation)
        return self.second.stream_modulated(first, modulation)


class SelectiveS6Branch(nn.Module):
    """Smooth diagonal selective state-space nonlinear branch."""

    def __init__(self, state_dim: int = 12) -> None:
        super().__init__()
        if state_dim < 1:
            raise ValueError("state_dim must be positive")
        self.state_dim = state_dim
        self.input_projection = nn.Linear(1, state_dim)
        self.delta_projection = nn.Linear(1, state_dim)
        self.input_gate = nn.Linear(1, state_dim)
        self.read_gate = nn.Linear(1, state_dim)
        self.log_a = nn.Parameter(torch.linspace(-5.0, -1.0, state_dim))
        self.output_projection = nn.Linear(state_dim, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.residual_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        self._hidden: Tensor | None = None

    @property
    def residual_scale(self) -> Tensor:
        return 0.5 * torch.sigmoid(self.residual_logit)

    @property
    def estimated_macs_per_internal_sample(self) -> int:
        return 5 * self.state_dim

    def reset_state(self) -> None:
        self._hidden = None

    def _step(self, sample: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        value = sample[:, None]
        delta = functional.softplus(self.delta_projection(value)) + 1.0e-4
        transition = -torch.exp(self.log_a)[None, :]
        decay = torch.exp(delta * transition)
        driven = torch.tanh(self.input_projection(value)) * torch.sigmoid(
            self.input_gate(value)
        )
        hidden = decay * hidden + (1.0 - decay) * driven
        observed = torch.sigmoid(self.read_gate(value)) * hidden
        residual = self.output_projection(observed)[:, 0]
        return sample + self.residual_scale * torch.tanh(residual), hidden

    def _run(self, signal: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        outputs: list[Tensor] = []
        for index in range(signal.shape[-1]):
            output, hidden = self._step(signal[:, index], hidden)
            outputs.append(output)
        return torch.stack(outputs, dim=-1), hidden

    def forward_modulated(self, signal: Tensor, modulation: Tensor) -> Tensor:
        del modulation
        batched, scalar = _batch(signal)
        hidden = batched.new_zeros((len(batched), self.state_dim))
        output, _ = self._run(batched, hidden)
        return output[0] if scalar else output

    def stream_modulated(self, signal: Tensor, modulation: Tensor) -> Tensor:
        del modulation
        batched, scalar = _batch(signal)
        if self._hidden is None:
            self._hidden = batched.new_zeros((len(batched), self.state_dim))
        if self._hidden.shape[0] != len(batched):
            raise ValueError("S6 stream batch size changed without reset")
        output, hidden = self._run(batched, self._hidden)
        self._hidden = hidden.detach()
        return output[0] if scalar else output


class SelectiveS6X2(nn.Module):
    """Pure S6 candidate wrapped in the qualified x2 resampling route."""

    sample_rate_hz = 48_000
    internal_sample_rate = 96_000
    dilation_scale = 2
    latency_samples = 32
    aa_mode = "full_island_x2"

    def __init__(self, state_dim: int = 12) -> None:
        super().__init__()
        self.branch = SelectiveS6Branch(state_dim)
        self.island = ConditionedFullRateIsland(
            self.branch, latency_samples=self.latency_samples
        )

    def reset_state(self) -> None:
        self.island.reset_state()

    def _modulation(self, signal: Tensor) -> Tensor:
        batched, _ = _batch(signal)
        return batched.new_zeros((len(batched), 1, batched.shape[-1]))

    def forward(self, signal: Tensor) -> Tensor:
        return self.island.forward_modulated(signal, self._modulation(signal))

    def stream(self, signal: Tensor) -> Tensor:
        return self.island.stream_modulated(signal, self._modulation(signal))


DEPLOYMENT_PROFILES = {
    "slim": (12, 32, 48),
    "balanced": (16, 48, 64),
    "full": (24, 64, 96),
    "max": (32, 96, 128),
}


def build_arch_v1_candidate(
    family: str, *, teacher: bool = False, profile: str = "full"
) -> nn.Module:
    """Build one of the six frozen candidate families without hidden tuning."""
    if teacher and family not in {
        "micro_tcn_x2",
        "phys_s6_tcn_x2",
    }:
        raise ValueError("family is not eligible for an AMP-QUALITY-ARCH-v1 teacher")
    if profile not in DEPLOYMENT_PROFILES:
        raise ValueError("unknown AMP-QUALITY-ARCH-v1 deployment profile")
    profile_channels, profile_observer, profile_s6 = DEPLOYMENT_PROFILES[profile]
    channels = 48 if teacher else profile_channels
    observer_state = 128 if teacher else profile_observer
    if family == "selective_s6_x2":
        return SelectiveS6X2(state_dim=256 if teacher else profile_s6)
    if family == "micro_tcn_x2":
        return PhysicsConditionedTCN(
            conditioning="none",
            channels=channels,
            observer_state_dim=observer_state,
        )
    if family == "phys_s6_tcn_x2":
        model = PhysicsConditionedTCN(
            conditioning="observer",
            channels=channels,
            observer_state_dim=observer_state,
        )
        model.uses_auxiliary_state_supervision = True
        return model
    if family == "phys_det_tcn_x2":
        return PhysicsConditionedTCN(
            conditioning="deterministic",
            channels=channels,
            observer_state_dim=observer_state,
        )
    if family == "rf2047_tfilm_x2":
        model = PhysicsConditionedTCN(
            conditioning="observer",
            channels=channels,
            observer_state_dim=observer_state,
        )
        model.uses_auxiliary_state_supervision = False
        return model
    if family == "cascade_rf2047_tfilm_x2":
        return PhysicsConditionedTCN(
            conditioning="observer",
            channels=channels,
            observer_state_dim=observer_state,
            cascade=True,
        )
    raise ValueError("unknown AMP-QUALITY-ARCH-v1 candidate family")
