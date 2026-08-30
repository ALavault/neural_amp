"""Prospective slow-state long-TCN prototype for SOTA-PROTOTYPE-v1."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .approximants import QuinticHermiteSpline, SafeRationalActivation
from .arch_v1 import (
    DEPLOYMENT_PROFILES,
    CausalBlockFeatureBus,
    CausalFeatureDelay,
    CausalSelectiveObserver,
)
from .arch_v3 import LONG_BASE_DILATIONS
from .equiripple import (
    PolyphaseHalfbandDecimator2x,
    PolyphaseHalfbandInterpolator2x,
    design_equiripple_halfband,
)
from .oversampling import FixedCausalFIR, design_resampling_lowpass
from .r1 import CausalDepthwiseSeparableConv1d
from .structured import CausalDelay, _batch

PROTOTYPE_ACTIVATIONS = (
    "tanh",
    "quintic_hermite_c2",
    "safe_rational_4_3",
)
PROTOTYPE_RESAMPLERS = (
    "kaiser_windowed_sinc",
    "equiripple_halfband_polyphase",
)
PROTOTYPE_SLOW_CONTROLS = (
    "causal_zero_order_hold",
    "causal_exponential_hold",
    "causal_slope_limited_hold",
)

_SLOW_CONTROL_ALIASES = {
    "zoh": "causal_zero_order_hold",
    "exponential": "causal_exponential_hold",
    "slope_limited": "causal_slope_limited_hold",
}


def _inverse_softplus(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("softplus target must be finite and positive")
    return value + math.log(-math.expm1(-value))


def _activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "quintic_hermite_c2":
        return QuinticHermiteSpline()
    if name == "safe_rational_4_3":
        return SafeRationalActivation()
    raise ValueError("unknown SOTA prototype activation")


class CausalSlowControl(nn.Module):
    """Post-observer causal hold with a zero neutral warm-up block."""

    def __init__(
        self,
        channels: int,
        mode: str,
        *,
        update_samples: int = 64,
        coefficient: float = 0.25,
    ) -> None:
        super().__init__()
        mode = _SLOW_CONTROL_ALIASES.get(mode, mode)
        if channels < 1 or update_samples < 1:
            raise ValueError("slow-control dimensions must be positive")
        if mode not in PROTOTYPE_SLOW_CONTROLS:
            raise ValueError("unknown SOTA prototype slow control")
        if not 0.0 < coefficient <= 1.0:
            raise ValueError("slow-control coefficient must be in (0, 1]")
        self.channels = channels
        self.protocol_name = mode
        self.update_samples = update_samples
        self.coefficient = coefficient
        self._state: Tensor | None = None
        self._processed = 0

    @property
    def mode(self) -> str:
        return {
            "causal_zero_order_hold": "zoh",
            "causal_exponential_hold": "exponential",
            "causal_slope_limited_hold": "slope_limited",
        }[self.protocol_name]

    def reset_state(self) -> None:
        self._state = None
        self._processed = 0

    def _validate(self, targets: Tensor) -> None:
        if (
            targets.ndim != 3
            or targets.shape[1] != self.channels
            or targets.shape[-1] < 1
        ):
            raise ValueError(
                "slow-control targets must be nonempty [batch,channels,time]"
            )

    def _run(
        self, targets: Tensor, state: Tensor, processed: int
    ) -> tuple[Tensor, Tensor, int]:
        if self.mode == "zoh":
            positions = torch.arange(targets.shape[-1], device=targets.device)
            valid = positions + processed >= self.update_samples
            output = torch.where(valid[None, None], targets, torch.zeros_like(targets))
            return (
                output,
                output[..., -1],
                min(self.update_samples, processed + targets.shape[-1]),
            )

        outputs: list[Tensor] = []
        for position, target in enumerate(targets.unbind(dim=-1)):
            if processed + position < self.update_samples:
                target = torch.zeros_like(target)
            delta = target - state
            if self.mode == "exponential":
                state = state + self.coefficient * delta
            else:
                limit = self.coefficient * (1.0 + state.abs())
                state = state + torch.clamp(delta, -limit, limit)
            outputs.append(state)
        return (
            torch.stack(outputs, dim=-1),
            state,
            min(self.update_samples, processed + targets.shape[-1]),
        )

    def forward(self, targets: Tensor) -> Tensor:
        self._validate(targets)
        state = targets.new_zeros((len(targets), self.channels))
        output, _, _ = self._run(targets, state, 0)
        return output

    def stream(self, targets: Tensor) -> Tensor:
        self._validate(targets)
        if self._state is None:
            self._state = targets.new_zeros((len(targets), self.channels))
        elif self._state.shape != (len(targets), self.channels):
            raise ValueError("slow-control stream shape changed without reset")
        output, self._state, self._processed = self._run(
            targets, self._state, self._processed
        )
        return output


class PrototypeTFiLMBlock(nn.Module):
    """Causal TFiLM block with one prospectively selected activation."""

    def __init__(self, channels: int, dilation: int, activation: str) -> None:
        super().__init__()
        self.channels = channels
        self.convolution = CausalDepthwiseSeparableConv1d(channels, 7, dilation)
        self.activation = _activation(activation)

    def reset_state(self) -> None:
        self.convolution.reset_state()

    def _run(self, hidden: Tensor, modulation: Tensor, *, streaming: bool) -> Tensor:
        if modulation.shape != (len(hidden), 2 * self.channels, hidden.shape[-1]):
            raise ValueError("prototype TFiLM modulation shape is invalid")
        update = (
            self.convolution.stream(hidden) if streaming else self.convolution(hidden)
        )
        gamma_raw, beta_raw = modulation.chunk(2, dim=1)
        gamma = 1.0 + 0.25 * torch.tanh(gamma_raw)
        beta = 0.10 * torch.tanh(beta_raw)
        return hidden + self.activation(gamma * update + beta)

    def forward(self, hidden: Tensor, modulation: Tensor) -> Tensor:
        return self._run(hidden, modulation, streaming=False)

    def stream(self, hidden: Tensor, modulation: Tensor) -> Tensor:
        return self._run(hidden, modulation, streaming=True)


class SlowLongTCNBranch(nn.Module):
    """Identity-initialized long-RF branch with shared slow conditioning."""

    residual_scale_floor = 1.0e-4

    def __init__(
        self,
        *,
        channels: int,
        activation: str,
        initial_residual_scale: float,
        dilation_scale: int = 2,
    ) -> None:
        super().__init__()
        if channels < 1 or dilation_scale < 1:
            raise ValueError("channels and dilation scale must be positive")
        if activation not in PROTOTYPE_ACTIVATIONS:
            raise ValueError("unknown SOTA prototype activation")
        if initial_residual_scale <= self.residual_scale_floor:
            raise ValueError("initial residual scale is too small")
        self.channels = channels
        self.activation_name = activation
        self.base_dilations = LONG_BASE_DILATIONS
        self.dilations = tuple(
            dilation_scale * dilation for dilation in self.base_dilations
        )
        self.input_projection = nn.Conv1d(1, channels, 1)
        self.blocks = nn.ModuleList(
            PrototypeTFiLMBlock(channels, dilation, activation)
            for dilation in self.dilations
        )
        self.output_projection = nn.Conv1d(channels, 1, 1)
        self.output_activation = _activation(activation)
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
            raise ValueError("prototype branch modulation shape is invalid")
        hidden = self.input_projection(batched[:, None])
        for block in self.blocks:
            hidden = (
                block.stream(hidden, modulation)
                if streaming
                else block(hidden, modulation)
            )
        residual = self.residual_scale * self.output_activation(
            self.output_projection(hidden)[:, 0]
        )
        output = self.dry_gain * batched + residual
        return output[0] if scalar else output

    def forward_modulated(self, signal: Tensor, modulation: Tensor) -> Tensor:
        return self._run(signal, modulation, streaming=False)

    def stream_modulated(self, signal: Tensor, modulation: Tensor) -> Tensor:
        return self._run(signal, modulation, streaming=True)


class PrototypeFullRateIsland(nn.Module):
    """Causal x2 island with the registered Kaiser or polyphase pair."""

    factor = 2

    def __init__(self, branch: nn.Module, *, resampler: str) -> None:
        super().__init__()
        if resampler == "kaiser_windowed_sinc":
            self.latency_samples = 32
            lowpass = design_resampling_lowpass(2, 65, 8.6)
            self.upsample_filter = FixedCausalFIR(2.0 * lowpass)
            self.downsample_filter = FixedCausalFIR(lowpass)
            self._uses_polyphase = False
            self.active_multiply_count_per_base_sample = 4 * len(lowpass)
        elif resampler == "equiripple_halfband_polyphase":
            self.latency_samples = 24
            lowpass = design_equiripple_halfband(taps=49)
            self.upsample_filter = PolyphaseHalfbandInterpolator2x(lowpass)
            self.downsample_filter = PolyphaseHalfbandDecimator2x(lowpass)
            self._uses_polyphase = True
            self.active_multiply_count_per_base_sample = (
                self.upsample_filter.active_multiply_count_per_input_sample
                + self.downsample_filter.active_multiply_count_per_output_sample
            )
        else:
            raise ValueError("unknown SOTA prototype resampler")
        self.branch = branch
        self.resampler_name = resampler
        self.linear_delay = CausalDelay(self.latency_samples)
        self.modulation_delay = CausalFeatureDelay(self.latency_samples // 2)

    def reset_state(self) -> None:
        self.upsample_filter.reset_state()
        self.downsample_filter.reset_state()
        self.linear_delay.reset_state()
        self.modulation_delay.reset_state()
        reset = getattr(self.branch, "reset_state", None)
        if reset is not None:
            reset()

    @staticmethod
    def _zero_insert(signal: Tensor) -> Tensor:
        high_rate = signal.new_zeros((len(signal), 2 * signal.shape[-1]))
        high_rate[:, ::2] = signal
        return high_rate

    def _run(self, signal: Tensor, modulation: Tensor, *, streaming: bool) -> Tensor:
        batched, scalar = _batch(signal)
        if modulation.ndim == 2:
            modulation = modulation[None]
        if modulation.shape != (
            len(batched),
            2 * self.branch.channels,
            batched.shape[-1],
        ):
            raise ValueError("prototype base-rate modulation shape is invalid")
        if self._uses_polyphase:
            interpolated = (
                self.upsample_filter.stream(batched)
                if streaming
                else self.upsample_filter(batched)
            )
        else:
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
        residual = filtered if self._uses_polyphase else filtered[:, ::2]
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
class SOTAPrototypeOutput:
    audio: Tensor
    state_prediction: Tensor
    dry_features: Tensor


def _detached(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detached(item) for item in value)
    if isinstance(value, list):
        return [_detached(item) for item in value]
    return value


class SOTAPrototypeAmplifier(nn.Module):
    """Slow-state conditioned, long-receptive-field prospective amplifier."""

    family = "slow_long_tcn_x2"
    sample_rate_hz = 48_000
    internal_sample_rate = 96_000
    aa_mode = "full_island_x2"

    def __init__(
        self,
        *,
        channels: int,
        activation: str,
        resampler: str,
        slow_control: str = "causal_zero_order_hold",
        initial_residual_scale: float = 0.5,
        slow_state_units: int = 16,
        slow_update_samples: int = 64,
    ) -> None:
        super().__init__()
        if slow_state_units < 1 or slow_update_samples < 1:
            raise ValueError("slow-state dimensions must be positive")
        self.channels = channels
        self.activation_name = activation
        self.resampler_name = resampler
        self.slow_update_samples = slow_update_samples
        self.feature_bus = CausalBlockFeatureBus(decimation=slow_update_samples)
        self.observer = CausalSelectiveObserver(
            6,
            slow_state_units,
            2 * channels,
            decimation=slow_update_samples,
            auxiliary_dim=6,
        )
        self.slow_control = CausalSlowControl(
            2 * channels,
            slow_control,
            update_samples=slow_update_samples,
        )
        self.slow_control_name = self.slow_control.protocol_name
        self.branch = SlowLongTCNBranch(
            channels=channels,
            activation=activation,
            initial_residual_scale=initial_residual_scale,
        )
        self.island = PrototypeFullRateIsland(self.branch, resampler=resampler)
        if self.latency_samples > 64:
            raise ValueError("prototype latency exceeds the registered ceiling")

    @property
    def latency_samples(self) -> int:
        return self.island.latency_samples

    @property
    def receptive_field_samples(self) -> int:
        return (self.branch.receptive_field + 1) // 2

    @property
    def residual_scale(self) -> Tensor:
        return self.branch.residual_scale

    @property
    def resampler_active_multiply_count_per_base_sample(self) -> int:
        return self.island.active_multiply_count_per_base_sample

    def reset_state(self) -> None:
        self.feature_bus.reset_state()
        self.observer.reset_state()
        self.slow_control.reset_state()
        self.island.reset_state()

    def reset(self) -> None:
        """Reset all persistent causal state."""
        self.reset_state()

    def detach_stream_state(self) -> None:
        """Cut persistent tensors at a truncated-BPTT boundary."""
        for module in self.modules():
            for name in (
                "_state",
                "_stream_state",
                "_hidden",
                "_modulation",
                "_auxiliary",
                "_previous_odd",
            ):
                value = getattr(module, name, None)
                if value is not None:
                    setattr(module, name, _detached(value))

    def detach(self) -> None:
        """Alias matching the prospective streaming interface."""
        self.detach_stream_state()

    def _run(self, signal: Tensor, *, streaming: bool) -> SOTAPrototypeOutput:
        batched, scalar = _batch(signal)
        if batched.shape[-1] < 1:
            raise ValueError("prototype input must contain samples")
        features = (
            self.feature_bus.stream(batched) if streaming else self.feature_bus(batched)
        )
        raw_modulation, auxiliary = (
            self.observer.stream(features) if streaming else self.observer(features)
        )
        modulation = (
            self.slow_control.stream(raw_modulation)
            if streaming
            else self.slow_control(raw_modulation)
        )
        audio = (
            self.island.stream_modulated(batched, modulation)
            if streaming
            else self.island.forward_modulated(batched, modulation)
        )
        if scalar:
            return SOTAPrototypeOutput(audio[0], auxiliary[0], features[0])
        return SOTAPrototypeOutput(audio, auxiliary, features)

    def forward_components(self, signal: Tensor) -> SOTAPrototypeOutput:
        return self._run(signal, streaming=False)

    def stream_components(self, signal: Tensor) -> SOTAPrototypeOutput:
        return self._run(signal, streaming=True)

    def forward(self, signal: Tensor) -> Tensor:
        return self.forward_components(signal).audio

    def stream(self, signal: Tensor) -> Tensor:
        return self.stream_components(signal).audio


def build_sota_prototype_candidate(
    *,
    activation: str = "tanh",
    resampler: str = "kaiser_windowed_sinc",
    slow_control: str = "causal_zero_order_hold",
    profile: str = "max",
    initial_residual_scale: float = 0.5,
) -> SOTAPrototypeAmplifier:
    """Build the single registered slow-long prototype anchor."""
    if profile not in DEPLOYMENT_PROFILES:
        raise ValueError("unknown SOTA prototype deployment profile")
    channels, observer_units, _ = DEPLOYMENT_PROFILES[profile]
    return SOTAPrototypeAmplifier(
        channels=channels,
        activation=activation,
        resampler=resampler,
        slow_control=slow_control,
        initial_residual_scale=initial_residual_scale,
        slow_state_units=observer_units,
        slow_update_samples=64,
    )
