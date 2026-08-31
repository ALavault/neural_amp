"""Causal quality-first teacher registered by AMP-QUALITY-TEACHER-v1."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .oversampling import FixedCausalFIR, design_resampling_lowpass
from .structured import CausalDelay, CausalFIR, _batch

QUALITY_TEACHER_FAMILY = "s4_tfilm_wavenet_x2_teacher"
QUALITY_TEACHER_FAST_CONTROL = "wavenet_x2_teacher_fast_only"
QUALITY_TEACHER_FAMILIES = (
    QUALITY_TEACHER_FAMILY,
    QUALITY_TEACHER_FAST_CONTROL,
)
WAVENET_DILATIONS = tuple(2**index for _ in range(2) for index in range(12))


def _detach(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach(item) for item in value)
    if isinstance(value, list):
        return [_detach(item) for item in value]
    return value


class CausalDenseConv1d(nn.Module):
    """Dense causal convolution with exact arbitrary-block stream state."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, kernel_size, dilation) < 1:
            raise ValueError("causal convolution dimensions must be positive")
        self.in_channels = in_channels
        self.history = (kernel_size - 1) * dilation
        self.convolution = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )
        self.register_buffer("_stream_state", torch.empty(0), persistent=False)

    def _validate(self, signal: Tensor) -> None:
        if (
            signal.ndim != 3
            or signal.shape[1] != self.in_channels
            or signal.shape[-1] < 1
        ):
            raise ValueError(
                f"signal must have shape (batch,{self.in_channels},nonempty_time)"
            )

    def reset_state(self) -> None:
        self._stream_state = self._stream_state.new_empty(0)

    def forward(self, signal: Tensor) -> Tensor:
        self._validate(signal)
        return self.convolution(functional.pad(signal, (self.history, 0)))

    def stream(self, signal: Tensor) -> Tensor:
        self._validate(signal)
        if self._stream_state.numel() == 0:
            self._stream_state = signal.new_zeros(
                (signal.shape[0], self.in_channels, self.history)
            )
        elif self._stream_state.shape[:2] != signal.shape[:2]:
            raise ValueError("stream shape changed without reset")
        joined = torch.cat((self._stream_state, signal), dim=-1)
        output = self.convolution(joined)
        self._stream_state = (
            joined[..., -self.history :] if self.history else joined[..., :0]
        )
        return output


class CausalTensorDelay(nn.Module):
    """Delay any tensor whose final axis is time."""

    def __init__(self, samples: int) -> None:
        super().__init__()
        if samples < 0:
            raise ValueError("delay must be non-negative")
        self.samples = samples
        self.register_buffer("_state", torch.empty(0), persistent=False)

    def reset_state(self) -> None:
        self._state = self._state.new_empty(0)

    def forward(self, signal: Tensor) -> Tensor:
        if signal.ndim < 2 or signal.shape[-1] < 1:
            raise ValueError("delayed tensor must have batch and nonempty time axes")
        if self.samples == 0:
            return signal
        return functional.pad(signal, (self.samples, 0))[..., : -self.samples]

    def stream(self, signal: Tensor) -> Tensor:
        if signal.ndim < 2 or signal.shape[-1] < 1:
            raise ValueError("delayed tensor must have batch and nonempty time axes")
        if self.samples == 0:
            return signal
        expected = (*signal.shape[:-1], self.samples)
        if self._state.numel() == 0:
            self._state = signal.new_zeros(expected)
        elif self._state.shape != expected:
            raise ValueError("delayed tensor stream shape changed without reset")
        joined = torch.cat((self._state, signal), dim=-1)
        output = joined[..., : signal.shape[-1]]
        self._state = joined[..., -self.samples :]
        return output


class StreamingDiagonalS4(nn.Module):
    """Exact diagonal S4 recurrence evaluated in bounded vectorized blocks."""

    def __init__(
        self,
        channels: int,
        state_dim: int,
        *,
        evaluation_block_samples: int = 1_024,
    ) -> None:
        super().__init__()
        if min(channels, state_dim, evaluation_block_samples) < 1:
            raise ValueError("S4 dimensions must be positive")
        self.channels = channels
        self.state_dim = state_dim
        self.evaluation_block_samples = evaluation_block_samples
        log_dt = torch.rand(channels) * (math.log(0.1) - math.log(0.001))
        self.log_dt = nn.Parameter(log_dt + math.log(0.001))
        complex_c = torch.randn(channels, state_dim, dtype=torch.cfloat)
        self.c_as_real = nn.Parameter(torch.view_as_real(complex_c))
        self.log_a_real = nn.Parameter(
            torch.log(torch.full((channels, state_dim), 0.5))
        )
        self.a_imag = nn.Parameter(
            math.pi * torch.arange(state_dim).repeat(channels, 1)
        )
        self.direct = nn.Parameter(torch.randn(channels))
        self.register_buffer("_stream_state", torch.empty(0), persistent=False)

    def reset_state(self) -> None:
        self._stream_state = self._stream_state.new_empty(0)

    def _discretized(self) -> tuple[Tensor, Tensor, Tensor]:
        dt = self.log_dt.exp()
        c = torch.view_as_complex(self.c_as_real)
        a = -self.log_a_real.exp() + 1j * self.a_imag
        transition = (a * dt[:, None]).exp()
        input_gain = (transition - 1.0) / a
        return transition, input_gain, c

    def _run_block(self, signal: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        samples = signal.shape[-1]
        transition, input_gain, c = self._discretized()
        exponents = torch.arange(samples, device=signal.device)
        powers = transition[..., None].pow(exponents)
        kernel = 2.0 * (c * input_gain)[..., None].mul(powers).real.sum(dim=1)
        fft_size = 2 * samples
        response = torch.fft.irfft(
            torch.fft.rfft(signal, n=fft_size)
            * torch.fft.rfft(kernel, n=fft_size)[None],
            n=fft_size,
        )[..., :samples]
        initial_powers = transition[..., None] * powers
        initial = 2.0 * torch.einsum("bcn,cnt,cn->bct", state, initial_powers, c).real
        output = response + initial + signal * self.direct[None, :, None]
        weighted_input = torch.einsum(
            "bct,cnt->bcn", signal.to(powers.dtype), powers.flip(-1)
        )
        final_state = (
            transition.pow(samples)[None] * state + input_gain[None] * weighted_input
        )
        return output, final_state

    def _validate(self, signal: Tensor) -> None:
        if signal.ndim != 3 or signal.shape[1] != self.channels or signal.shape[-1] < 1:
            raise ValueError(
                f"S4 input must have shape (batch,{self.channels},nonempty_time)"
            )

    def _zero_state(self, signal: Tensor) -> Tensor:
        dtype = torch.complex64 if signal.dtype == torch.float32 else torch.complex128
        return torch.zeros(
            signal.shape[0],
            self.channels,
            self.state_dim,
            dtype=dtype,
            device=signal.device,
        )

    def _run(self, signal: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        outputs: list[Tensor] = []
        for start in range(0, signal.shape[-1], self.evaluation_block_samples):
            stop = min(start + self.evaluation_block_samples, signal.shape[-1])
            output, state = self._run_block(signal[..., start:stop], state)
            outputs.append(output)
        return torch.cat(outputs, dim=-1), state

    def forward(self, signal: Tensor) -> Tensor:
        self._validate(signal)
        output, _ = self._run(signal, self._zero_state(signal))
        return output

    def stream(self, signal: Tensor) -> Tensor:
        self._validate(signal)
        if self._stream_state.numel() == 0:
            self._stream_state = self._zero_state(signal)
        elif self._stream_state.shape[:2] != signal.shape[:2]:
            raise ValueError("S4 stream shape changed without reset")
        output, self._stream_state = self._run(signal, self._stream_state)
        return output


class S4ObserverBlock(nn.Module):
    def __init__(self, channels: int, state_dim: int, evaluation_block_samples: int):
        super().__init__()
        self.mix = nn.Conv1d(channels, channels, 1)
        self.s4 = StreamingDiagonalS4(
            channels,
            state_dim,
            evaluation_block_samples=evaluation_block_samples,
        )
        self.output = nn.Conv1d(channels, channels, 1)

    def reset_state(self) -> None:
        self.s4.reset_state()

    def _run(self, signal: Tensor, *, streaming: bool) -> Tensor:
        hidden = torch.tanh(self.mix(signal))
        hidden = self.s4.stream(hidden) if streaming else self.s4(hidden)
        return signal + torch.tanh(self.output(hidden))

    def forward(self, signal: Tensor) -> Tensor:
        return self._run(signal, streaming=False)

    def stream(self, signal: Tensor) -> Tensor:
        return self._run(signal, streaming=True)


class CausalS4FiLMObserver(nn.Module):
    """Eight-block dry-audio S4 observer producing per-WaveNet-layer FiLM."""

    def __init__(
        self,
        *,
        channels: int,
        state_dim: int,
        blocks: int,
        wavenet_layers: int,
        wavenet_channels: int,
        evaluation_block_samples: int = 1_024,
    ) -> None:
        super().__init__()
        if min(channels, state_dim, blocks, wavenet_layers, wavenet_channels) < 1:
            raise ValueError("observer dimensions must be positive")
        self.channels = channels
        self.state_dim = state_dim
        self.block_count = blocks
        self.wavenet_layers = wavenet_layers
        self.wavenet_channels = wavenet_channels
        self.input_projection = nn.Conv1d(1, channels, 1)
        self.blocks = nn.ModuleList(
            S4ObserverBlock(channels, state_dim, evaluation_block_samples)
            for _ in range(blocks)
        )
        self.film_projection = nn.Conv1d(
            channels, wavenet_layers * 2 * wavenet_channels, 1
        )
        nn.init.zeros_(self.film_projection.weight)
        nn.init.zeros_(self.film_projection.bias)

    def reset_state(self) -> None:
        for block in self.blocks:
            block.reset_state()

    def _run(self, signal: Tensor, *, streaming: bool) -> Tensor:
        if signal.ndim != 2 or signal.shape[-1] < 1:
            raise ValueError("observer dry input must have shape (batch,nonempty_time)")
        hidden = self.input_projection(signal[:, None])
        for block in self.blocks:
            hidden = block.stream(hidden) if streaming else block(hidden)
        film = self.film_projection(hidden)
        return film.reshape(
            len(signal),
            self.wavenet_layers,
            2 * self.wavenet_channels,
            signal.shape[-1],
        )

    def forward(self, signal: Tensor) -> Tensor:
        return self._run(signal, streaming=False)

    def stream(self, signal: Tensor) -> Tensor:
        return self._run(signal, streaming=True)


class GatedWaveNetBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.channels = channels
        self.dilated = CausalDenseConv1d(channels, 2 * channels, 3, dilation)
        self.residual_projection = nn.Conv1d(channels, channels, 1)

    def reset_state(self) -> None:
        self.dilated.reset_state()

    def _run(self, signal: Tensor, film: Tensor, *, streaming: bool) -> Tensor:
        if film.shape != (len(signal), 2 * self.channels, signal.shape[-1]):
            raise ValueError("per-layer FiLM shape is invalid")
        convolved = self.dilated.stream(signal) if streaming else self.dilated(signal)
        filter_term, gate_term = convolved.chunk(2, dim=1)
        gated = torch.tanh(filter_term) * torch.sigmoid(gate_term)
        gamma_raw, beta_raw = film.chunk(2, dim=1)
        conditioned = (1.0 + 0.25 * torch.tanh(gamma_raw)) * gated + 0.10 * torch.tanh(
            beta_raw
        )
        return signal + self.residual_projection(conditioned)

    def forward(self, signal: Tensor, film: Tensor) -> Tensor:
        return self._run(signal, film, streaming=False)

    def stream(self, signal: Tensor, film: Tensor) -> Tensor:
        return self._run(signal, film, streaming=True)


class DenseGatedWaveNetResidual(nn.Module):
    """Dense 64-channel gated WaveNet residual with 8,191-sample base RF."""

    def __init__(
        self,
        *,
        channels: int = 64,
        dilations: tuple[int, ...] = WAVENET_DILATIONS,
    ) -> None:
        super().__init__()
        if channels < 1 or not dilations or any(value < 1 for value in dilations):
            raise ValueError("WaveNet dimensions must be positive")
        self.channels = channels
        self.dilations = tuple(dilations)
        self.input_projection = nn.Conv1d(1, channels, 1)
        self.blocks = nn.ModuleList(
            GatedWaveNetBlock(channels, dilation) for dilation in self.dilations
        )
        self.output_projection = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.receptive_field_internal_samples = 1 + 2 * sum(self.dilations)

    @property
    def receptive_field_base_samples(self) -> int:
        return (self.receptive_field_internal_samples + 1) // 2

    def reset_state(self) -> None:
        for block in self.blocks:
            block.reset_state()

    def _run(self, signal: Tensor, film: Tensor, *, streaming: bool) -> Tensor:
        batched, scalar = _batch(signal)
        expected = (
            len(batched),
            len(self.blocks),
            2 * self.channels,
            batched.shape[-1],
        )
        if film.shape != expected:
            raise ValueError(f"WaveNet FiLM shape must be {expected}")
        hidden = self.input_projection(batched[:, None])
        for index, block in enumerate(self.blocks):
            hidden = (
                block.stream(hidden, film[:, index])
                if streaming
                else block(hidden, film[:, index])
            )
        residual = self.output_projection(hidden)[:, 0]
        return residual[0] if scalar else residual

    def forward_modulated(self, signal: Tensor, film: Tensor) -> Tensor:
        return self._run(signal, film, streaming=False)

    def stream_modulated(self, signal: Tensor, film: Tensor) -> Tensor:
        return self._run(signal, film, streaming=True)


class TeacherFullRateIslandX2(nn.Module):
    """Qualified 65-tap Kaiser x2 island returning only nonlinear residual."""

    factor = 2
    latency_samples = 32

    def __init__(self, branch: DenseGatedWaveNetResidual) -> None:
        super().__init__()
        lowpass = design_resampling_lowpass(2, 65, 8.6)
        self.branch = branch
        self.upsample_filter = FixedCausalFIR(2.0 * lowpass)
        self.downsample_filter = FixedCausalFIR(lowpass)
        self.modulation_delay = CausalTensorDelay(self.latency_samples // 2)

    def reset_state(self) -> None:
        self.upsample_filter.reset_state()
        self.downsample_filter.reset_state()
        self.modulation_delay.reset_state()
        self.branch.reset_state()

    @staticmethod
    def _zero_insert(signal: Tensor) -> Tensor:
        high_rate = signal.new_zeros((len(signal), 2 * signal.shape[-1]))
        high_rate[:, ::2] = signal
        return high_rate

    def _run(self, signal: Tensor, film: Tensor, *, streaming: bool) -> Tensor:
        batched, scalar = _batch(signal)
        if film.ndim != 4 or film.shape[0] != len(batched):
            raise ValueError("base-rate teacher FiLM shape is invalid")
        if film.shape[-1] != batched.shape[-1]:
            raise ValueError("teacher FiLM and audio lengths differ")
        high_rate = self._zero_insert(batched)
        interpolated = (
            self.upsample_filter.stream(high_rate)
            if streaming
            else self.upsample_filter(high_rate)
        )
        aligned = (
            self.modulation_delay.stream(film)
            if streaming
            else self.modulation_delay(film)
        )
        high_film = aligned.repeat_interleave(2, dim=-1)
        high_residual = (
            self.branch.stream_modulated(interpolated, high_film)
            if streaming
            else self.branch.forward_modulated(interpolated, high_film)
        )
        filtered = (
            self.downsample_filter.stream(high_residual)
            if streaming
            else self.downsample_filter(high_residual)
        )
        residual = filtered[:, ::2]
        return residual[0] if scalar else residual

    def forward_modulated(self, signal: Tensor, film: Tensor) -> Tensor:
        return self._run(signal, film, streaming=False)

    def stream_modulated(self, signal: Tensor, film: Tensor) -> Tensor:
        return self._run(signal, film, streaming=True)


@dataclass(frozen=True)
class QualityTeacherOutput:
    audio: Tensor
    fast: Tensor
    slow: Tensor
    fir: Tensor


class QualityTeacherAmplifier(nn.Module):
    """FIR plus x2 dense WaveNet, causally modulated by a dry-audio S4."""

    sample_rate_hz = 48_000
    internal_sample_rate_hz = 96_000
    channels_in = 1
    latency_samples = 32
    precision = "float32"
    aa_mode = "full_island_x2"

    def __init__(
        self,
        *,
        family: str = QUALITY_TEACHER_FAMILY,
        fir_taps: int = 257,
        wavenet_channels: int = 64,
        wavenet_dilations: tuple[int, ...] = WAVENET_DILATIONS,
        observer_channels: int = 64,
        observer_state_dim: int = 64,
        observer_blocks: int = 8,
        observer_evaluation_block_samples: int = 1_024,
    ) -> None:
        super().__init__()
        if family not in QUALITY_TEACHER_FAMILIES:
            raise ValueError("unknown AMP-QUALITY-TEACHER-v1 family")
        self.family = family
        self.fir = CausalFIR(fir_taps)
        self.fir_delay = CausalDelay(self.latency_samples)
        self.branch = DenseGatedWaveNetResidual(
            channels=wavenet_channels,
            dilations=wavenet_dilations,
        )
        self.observer = CausalS4FiLMObserver(
            channels=observer_channels,
            state_dim=observer_state_dim,
            blocks=observer_blocks,
            wavenet_layers=len(wavenet_dilations),
            wavenet_channels=wavenet_channels,
            evaluation_block_samples=observer_evaluation_block_samples,
        )
        self.island = TeacherFullRateIslandX2(self.branch)
        self.slow_diagnostic_delay = CausalTensorDelay(self.latency_samples)
        self.slow_modulation_enabled = family == QUALITY_TEACHER_FAMILY
        if not self.slow_modulation_enabled:
            for parameter in self.observer.parameters():
                parameter.requires_grad_(False)

    @property
    def receptive_field_samples(self) -> int:
        return self.branch.receptive_field_base_samples

    def initialize_fir(self, coefficients: Tensor) -> None:
        """Load externally fitted coefficients without changing FIR topology."""
        if coefficients.shape != self.fir.coefficients.shape:
            raise ValueError("teacher FIR coefficient shape changed")
        if not torch.isfinite(coefficients).all():
            raise ValueError("teacher FIR coefficients must be finite")
        with torch.no_grad():
            self.fir.coefficients.copy_(
                coefficients.to(
                    device=self.fir.coefficients.device,
                    dtype=self.fir.coefficients.dtype,
                )
            )

    def set_slow_modulation_enabled(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("slow-modulation flag must be boolean")
        self.slow_modulation_enabled = enabled

    def reset_state(self) -> None:
        self.fir.reset_state()
        self.fir_delay.reset_state()
        self.observer.reset_state()
        self.island.reset_state()
        self.slow_diagnostic_delay.reset_state()

    def reset(self) -> None:
        self.reset_state()

    def detach_stream_state(self) -> None:
        """Cut every persistent tensor at a truncated-BPTT boundary."""
        for module in self.modules():
            for name in ("_state", "_stream_state"):
                value = getattr(module, name, None)
                if value is not None:
                    setattr(module, name, _detach(value))

    def _film(self, signal: Tensor, *, streaming: bool) -> Tensor:
        if self.slow_modulation_enabled:
            return self.observer.stream(signal) if streaming else self.observer(signal)
        return signal.new_zeros(
            (
                len(signal),
                len(self.branch.blocks),
                2 * self.branch.channels,
                signal.shape[-1],
            )
        )

    def _run(self, signal: Tensor, *, streaming: bool) -> QualityTeacherOutput:
        batched, scalar = _batch(signal)
        if batched.dtype != torch.float32 or batched.shape[-1] < 1:
            raise ValueError("teacher input must be nonempty float32 mono audio")
        film = self._film(batched, streaming=streaming)
        fir_raw = self.fir.stream(batched) if streaming else self.fir(batched)
        fir = self.fir_delay.stream(fir_raw) if streaming else self.fir_delay(fir_raw)
        fast = (
            self.island.stream_modulated(batched, film)
            if streaming
            else self.island.forward_modulated(batched, film)
        )
        slow_trace = film.abs().mean(dim=(1, 2))
        slow = (
            self.slow_diagnostic_delay.stream(slow_trace)
            if streaming
            else self.slow_diagnostic_delay(slow_trace)
        )
        audio = fir + fast
        if scalar:
            return QualityTeacherOutput(audio[0], fast[0], slow[0], fir[0])
        return QualityTeacherOutput(audio, fast, slow, fir)

    def forward_components(self, signal: Tensor) -> QualityTeacherOutput:
        return self._run(signal, streaming=False)

    def stream_components(self, signal: Tensor) -> QualityTeacherOutput:
        return self._run(signal, streaming=True)

    def forward(self, signal: Tensor) -> Tensor:
        return self.forward_components(signal).audio

    def stream(self, signal: Tensor) -> Tensor:
        return self.stream_components(signal).audio


def build_quality_teacher_model(
    family: str, *, seed: int | None = None
) -> QualityTeacherAmplifier:
    """Build one registered graph; a shared seed gives exact paired weights."""
    if seed is None:
        return QualityTeacherAmplifier(family=family)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("teacher initialization seed must be non-negative")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = QualityTeacherAmplifier(family=family)
    model.initialization_seed = seed
    return model
