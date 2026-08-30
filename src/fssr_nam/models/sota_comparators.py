"""Audited local adapters for the frozen AMP-QUALITY-ARCH-v1 comparators.

The NablaFX implementations below intentionally cover only the two black-box
TFiLM configurations frozen by the campaign.  Their equations and parameter
initialisation follow the pinned upstream source; importing the entire project
would otherwise require its unused ``rational`` activation dependency.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _mono_batch(signal: Tensor) -> tuple[Tensor, bool]:
    if signal.ndim == 1:
        return signal[None, :], True
    if signal.ndim == 2:
        return signal, False
    raise ValueError("signal must have shape (time,) or (batch,time)")


def _delay(signal: Tensor, samples: int) -> Tensor:
    if samples == 0:
        return signal
    if samples >= signal.shape[-1]:
        return torch.zeros_like(signal)
    return F.pad(signal, (samples, 0))[..., :-samples]


class TemporalFiLM(nn.Module):
    """NablaFX TFiLM: block max-pooling followed by an LSTM affine adaptor."""

    def __init__(self, channels: int, block_size: int = 128) -> None:
        super().__init__()
        if channels < 1 or block_size < 1:
            raise ValueError("channels and block_size must be positive")
        self.channels = channels
        self.block_size = block_size
        self.lstm = nn.LSTM(channels, 2 * channels, num_layers=1)

    def forward(self, signal: Tensor) -> Tensor:
        if signal.ndim != 3 or signal.shape[1] != self.channels:
            raise ValueError("TFiLM input must have shape (batch,channels,time)")
        original_samples = signal.shape[-1]
        padding = (-original_samples) % self.block_size
        padded = F.pad(signal, (0, padding)) if padding else signal
        steps = padded.shape[-1] // self.block_size
        pooled = F.max_pool1d(
            padded, kernel_size=self.block_size, stride=self.block_size
        )
        initial = pooled.new_zeros((1, pooled.shape[0], 2 * self.channels))
        modulation, _ = self.lstm(pooled.permute(2, 0, 1), (initial, initial.clone()))
        modulation = modulation.permute(1, 2, 0).unsqueeze(-1)
        gain, bias = modulation.chunk(2, dim=1)
        blocked = padded.reshape(padded.shape[0], self.channels, steps, self.block_size)
        output = (blocked * gain + bias).reshape_as(padded)
        return output[..., :original_samples]


class _DelayedStreamingModel(nn.Module):
    """Make a bounded-lookahead offline model causal through an explicit delay.

    ``stream`` deliberately recomputes the prefix.  It is a numerical reference
    used to prove arbitrary-block equivalence, not the deployable benchmark path.
    """

    latency_samples: int

    def __init__(self, latency_samples: int) -> None:
        super().__init__()
        self.latency_samples = latency_samples
        self.register_buffer("_stream_input", torch.empty(0), persistent=False)

    def _forward_raw(self, signal: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, signal: Tensor) -> Tensor:
        batched, squeeze = _mono_batch(signal)
        output = _delay(self._forward_raw(batched), self.latency_samples)
        return output.squeeze(0) if squeeze else output

    def reset_state(self) -> None:
        self._stream_input = self._stream_input.new_empty(0)

    def stream(self, signal: Tensor) -> Tensor:
        batched, squeeze = _mono_batch(signal)
        if self._stream_input.numel() == 0:
            history = batched
            previous_samples = 0
        else:
            if self._stream_input.shape[0] != batched.shape[0]:
                raise ValueError("stream batch size changed without reset")
            previous_samples = self._stream_input.shape[-1]
            history = torch.cat((self._stream_input, batched), dim=-1)
        self._stream_input = history.detach()
        output = self.forward(history)[..., previous_samples:]
        return output.squeeze(0) if squeeze else output


class _TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        block_size: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            bias=True,
        )
        self.film = TemporalFiLM(out_channels, block_size)
        self.res = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            groups=in_channels,
            bias=False,
        )

    def forward(self, signal: Tensor) -> Tensor:
        output = torch.tanh(self.film(self.conv(signal)))
        # Pinned NablaFX ``causal_crop`` deliberately excludes the newest
        # residual sample (stop = length - 1).
        residual = self.res(signal)[..., -(output.shape[-1] + 1) : -1]
        return output + residual


@dataclass(frozen=True)
class TCNVariant:
    blocks: int
    kernel_size: int
    dilation_growth: int
    channels: int = 16
    block_size: int = 128


TCN_VARIANTS = {
    "small": TCNVariant(blocks=5, kernel_size=13, dilation_growth=10),
    "large": TCNVariant(blocks=10, kernel_size=5, dilation_growth=3),
}


class NablafxTCNTFiLM(_DelayedStreamingModel):
    """Pinned causal NablaFX TCN-TFiLM black-box comparator."""

    def __init__(self, variant: str) -> None:
        try:
            config = TCN_VARIANTS[variant]
        except KeyError as exc:
            raise ValueError(f"unknown TCN-TFiLM variant: {variant}") from exc
        # Each TFiLM layer can inspect the rest of its current feature block.
        # Summing those bounds is conservative and makes the adapter causal.
        super().__init__(config.blocks * (config.block_size - 1))
        self.variant = variant
        self.config = config
        dilations = [config.dilation_growth**index for index in range(config.blocks)]
        self.receptive_field = 1 + (config.kernel_size - 1) * sum(dilations)
        blocks: list[nn.Module] = []
        for index, dilation in enumerate(dilations):
            blocks.append(
                _TCNBlock(
                    1 if index == 0 else config.channels,
                    config.channels,
                    kernel_size=config.kernel_size,
                    dilation=dilation,
                    block_size=config.block_size,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Conv1d(config.channels, 1, kernel_size=1, bias=True)

    def _forward_raw(self, signal: Tensor) -> Tensor:
        output = F.pad(signal[:, None, :], (self.receptive_field - 1, 0))
        for block in self.blocks:
            output = block(output)
        return torch.tanh(self.output(output)).squeeze(1)


class DiagonalS4(nn.Module):
    """Diagonal S4 convolution used by the pinned NablaFX DSSM block."""

    def __init__(self, channels: int, state_dim: int) -> None:
        super().__init__()
        if channels < 1 or state_dim < 1:
            raise ValueError("channels and state_dim must be positive")
        self.channels = channels
        self.state_dim = state_dim
        log_dt = torch.rand(channels) * (math.log(0.1) - math.log(0.001))
        log_dt += math.log(0.001)
        self.log_dt = nn.Parameter(log_dt)
        complex_c = torch.randn(channels, state_dim, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(complex_c))
        self.log_A_real = nn.Parameter(
            torch.log(torch.full((channels, state_dim), 0.5))
        )
        self.A_imag = nn.Parameter(
            math.pi * torch.arange(state_dim).repeat(channels, 1)
        )
        self.D = nn.Parameter(torch.randn(channels))

    def kernel(self, samples: int) -> Tensor:
        if samples < 1:
            raise ValueError("kernel length must be positive")
        dt = self.log_dt.exp()
        c = torch.view_as_complex(self.C)
        a = -self.log_A_real.exp() + 1j * self.A_imag
        dt_a = a * dt[:, None]
        powers = dt_a[..., None] * torch.arange(samples, device=a.device)
        c_bar = c * (dt_a.exp() - 1.0) / a
        return 2 * torch.einsum("hn,hnl->hl", c_bar, powers.exp()).real

    def forward(self, signal: Tensor) -> Tensor:
        if signal.ndim != 3 or signal.shape[1] != self.channels:
            raise ValueError("S4 input must have shape (batch,channels,time)")
        samples = signal.shape[-1]
        kernel_fft = torch.fft.rfft(self.kernel(samples), n=2 * samples)
        signal_fft = torch.fft.rfft(signal, n=2 * samples)
        output = torch.fft.irfft(signal_fft * kernel_fft, n=2 * samples)
        return output[..., :samples] + signal * self.D[None, :, None]

    def recurrent(self, signal: Tensor) -> Tensor:
        """Evaluate the same causal convolution as an exact state recurrence."""
        if signal.ndim != 3 or signal.shape[1] != self.channels:
            raise ValueError("S4 input must have shape (batch,channels,time)")
        dt = self.log_dt.exp()
        c = torch.view_as_complex(self.C)
        a = -self.log_A_real.exp() + 1j * self.A_imag
        transition = (a * dt[:, None]).exp()
        input_gain = ((a * dt[:, None]).exp() - 1.0) / a
        state = torch.zeros(
            signal.shape[0],
            self.channels,
            self.state_dim,
            dtype=c.dtype,
            device=signal.device,
        )
        outputs: list[Tensor] = []
        for sample in signal.unbind(dim=-1):
            state = transition[None] * state + input_gain[None] * sample[..., None]
            response = 2 * torch.einsum("bhn,hn->bh", state, c).real
            outputs.append(response + self.D[None] * sample)
        return torch.stack(outputs, dim=-1)


class _S4Block(nn.Module):
    def __init__(self, channels: int, state_dim: int, block_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(channels, channels)
        self.s4 = DiagonalS4(channels, state_dim)
        self.film = TemporalFiLM(channels, block_size)
        self.res = nn.Conv1d(
            channels, channels, kernel_size=1, groups=channels, bias=False
        )

    def forward(self, signal: Tensor) -> Tensor:
        mixed = self.linear(signal.transpose(1, 2)).transpose(1, 2)
        output = torch.tanh(mixed)
        output = torch.tanh(self.film(self.s4(output)))
        return output + self.res(signal)


@dataclass(frozen=True)
class S4Variant:
    blocks: int
    state_dim: int
    channels: int = 16
    block_size: int = 128


S4_VARIANTS = {
    "small": S4Variant(blocks=4, state_dim=4),
    "large": S4Variant(blocks=8, state_dim=32),
}

COMPARATOR_VARIANTS = {
    "nam_a2_full": ("official",),
    "nam_a2_lite": ("official",),
    "wright_lstm64": ("official",),
    "nablafx_tcn_tfilm": tuple(TCN_VARIANTS),
    "nablafx_s4_tfilm": tuple(S4_VARIANTS),
}


class NablafxS4TFiLM(_DelayedStreamingModel):
    """Pinned diagonal-S4 TFiLM black-box comparator."""

    def __init__(self, variant: str) -> None:
        try:
            config = S4_VARIANTS[variant]
        except KeyError as exc:
            raise ValueError(f"unknown S4-TFiLM variant: {variant}") from exc
        super().__init__(config.blocks * (config.block_size - 1))
        self.variant = variant
        self.config = config
        self.expand = nn.Linear(1, config.channels)
        self.blocks = nn.ModuleList(
            [
                _S4Block(config.channels, config.state_dim, config.block_size)
                for _ in range(config.blocks)
            ]
        )
        self.contract = nn.Linear(config.channels, 1)

    def _forward_raw(self, signal: Tensor) -> Tensor:
        output = self.expand(signal[..., None]).transpose(1, 2)
        for block in self.blocks:
            output = block(output)
        output = self.contract(output.transpose(1, 2)).squeeze(-1)
        return torch.tanh(output)


def build_nablafx_comparator(family: str, variant: str) -> nn.Module:
    """Construct an audited NablaFX comparator without optional dependencies."""
    if family == "nablafx_tcn_tfilm":
        return NablafxTCNTFiLM(variant)
    if family == "nablafx_s4_tfilm":
        return NablafxS4TFiLM(variant)
    raise ValueError(f"not a NablaFX comparator family: {family}")


def build_sota_comparator(
    family: str, *, variant: str, root: Path | None = None
) -> nn.Module:
    """Build one of the four frozen open comparators from audited sources."""
    try:
        allowed = COMPARATOR_VARIANTS[family]
    except KeyError as exc:
        raise ValueError(f"unknown SOTA comparator family: {family}") from exc
    if variant not in allowed:
        raise ValueError(f"invalid {family} variant: {variant}")
    if family.startswith("nablafx_"):
        return build_nablafx_comparator(family, variant)
    if family == "wright_lstm64":
        from fssr_nam.models.wright import WrightLSTM

        return WrightLSTM(hidden_size=64, sample_rate=48_000)
    if root is None:
        raise ValueError("repository root is required to construct NAM A2")
    from nam.models.factory import init as init_nam_model

    config_path = (
        root / "third_party/neural-amp-modeler/nam/train/_resources/"
        "config_model_packed.json"
    )
    packed = json.loads(config_path.read_text(encoding="utf-8"))["net"]
    model = init_nam_model(packed["name"], kwargs={"config": packed["config"]})
    return model.extract_submodel(0 if family == "nam_a2_lite" else 1)
