"""SSM-WaveNet: a WaveNet whose dilated convolutions are diagonal SSMs.

Each layer replaces the dilated causal convolution of a standard WaveNet with a
diagonal state-space model (DSSM). This gives each layer an infinite effective
memory via learned exponential decay, while keeping the gated nonlinearity and
residual structure of the WaveNet — which is what NAM's C++ engine already runs
fast.

At inference (sample-by-sample), each DSSM layer is a recurrence:
    h[t+1] = a ⊙ h[t] + b · x[t]
    y[t]   = Re(c · h[t]) + d · x[t]
This costs O(H) multiply-adds per sample per layer, similar to a dilated conv
of width H.

In training (batch of segments), the recurrence is materialised as a causal
convolution via FFT, exactly as nablafx's DSSM does it.

Parents: DSSM from nablafx (processors/blocks.py), WaveNet from NAM (dsp.h).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class DiagonalSSMLayer(nn.Module):
    """One diagonal SSM: complex state, real I/O, FFT convolution in training."""

    def __init__(self, channels: int, state_dim: int) -> None:
        super().__init__()
        self.channels = channels
        self.state_dim = state_dim
        # S4D parametrization: Re(A) < 0 always, discretized by a learnable dt.
        # log_A_real controls the continuous-time decay rate (always negative
        # after -exp). A_imag sets the pole angles — with dt in [1e-3, 1e-1]
        # the first poles land at 0-1.6 kHz, where S4-TFiLM won by 3.9x.
        self.log_A_real = nn.Parameter(torch.full((channels, state_dim), math.log(0.5)))
        self.A_imag = nn.Parameter(
            math.pi
            * torch.arange(state_dim).float().unsqueeze(0).expand(channels, -1).clone()
        )
        # dt ∈ [1e-3, 1e-1] → time constants 20-2000 samples (0.4-42 ms at 48 kHz)
        self.log_dt = nn.Parameter(
            torch.empty(channels).uniform_(math.log(1e-3), math.log(1e-1))
        )
        self.B = nn.Parameter(0.02 * torch.randn(channels, state_dim, 2))
        self.C = nn.Parameter(0.02 * torch.randn(channels, state_dim, 2))
        self.D = nn.Parameter(torch.ones(channels))
        # Inference state
        self._h: Tensor | None = None

    def _a(self) -> Tensor:
        """Discrete-time poles, |a| < 1 guaranteed by Re(A) < 0."""
        a_continuous = torch.complex(-torch.exp(self.log_A_real), self.A_imag)
        dt = torch.exp(self.log_dt).unsqueeze(-1)  # (C, 1)
        return torch.exp(a_continuous * dt)

    def _complex(self, param: Tensor) -> Tensor:
        return torch.complex(param[..., 0], param[..., 1])

    def _kernel(self, length: int) -> Tensor:
        """Build the causal convolution kernel via geometric series."""
        a = self._a()  # (C, H)
        b = self._complex(self.B)  # (C, H)
        c = self._complex(self.C)  # (C, H)
        # Powers of a: a^0, a^1, ..., a^{L-1}
        powers = torch.arange(length, device=a.device).float()
        # (C, H, L)
        a_powers = a.unsqueeze(-1) ** powers.unsqueeze(0).unsqueeze(0)
        # kernel[t] = Re(c · a^t · b), summed over state dim
        kernel = 2 * torch.einsum("ch,chl->cl", c * b, a_powers).real
        return kernel  # (C, L)

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, channels, length) -> same shape."""
        _, _, length = x.shape
        kernel = self._kernel(length)  # (C, L)
        # Causal convolution via FFT
        fft_length = 1
        while fft_length < length + length - 1:
            fft_length *= 2
        x_f = torch.fft.rfft(x, n=fft_length, dim=-1)
        k_f = torch.fft.rfft(kernel.unsqueeze(0), n=fft_length, dim=-1)
        y = torch.fft.irfft(x_f * k_f, n=fft_length, dim=-1)[..., :length]
        return y + x * self.D.unsqueeze(0).unsqueeze(-1)

    def reset_state(self) -> None:
        self._h = None

    def step(self, x: Tensor) -> Tensor:
        """Single-sample inference: x is (batch, channels)."""
        a = self._a()
        b = self._complex(self.B)
        c = self._complex(self.C)
        x_complex = x.to(torch.complex64)
        if self._h is None:
            self._h = torch.zeros(
                x.shape[0],
                self.channels,
                self.state_dim,
                dtype=torch.complex64,
                device=x.device,
            )
        self._h = a.unsqueeze(0) * self._h + b.unsqueeze(0) * x_complex.unsqueeze(-1)
        y = 2 * torch.einsum("bch,ch->bc", self._h, c).real
        return y + x * self.D.unsqueeze(0)


class GatedSSMBlock(nn.Module):
    """One WaveNet-style block with a DSSM instead of a dilated convolution.

    Input -> DSSM -> [tanh branch, sigmoid branch] -> 1x1 conv -> residual + skip
    """

    def __init__(self, channels: int, state_dim: int) -> None:
        super().__init__()
        self.ssm = DiagonalSSMLayer(channels, state_dim)
        # Project to double width for gated activation
        self.pre_gate = nn.Conv1d(channels, 2 * channels, 1)
        self.res_conv = nn.Conv1d(channels, channels, 1)
        self.skip_conv = nn.Conv1d(channels, channels, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (residual, skip)."""
        h = self.ssm(x)
        h = self.pre_gate(h)
        gate_a, gate_b = h.chunk(2, dim=1)
        h = torch.tanh(gate_a) * torch.sigmoid(gate_b)
        skip = self.skip_conv(h)
        residual = self.res_conv(h) + x
        return residual, skip

    def reset_state(self) -> None:
        self.ssm.reset_state()


class SSMWaveNet(nn.Module):
    """A WaveNet backbone with diagonal SSMs replacing dilated convolutions.

    The model takes (batch, 1, length) and returns (batch, 1, length).
    """

    def __init__(
        self,
        num_blocks: int = 8,
        channels: int = 16,
        state_dim: int = 8,
        input_channels: int = 1,
        output_channels: int = 1,
    ) -> None:
        super().__init__()
        self.input_conv = nn.Conv1d(input_channels, channels, 1)
        self.blocks = nn.ModuleList(
            [GatedSSMBlock(channels, state_dim) for _ in range(num_blocks)]
        )
        self.output_net = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv1d(channels, output_channels, 1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, 1, length) -> (batch, 1, length)."""
        h = self.input_conv(x)
        skip_sum = torch.zeros_like(h)
        for block in self.blocks:
            h, skip = block(h)
            skip_sum = skip_sum + skip
        return self.output_net(skip_sum)

    def reset_states(self) -> None:
        for block in self.blocks:
            block.reset_state()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
