"""SSM-WaveNet: a WaveNet whose dilated convolutions are diagonal SSMs.

Each layer replaces the dilated causal convolution of a standard WaveNet with a
diagonal state-space model (DSSM). This gives each layer an infinite effective
memory via learned exponential decay, while keeping the gated nonlinearity and
residual structure of the WaveNet -- which is what NAM's C++ engine already runs
fast.

At inference (sample-by-sample), each DSSM layer is a recurrence:
    h[t+1] = a * h[t] + b * x[t]
    y[t]   = Re(c * h[t]) + d * x[t]
This costs O(H) multiply-adds per sample per layer, similar to a dilated conv
of width H.

In training (batch of segments), the recurrence is materialised as a causal
convolution via FFT, exactly as nablafx's DSSM does it.

Parents: DSSM from nablafx (processors/blocks.py), WaveNet from NAM (dsp.h).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

# Time constants in seconds for --circuit-init. Order-of-magnitude guesses for
# coupling, tone-stack and bias-recovery RC products; NOT checked against a
# Big Muff schematic.
BIG_MUFF_TAU_S = (0.001, 0.0047, 0.010, 0.047)
SAMPLE_RATE = 48_000


class DiagonalSSMLayer(nn.Module):
    """One diagonal SSM: complex state, real I/O, FFT convolution in training.

    discretization="free" learns the input vector b (the first version).
    "zoh" fixes it by zero-order hold, b = (a - 1) / lambda, as S4D and
    nablafx's DSSM do, which bounds every mode's DC gain by |c / lambda|
    whatever dt; c then starts at unit scale.
    """

    def __init__(
        self,
        channels: int,
        state_dim: int,
        circuit_tau_s: Sequence[float] | None = None,
        discretization: str = "free",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.state_dim = state_dim
        self.discretization = discretization
        # S4D parametrization: Re(A) < 0 always, discretized by a learnable dt.
        self.log_A_real = nn.Parameter(torch.full((channels, state_dim), math.log(0.5)))
        self.A_imag = nn.Parameter(
            math.pi
            * torch.arange(state_dim).float().unsqueeze(0).expand(channels, -1).clone()
        )
        # Circuit-informed init: if we know the device's RC time constants,
        # set dt so the first poles land on them. Otherwise uniform in
        # [1e-3, 1e-1], giving tau = 20-2000 samples (0.4-42 ms at 48 kHz).
        if circuit_tau_s is not None:
            dt_init = self._circuit_dt(circuit_tau_s, channels)
        else:
            dt_init = torch.empty(channels).uniform_(math.log(1e-3), math.log(1e-1))
        self.log_dt = nn.Parameter(dt_init)
        # nablafx's DSSM marks the same parameters as exempt from weight decay;
        # an optimizer only honours this if it reads _optim.
        for parameter in (self.log_A_real, self.A_imag, self.log_dt):
            parameter._optim = {"weight_decay": 0.0}
        if discretization == "free":
            self.B = nn.Parameter(0.02 * torch.randn(channels, state_dim, 2))
            self.C = nn.Parameter(0.02 * torch.randn(channels, state_dim, 2))
        elif discretization == "zoh":
            c = torch.randn(channels, state_dim, dtype=torch.cfloat)
            self.C = nn.Parameter(torch.view_as_real(c).clone())
        else:
            raise ValueError(f"unknown discretization: {discretization}")
        self.D = nn.Parameter(torch.ones(channels))
        self._h: Tensor | None = None

    @staticmethod
    def _circuit_dt(tau_s: Sequence[float], channels: int) -> Tensor:
        """Set dt so |a| = exp(-dt/tau), targeting the known time constants.

        With log_A_real init at log(0.5), the continuous-time decay rate is
        exp(log(0.5)) = 0.5. The discrete pole is exp(-0.5 * dt). To get
        tau_samples = tau_s * SR, we need dt = 1 / (0.5 * tau_samples)
        = 2 / (tau_s * SR).
        """
        tau_samples = [t * SAMPLE_RATE for t in tau_s]
        dt_values = [2.0 / ts for ts in tau_samples]
        # Tile across channels, cycling through the known constants
        log_dt = torch.zeros(channels)
        for i in range(channels):
            log_dt[i] = math.log(dt_values[i % len(dt_values)])
        return log_dt

    def _lambda(self) -> Tensor:
        return torch.complex(-torch.exp(self.log_A_real), self.A_imag)

    def _a(self) -> Tensor:
        """Discrete-time poles, |a| < 1 guaranteed by Re(A) < 0."""
        dt = torch.exp(self.log_dt).unsqueeze(-1)
        return torch.exp(self._lambda() * dt)

    def _b(self, a: Tensor) -> Tensor:
        if self.discretization == "zoh":
            return (a - 1) / self._lambda()
        return self._complex(self.B)

    def _complex(self, param: Tensor) -> Tensor:
        return torch.complex(param[..., 0], param[..., 1])

    def _kernel(self, length: int) -> Tensor:
        """Build the causal convolution kernel via geometric series."""
        a = self._a()
        b = self._b(a)
        c = self._complex(self.C)
        powers = torch.arange(length, device=a.device).float()
        a_powers = a.unsqueeze(-1) ** powers.unsqueeze(0).unsqueeze(0)
        kernel = 2 * torch.einsum("ch,chl->cl", c * b, a_powers).real
        return kernel

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, channels, length) -> same shape."""
        _, _, length = x.shape
        kernel = self._kernel(length)
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
        b = self._b(a)
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


class SineGate(nn.Module):
    """x + sin(w * x) with learnable frequency per channel.

    A smooth, bounded non-linearity that can represent asymmetric distortion
    curves (unlike tanh which is odd-symmetric). At w=0 it is the identity;
    at large w it oscillates, adding harmonics the way a real transistor does.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(channels, 1) * math.pi)

    def forward(self, x: Tensor) -> Tensor:
        return x + torch.sin(self.w.unsqueeze(0) * x)


class GatedSSMBlock(nn.Module):
    """One WaveNet-style block with a DSSM instead of a dilated convolution.

    Input -> DSSM -> [tanh branch, sigmoid branch] -> 1x1 conv -> residual + skip

    With act_type="sine", the gated activation is replaced by x + sin(w*x),
    which can model asymmetric distortion curves.
    """

    def __init__(
        self,
        channels: int,
        state_dim: int,
        act_type: str = "gated",
        circuit_tau_s: Sequence[float] | None = None,
        discretization: str = "free",
    ) -> None:
        super().__init__()
        self.ssm = DiagonalSSMLayer(
            channels,
            state_dim,
            circuit_tau_s=circuit_tau_s,
            discretization=discretization,
        )
        self.act_type = act_type
        if act_type == "sine":
            self.pre_act = nn.Conv1d(channels, channels, 1)
            self.activation = SineGate(channels)
        else:
            self.pre_gate = nn.Conv1d(channels, 2 * channels, 1)
        self.res_conv = nn.Conv1d(channels, channels, 1)
        self.skip_conv = nn.Conv1d(channels, channels, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (residual, skip)."""
        h = self.ssm(x)
        if self.act_type == "sine":
            h = self.activation(self.pre_act(h))
        else:
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

    Args:
        act_type: "gated" (tanh * sigmoid, the WaveNet default) or "sine"
            (x + sin(w*x), learnable asymmetric distortion).
        output_act: "tanh" (bounded, safe but compresses amplifying devices),
            "softsign" (bounded, gentler saturation), or "none" (unbounded,
            lets the model reproduce gain > 1 but needs a lower lr).
        circuit_tau_s: known RC time constants of the device, in seconds.
            When given, the SSM poles are initialised on these constants
            instead of uniform random.
        discretization: "free" (learned input vector) or "zoh" (S4D
            zero-order hold); see DiagonalSSMLayer.
    """

    def __init__(
        self,
        num_blocks: int = 8,
        channels: int = 16,
        state_dim: int = 8,
        input_channels: int = 1,
        output_channels: int = 1,
        act_type: str = "gated",
        output_act: str = "tanh",
        circuit_tau_s: Sequence[float] | None = None,
        discretization: str = "free",
    ) -> None:
        super().__init__()
        self.input_conv = nn.Conv1d(input_channels, channels, 1)
        self.blocks = nn.ModuleList(
            [
                GatedSSMBlock(
                    channels,
                    state_dim,
                    act_type=act_type,
                    circuit_tau_s=circuit_tau_s,
                    discretization=discretization,
                )
                for _ in range(num_blocks)
            ]
        )
        final_act: nn.Module
        if output_act == "tanh":
            final_act = nn.Tanh()
        elif output_act == "softsign":
            final_act = nn.Softsign()
        elif output_act == "none":
            final_act = nn.Identity()
        else:
            raise ValueError(f"unknown output_act: {output_act}")
        self.output_net = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv1d(channels, output_channels, 1),
            final_act,
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
