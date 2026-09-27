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
import torch.utils.checkpoint
from torch import Tensor

# Time constants in seconds for --circuit-init. Order-of-magnitude guesses for
# coupling, tone-stack and bias-recovery RC products; NOT checked against a
# Big Muff schematic.
BIG_MUFF_TAU_S = (0.001, 0.0047, 0.010, 0.047)
SAMPLE_RATE = 48_000
# Knob-driven pole shifts are POLE_SCALE * tanh(.), so a knob moves log_dt and each
# log_A_real by at most e^2.3, a factor 10 either way (demo/param/predictions_poles.md).
POLE_SCALE = 2.3


def phi2(z: Tensor) -> Tensor:
    """(e^z - 1 - z) / z^2 in complex128, by its series where the closed form cancels.
    The inner where keeps the unused closed-form branch finite, gradients included."""
    z = z.to(torch.complex128)
    small = z.abs() < 1e-4
    safe = torch.where(small, torch.ones_like(z), z)
    closed = (torch.exp(safe) - 1 - safe) / safe**2
    return torch.where(small, 0.5 + z / 6 + z**2 / 24, closed)


class DiagonalSSMLayer(nn.Module):
    """One diagonal SSM: complex state, real I/O, FFT convolution in training.

    discretization="free" learns the input vector b (the first version).
    "zoh" fixes it by zero-order hold, b = (a - 1) / lambda, as S4D and
    nablafx's DSSM do, which bounds every mode's DC gain by |c / lambda|
    whatever dt; c then starts at unit scale.
    "foh" is the first-order hold: the input is linear between samples, so
    h[n] = a h[n-1] + beta1 x[n] + beta0 x[n-1]. ZOH holds x[n] over the interval
    before t_n, a half-sample advance learned at the training rate that does not
    rescale with dt; FOH has none (demo/sr_transfer/FIX_foh.md). Same parameters and
    initialisation as "zoh".
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
        elif discretization in ("zoh", "foh"):
            c = torch.randn(channels, state_dim, dtype=torch.cfloat)
            self.C = nn.Parameter(torch.view_as_real(c).clone())
        else:
            raise ValueError(f"unknown discretization: {discretization}")
        self.D = nn.Parameter(torch.ones(channels))
        self._h: Tensor | None = None
        self._x_prev: Tensor | None = None  # first-order hold only

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

    def _foh_inputs(self) -> tuple[Tensor, Tensor, Tensor]:
        """Pole a and first-order-hold input weights beta1 (of x[n]) and beta0 (of
        x[n-1]); beta0 + beta1 = (a - 1) / lambda, the ZOH weight, so DC is kept."""
        a = self._a()
        dt = torch.exp(self.log_dt).unsqueeze(-1).to(torch.float64)
        z = self._lambda().to(torch.complex128) * dt
        beta1 = dt * phi2(z)
        beta0 = dt * torch.exp(z) * phi2(-z)
        return a, beta1.to(a.dtype), beta0.to(a.dtype)

    def _foh_kernel(self, length: int) -> Tensor:
        a, beta1, beta0 = self._foh_inputs()
        c = self._complex(self.C)
        powers = torch.arange(length, device=a.device).float()
        a_powers = a.unsqueeze(-1) ** powers.unsqueeze(0).unsqueeze(0)
        now = 2 * torch.einsum("ch,chl->cl", c * beta1, a_powers).real
        late = 2 * torch.einsum("ch,chl->cl", c * beta0, a_powers).real
        return torch.cat([now[:, :1], now[:, 1:] + late[:, :-1]], dim=-1)

    def _kernel(self, length: int) -> Tensor:
        """Build the causal convolution kernel via geometric series."""
        if self.discretization == "foh":
            return self._foh_kernel(length)
        a = self._a()
        b = self._b(a)
        c = self._complex(self.C)
        powers = torch.arange(length, device=a.device).float()
        a_powers = a.unsqueeze(-1) ** powers.unsqueeze(0).unsqueeze(0)
        kernel = 2 * torch.einsum("ch,chl->cl", c * b, a_powers).real
        return kernel

    def forward(self, x: Tensor, shift: tuple[Tensor, Tensor] | None = None) -> Tensor:
        """x: (batch, channels, length) -> same shape. shift: per-example pole shifts
        (dlog_dt (batch, channels), dlog_A_real (batch, channels, state_dim)), which
        give every example its own kernel."""
        _, _, length = x.shape
        if shift is None:
            kernel = self._kernel(length).unsqueeze(0)
        else:
            kernel = self._shifted_kernel(length, shift)
        fft_length = 1
        while fft_length < length + length - 1:
            fft_length *= 2
        x_f = torch.fft.rfft(x, n=fft_length, dim=-1)
        k_f = torch.fft.rfft(kernel, n=fft_length, dim=-1)
        y = torch.fft.irfft(x_f * k_f, n=fft_length, dim=-1)[..., :length]
        return y + x * self.D.unsqueeze(0).unsqueeze(-1)

    def _shifted_inputs(
        self, shift: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Per example: pole a, weight of x[n] and weight of x[n-1], all (batch,
        channels, state_dim). The shifts add to log_dt and log_A_real; A_imag is not
        moved, a passive tone stack having real poles."""
        if self.discretization == "free":
            raise ValueError("pole shifts need discretization 'zoh' or 'foh'")
        dlog_dt, dlog_a = shift
        lam = torch.complex(
            -torch.exp(self.log_A_real + dlog_a), self.A_imag.expand_as(dlog_a)
        )
        dt = torch.exp(self.log_dt + dlog_dt).unsqueeze(-1)
        a = torch.exp(lam * dt)
        if self.discretization == "zoh":
            return a, (a - 1) / lam, torch.zeros_like(a)
        dt64 = dt.to(torch.float64)
        z = lam.to(torch.complex128) * dt64
        beta1 = dt64 * phi2(z)
        beta0 = dt64 * torch.exp(z) * phi2(-z)
        return a, beta1.to(a.dtype), beta0.to(a.dtype)

    def _shifted_kernel(self, length: int, shift: tuple[Tensor, Tensor]) -> Tensor:
        """(batch, channels, length): one kernel per example."""
        a, now, late = self._shifted_inputs(shift)
        c = self._complex(self.C)
        powers = torch.arange(length, device=a.device).float()
        a_powers = a.unsqueeze(-1) ** powers
        kernel = 2 * torch.einsum("bch,bchl->bcl", c * now, a_powers).real
        if self.discretization == "zoh":
            return kernel
        delayed = 2 * torch.einsum("bch,bchl->bcl", c * late, a_powers).real
        return torch.cat([kernel[..., :1], kernel[..., 1:] + delayed[..., :-1]], dim=-1)

    def reset_state(self) -> None:
        self._h = None

    def step(self, x: Tensor, shift: tuple[Tensor, Tensor] | None = None) -> Tensor:
        """Single-sample inference: x is (batch, channels)."""
        if shift is not None:
            return self._shifted_step(x, shift)
        if self.discretization == "foh":
            return self._foh_step(x)
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

    def _foh_step(self, x: Tensor) -> Tensor:
        a, beta1, beta0 = self._foh_inputs()
        c = self._complex(self.C)
        x_complex = x.to(a.dtype).unsqueeze(-1)
        if self._h is None:
            self._h = torch.zeros(
                x.shape[0],
                self.channels,
                self.state_dim,
                dtype=a.dtype,
                device=x.device,
            )
            self._x_prev = torch.zeros_like(x_complex)
        self._h = a * self._h + beta1 * x_complex + beta0 * self._x_prev
        self._x_prev = x_complex
        y = 2 * torch.einsum("bch,ch->bc", self._h, c).real
        return y + x * self.D.unsqueeze(0)

    def _shifted_step(self, x: Tensor, shift: tuple[Tensor, Tensor]) -> Tensor:
        a, now, late = self._shifted_inputs(shift)
        c = self._complex(self.C)
        x_complex = x.to(a.dtype).unsqueeze(-1)
        if self._h is None:
            self._h = torch.zeros_like(a)
            self._x_prev = torch.zeros_like(x_complex)
        self._h = a * self._h + now * x_complex + late * self._x_prev
        self._x_prev = x_complex
        y = 2 * torch.einsum("bch,ch->bc", self._h, c).real
        return y + x * self.D.unsqueeze(0)


@torch.no_grad()
def zoh_to_foh(model: nn.Module) -> None:
    """Read every ZOH layer of `model` as a first-order hold, in place, keeping its
    impulse response at the current dt: c' = c b / (beta1 + beta0 / a) carries the taps
    m >= 1 and D takes the difference at m = 0. Unlike ZOH, the result keeps each
    mode's DC gain and first moment when dt is rescaled, which is exactly what the
    two-coefficient correction imposes (two_tap_kernel; demo/sr_transfer/FIX_foh.md).

    a, b and the betas are taken as the layer computes them, in its own precision, and
    only then promoted: a float32 layer's b = (a - 1) / lambda carries cancellation
    error on slow modes, and the model was trained with that b, not the exact one.

    Refuses knob-driven poles (cond="poles"): c' and D would then depend on the knobs,
    and fixed ones would change the response of every setting but the central one."""
    if any(getattr(m, "pole_net", None) is not None for m in model.modules()):
        raise ValueError("zoh_to_foh cannot convert knob-driven poles")
    for layer in model.modules():
        if not isinstance(layer, DiagonalSSMLayer) or layer.discretization != "zoh":
            continue
        a, beta1, beta0 = layer._foh_inputs()
        b = layer._b(a)
        a, b, beta1, beta0 = (t.to(torch.complex128) for t in (a, b, beta1, beta0))
        # clone: on a float64 layer .double() is C itself, overwritten by copy_ below.
        c = torch.view_as_complex(layer.C.double().contiguous()).clone()
        c_foh = c * b / (beta1 + beta0 / a)
        layer.C.copy_(torch.view_as_real(c_foh).to(layer.C.dtype))
        layer.D.add_((2 * (c * b - c_foh * beta1).sum(-1).real).to(layer.D.dtype))
        layer.discretization = "foh"
        layer.reset_state()


def two_tap_kernel(module: nn.Module, dt0: Tensor, length: int) -> Tensor:
    """Mode kernel c (b0 a'^k + b1 a'^(k-1)): DC and first derivative at 0 Hz equal to
    the 48 kHz mode's (demo/sr_transfer/predictions_hold.md, two-coefficient arm).
    Moved unchanged from scripts/product_sr_transfer.py, where it measured H6, so that
    tests can compare zoh_to_foh with it."""
    lam = torch.complex(-torch.exp(module.log_A_real), module.A_imag).detach()
    lam = lam.to(torch.complex128)
    step = torch.exp(module.log_dt.detach()).double().unsqueeze(-1)
    base = dt0.double().unsqueeze(-1)
    a48, a = torch.exp(lam * base), torch.exp(lam * step)
    total = -(1 - a) / lam
    b1 = total * (base * a48 / (1 - a48) - step * a / (1 - a)) / step
    b0 = total - b1
    c = torch.view_as_complex(module.C.detach().contiguous()).to(torch.complex64)
    powers = torch.arange(length).float()
    a_powers = a.to(torch.complex64).unsqueeze(-1) ** powers
    kernel = 2 * torch.einsum("ch,chl->cl", c * b0.to(torch.complex64), a_powers).real
    late = 2 * torch.einsum("ch,chl->cl", c * b1.to(torch.complex64), a_powers).real
    kernel[:, 1:] += late[:, :-1]
    return kernel


class AdditiveHead(nn.Module):
    """One hidden layer per knob into a shared output layer: the output is a sum of
    functions of one knob each, so knobs cannot interact."""

    def __init__(self, widths: Sequence[int], out_features: int) -> None:
        super().__init__()
        self.inputs = nn.ModuleList(nn.Linear(1, width) for width in widths)
        self.output = nn.Linear(sum(widths), out_features)

    def forward(self, controls: Tensor) -> Tensor:
        hidden = torch.cat(
            [layer(controls[:, k : k + 1]) for k, layer in enumerate(self.inputs)],
            dim=1,
        )
        return self.output(torch.relu(hidden))


def control_head(
    num_controls: int, out_features: int, head: str, hidden: Sequence[int]
) -> nn.Module:
    """Knobs -> conditioning, output layer at zero. "joint" with hidden (32,) is the
    FiLM head of the parametric runs of 2026-09-23, built in the same order."""
    if head == "joint":
        (width,) = hidden
        net = nn.Sequential(
            nn.Linear(num_controls, width), nn.ReLU(), nn.Linear(width, out_features)
        )
        last = net[2]
    elif head == "additive":
        if len(hidden) != num_controls:
            raise ValueError("an additive head takes one width per knob")
        net = AdditiveHead(hidden, out_features)
        last = net.output
    else:
        raise ValueError(f"unknown cond_head: {head}")
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return net


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

    # Per-example kernels are (batch, channels, state_dim, length) complex before the
    # sum over modes; recomputing them in the backward pass keeps one block's alive.
    checkpoint_kernels = True

    def forward(
        self,
        x: Tensor,
        film: Tensor | None = None,
        shift: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Returns (residual, skip). film: (batch, 2 * gate channels), scale then shift
        of the 1x1 pre-gate output; memoryless, so it keeps the SSM rate-independent.
        shift: per-example pole shifts of the SSM (see DiagonalSSMLayer.forward)."""
        if shift is None:
            h = self.ssm(x)
        elif self.checkpoint_kernels:
            h = torch.utils.checkpoint.checkpoint(
                self.ssm, x, shift, use_reentrant=False
            )
        else:
            h = self.ssm(x, shift)
        if self.act_type == "sine":
            h = self.activation(self.pre_act(h))
        else:
            h = self.pre_gate(h)
            if film is not None:
                scale, shift = film.unsqueeze(-1).chunk(2, dim=1)
                h = (1 + scale) * h + shift
            gate_a, gate_b = h.chunk(2, dim=1)
            h = torch.tanh(gate_a) * torch.sigmoid(gate_b)
        skip = self.skip_conv(h)
        residual = self.res_conv(h) + x
        return residual, skip

    def reset_state(self) -> None:
        self.ssm.reset_state()


def head_activation(name: str) -> nn.Module:
    """Activation of SSMWaveNet's output head (see its head_act argument)."""
    if name == "relu":
        return nn.ReLU()
    if name == "leaky":
        return nn.LeakyReLU(0.01)
    raise ValueError(f"unknown head_act: {name}")


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
        num_controls: knob values in [0, 1] per example. When > 0, a head maps
            them to a static conditioning, zero at initialisation so training starts
            from the unconditioned model.
        cond: what the knobs act on. "film": a FiLM (scale, shift) of every block's
            gated 1x1 layer, memoryless. "poles": POLE_SCALE * tanh of a shift of
            every channel's log_dt and every mode's log_A_real, so the knobs move
            the SSM poles as a tone stack moves its RC corners.
        cond_head: "joint", one MLP over all knobs (hidden widths: cond_hidden, one
            entry); "additive", one hidden layer per knob (widths: cond_hidden, one
            entry per knob) into a shared output layer, so the head is a sum of
            functions of one knob each.
        head_act: activation of the output head, "relu" (default) or "leaky"
            (LeakyReLU, slope 0.01). A ReLU head closes for good once every skip
            sum is negative, and no gradient then reaches the trunk; "leaky" keeps
            one. It has no parameters, so initialisation draws are unchanged.
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
        num_controls: int = 0,
        cond: str = "film",
        cond_head: str = "joint",
        cond_hidden: Sequence[int] = (32,),
        head_act: str = "relu",
    ) -> None:
        super().__init__()
        self.num_controls = num_controls
        self.cond = cond
        self.film_net: nn.Module | None = None
        self.pole_net: nn.Module | None = None
        if num_controls > 0:
            if cond == "film" and act_type != "gated":
                raise ValueError("controls need act_type='gated'")
            if cond == "poles" and discretization not in ("zoh", "foh"):
                raise ValueError("pole conditioning needs 'zoh' or 'foh'")
            if cond not in ("film", "poles"):
                raise ValueError(f"unknown cond: {cond}")
            film_size = num_blocks * 4 * channels
            size = (
                film_size if cond == "film" else num_blocks * channels * (1 + state_dim)
            )
            if (cond, cond_head, tuple(cond_hidden)) == ("film", "joint", (32,)):
                head = control_head(num_controls, size, cond_head, cond_hidden)
            else:
                # Any other head first consumes exactly the draws of the FiLM head of
                # 2026-09-23, then is built on a forked generator: the trunk, drawn
                # next, is the same in every arm at a given seed, so arms differ by
                # their conditioning only (demo/param/predictions_poles.md).
                control_head(num_controls, film_size, "joint", (32,))
                with torch.random.fork_rng(devices=[]):
                    # CPU generator only: torch.manual_seed would reseed CUDA as well,
                    # which fork_rng(devices=[]) does not restore.
                    torch.random.default_generator.manual_seed(
                        int(torch.randint(2**62, (1,)))
                    )
                    head = control_head(num_controls, size, cond_head, cond_hidden)
            if cond == "film":
                self.film_net = head
            else:
                self.pole_net = head
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
            head_activation(head_act),
            nn.Conv1d(channels, channels, 1),
            head_activation(head_act),
            nn.Conv1d(channels, output_channels, 1),
            final_act,
        )

    def forward(self, x: Tensor, controls: Tensor | None = None) -> Tensor:
        """x: (batch, 1, length), controls: (batch, num_controls) -> like x."""
        films: list[Tensor | None] = [None] * len(self.blocks)
        shifts: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        if self.film_net is not None:
            films = list(self.film_net(controls).chunk(len(self.blocks), dim=1))
        if self.pole_net is not None:
            shifts = self.pole_shifts(controls)
        h = self.input_conv(x)
        skip_sum = torch.zeros_like(h)
        for block, film, shift in zip(self.blocks, films, shifts, strict=True):
            h, skip = block(h, film, shift)
            skip_sum = skip_sum + skip
        return self.output_net(skip_sum)

    def pole_shifts(self, controls: Tensor) -> list[tuple[Tensor, Tensor]]:
        """Per block: (dlog_dt (batch, channels), dlog_A_real (batch, channels,
        state_dim)), each bounded by POLE_SCALE."""
        channels = self.blocks[0].ssm.channels
        modes = self.blocks[0].ssm.state_dim
        raw = POLE_SCALE * torch.tanh(self.pole_net(controls))
        raw = raw.view(len(controls), len(self.blocks), channels * (1 + modes))
        return [
            (raw[:, i, :channels], raw[:, i, channels:].reshape(-1, channels, modes))
            for i in range(len(self.blocks))
        ]

    def reset_states(self) -> None:
        for block in self.blocks:
            block.reset_state()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
