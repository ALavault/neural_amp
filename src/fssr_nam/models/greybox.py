"""Grey-box amp model in the spirit of the KIPR engine (demo/greybox/predictions.md).

Structure, written for this repository (no code from the KIPR archive): input FIR ->
per branch [first-order filter -> drive compressed by an envelope of |u| -> odd soft
clipper between sign-dependent gains -> first-order filter] -> mix of the branches ->
output FIR. A few hundred parameters. The knobs drive the coefficients through a small
head whose last layer starts at zero, so training starts from the unconditioned model.
With the asymmetries at zero the model is odd in its input (no bias anywhere; the
envelope is an even function of the signal). Filters and envelope are linear
recurrences, computed as FFT convolutions over a segment and by `step` sample by
sample; both start from a zero state.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from fssr_nam.models.ssm_wavenet import DiagonalSSMLayer, fft_size

# Head outputs per branch, in order.
COEFFICIENTS = (
    "pre_pole",
    "pre_b1",
    "post_pole",
    "post_b1",
    "log_drive",
    "beta_in",
    "beta_out",
    "alpha",
    "compression",
    "hardness",
)


def first_order_kernel(b0: Tensor, b1: Tensor, pole: Tensor, length: int) -> Tensor:
    """Impulse response of y[n] = b0 x[n] + b1 x[n-1] + pole y[n-1], (..., length)."""
    n = torch.arange(length - 1, device=pole.device, dtype=pole.dtype)
    tail = (b0 * pole + b1).unsqueeze(-1) * pole.unsqueeze(-1) ** n
    return torch.cat([b0.unsqueeze(-1), tail], dim=-1)


def causal_convolve(x: Tensor, kernel: Tensor) -> Tensor:
    """x (batch, channels, length) * kernel (batch or 1, channels, any length),
    causal; the FFT is long enough for both, so nothing wraps around."""
    length = x.shape[-1]
    fft_length = fft_size(length + kernel.shape[-1] - 1, DiagonalSSMLayer.fft_size)
    product = torch.fft.rfft(x, n=fft_length) * torch.fft.rfft(kernel, n=fft_length)
    return torch.fft.irfft(product, n=fft_length)[..., :length]


def soft_clip(x: Tensor, hardness: Tensor) -> Tensor:
    """Odd algebraic soft clipper x / (1 + |x|^q)^(1/q), q >= 1, computed as
    x * exp(-softplus(q log|x|) / q) so that large |x|^q cannot overflow."""
    log_abs = torch.log(x.abs().clamp_min(1e-12))
    return x * torch.exp(-nn.functional.softplus(hardness * log_abs) / hardness)


def signed(x: Tensor, beta: Tensor) -> Tensor:
    """Positive values scaled by exp(beta), negative ones by exp(-beta)."""
    return x * torch.where(x >= 0, torch.exp(beta), torch.exp(-beta))


class GreyBoxAmp(nn.Module):
    """x (batch, 1, length), controls (batch, num_controls) -> (batch, 1, length)."""

    def __init__(
        self, num_controls: int = 0, branches: int = 4, taps: int = 32, hidden: int = 32
    ) -> None:
        super().__init__()
        self.num_controls = num_controls
        self.branches = branches
        self.taps = taps
        delta = torch.zeros(taps)
        delta[0] = 1
        self.pre_fir = nn.Parameter(delta.clone())
        self.post_fir = nn.Parameter(delta.clone())
        # Branch constants, spread so the branches differ from the start: poles
        # (pre: high-pass-ish to low-pass-ish), drives from 1 to 8.
        spread = torch.linspace(0, 1, branches)
        self.base = nn.ParameterDict(
            {
                "pre_pole": nn.Parameter(torch.atanh((0.5 + 0.45 * spread) / 0.9995)),
                "pre_b1": nn.Parameter(torch.zeros(branches)),
                "post_pole": nn.Parameter(torch.full((branches,), math.atanh(0.5))),
                "post_b1": nn.Parameter(torch.zeros(branches)),
                "log_drive": nn.Parameter(math.log(8) * spread),
                "beta_in": nn.Parameter(torch.zeros(branches)),
                "beta_out": nn.Parameter(torch.zeros(branches)),
                # Envelope time constant 5 ms at 48 kHz.
                "alpha": nn.Parameter(
                    torch.full((branches,), -math.log(48_000 * 0.005 - 1))
                ),
                "compression": nn.Parameter(torch.zeros(branches)),
                "hardness": nn.Parameter(torch.zeros(branches)),
            }
        )
        # Mix of the branches; weight only, so a sign flip negates the output.
        self.contract = nn.Conv1d(branches, 1, 1, bias=False)
        nn.init.constant_(self.contract.weight, 1 / branches)
        self.head: nn.Module | None = None
        if num_controls > 0:
            self.head = nn.Sequential(
                nn.Linear(num_controls, hidden),
                nn.ReLU(),
                nn.Linear(hidden, branches * len(COEFFICIENTS)),
            )
            nn.init.zeros_(self.head[2].weight)
            nn.init.zeros_(self.head[2].bias)
        # Structural coefficients are not pulled toward zero: AdamW without weight
        # decay for them, as for the SSM poles (bench --honor-optim).
        for p in [
            self.pre_fir,
            self.post_fir,
            self.contract.weight,
            *self.base.values(),
        ]:
            p._optim = {"weight_decay": 0.0}
        self._state: dict | None = None

    def coefficients(self, controls: Tensor | None, batch: int) -> dict[str, Tensor]:
        """Bounded coefficients, each (batch, branches, 1)."""
        raw = {k: v.expand(batch, -1) for k, v in self.base.items()}
        if self.head is not None:
            offsets = self.head(controls).view(batch, len(COEFFICIENTS), self.branches)
            raw = {k: raw[k] + offsets[:, i] for i, k in enumerate(COEFFICIENTS)}
        c = {
            "pre_pole": 0.9995 * torch.tanh(raw["pre_pole"]),
            "pre_b1": torch.tanh(raw["pre_b1"]),
            "post_pole": 0.9995 * torch.tanh(raw["post_pole"]),
            "post_b1": torch.tanh(raw["post_b1"]),
            "drive": torch.exp(raw["log_drive"].clamp(-6, 6)),
            "beta_in": raw["beta_in"].clamp(-3, 3),
            "beta_out": raw["beta_out"].clamp(-3, 3),
            "alpha": torch.sigmoid(raw["alpha"]),
            # log(1 - alpha), finite even where alpha rounds to 1.
            "log_retain": nn.functional.logsigmoid(-raw["alpha"]),
            "compression": nn.functional.softplus(raw["compression"]),
            "hardness": 1 + nn.functional.softplus(raw["hardness"]).clamp(max=20),
        }
        return {k: v.unsqueeze(-1) for k, v in c.items()}

    def forward(self, x: Tensor, controls: Tensor | None = None) -> Tensor:
        batch, _, length = x.shape
        c = self.coefficients(controls, batch)
        # Causal FIR: y[n] = sum_k h[k] x[n - k], the kernel taken as is.
        u = causal_convolve(x, self.pre_fir.view(1, 1, -1).to(x.dtype))
        ones = torch.ones_like(c["pre_pole"][..., 0])
        pre = first_order_kernel(
            ones, c["pre_b1"][..., 0], c["pre_pole"][..., 0], length
        )
        u = causal_convolve(u.expand(-1, self.branches, -1), pre)
        n = torch.arange(length, device=x.device, dtype=x.dtype)
        envelope = causal_convolve(u.abs(), c["alpha"] * torch.exp(n * c["log_retain"]))
        drive = c["drive"] / (1 + c["compression"] * envelope)
        y = soft_clip(signed(u * drive, c["beta_in"]), c["hardness"])
        y = signed(y, c["beta_out"])
        post = first_order_kernel(
            ones, c["post_b1"][..., 0], c["post_pole"][..., 0], length
        )
        y = self.contract(causal_convolve(y, post))
        return causal_convolve(y, self.post_fir.view(1, 1, -1).to(x.dtype))

    def reset_states(self) -> None:
        self._state = None

    def step(self, x: Tensor, controls: Tensor | None = None) -> Tensor:
        """Single sample: x (batch, 1) -> (batch, 1); state kept until reset_states."""
        batch = x.shape[0]
        c = {k: v[..., 0] for k, v in self.coefficients(controls, batch).items()}
        if self._state is None:
            zeros = x.new_zeros(batch, self.branches)
            self._state = {
                "pre_fir": x.new_zeros(batch, self.taps),
                "post_fir": x.new_zeros(batch, self.taps),
                "pre_x": zeros,
                "pre_y": zeros,
                "post_x": zeros,
                "post_y": zeros,
                "envelope": zeros,
            }
        s = self._state
        s["pre_fir"] = torch.cat([x, s["pre_fir"][:, :-1]], dim=1)
        u = (s["pre_fir"] * self.pre_fir).sum(1, keepdim=True).expand(-1, self.branches)
        v = u + c["pre_b1"] * s["pre_x"] + c["pre_pole"] * s["pre_y"]
        s["pre_x"], s["pre_y"] = u, v
        retain = torch.exp(c["log_retain"])
        s["envelope"] = retain * s["envelope"] + c["alpha"] * v.abs()
        drive = c["drive"] / (1 + c["compression"] * s["envelope"])
        w = soft_clip(signed(v * drive, c["beta_in"]), c["hardness"])
        w = signed(w, c["beta_out"])
        z = w + c["post_b1"] * s["post_x"] + c["post_pole"] * s["post_y"]
        s["post_x"], s["post_y"] = w, z
        mixed = (z * self.contract.weight[0, :, 0]).sum(1, keepdim=True)
        s["post_fir"] = torch.cat([mixed, s["post_fir"][:, :-1]], dim=1)
        return (s["post_fir"] * self.post_fir).sum(1, keepdim=True)
