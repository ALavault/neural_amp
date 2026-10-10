"""Deeper grey-box amp model: GreyBoxAmp stages in series, with a smooth bias and slow
envelopes (DIAGNOSIS_greybox.md, D-0014). Written for this repository, no KIPR code.

Chain: input FIR -> stage 1 -> ... -> stage S -> output FIR. A stage has `branches`
branches [first-order filter -> drive compressed by envelopes of |u| -> odd soft
clipper of (drive u + bias) between sign-dependent gains, minus the clipper of the
bias alone -> first-order filter], mixed by a 1x1 convolution. Stages after the
first add a direct path. Every addition starts neutral, so the model contains
GreyBoxAmp exactly: with stages=1, taps=32, bias=False and no slow envelope it computes
the same function from the same coefficients (`from_greybox`).
- bias: b + sum_k k_k e_k before the clipper, the clipper of that bias subtracted, so
  zero input gives zero output; b = k = 0 at the start.
- slow envelopes: one-pole envelopes of |u| at `slow_s` (time constants, learnable);
  they scale the drive by exp(-sum_k g_k e_k) and feed the bias; g = k = 0 at the start.
- extra stages: direct path of weight 1 and branch mix of weight 0 at the start.
Everything is feedforward: filters and envelopes are linear recurrences, computed as
FFT convolutions over a segment and by `step` sample by sample, from a zero state.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from fssr_nam.models.greybox import (
    COEFFICIENTS,
    GreyBoxAmp,
    causal_convolve,
    first_order_kernel,
    signed,
    soft_clip,
)


class DeepGreyBoxAmp(nn.Module):
    """x (batch, 1, length), controls (batch, num_controls) -> (batch, 1, length)."""

    def __init__(
        self,
        num_controls: int = 0,
        stages: int = 1,
        branches: int = 4,
        taps: int = 32,
        bias: bool = False,
        slow_s: tuple[float, ...] = (),
        hidden: int = 32,
        sample_rate: int = 48_000,
    ) -> None:
        super().__init__()
        self.num_controls = num_controls
        self.stages = stages
        self.branches = branches
        self.taps = taps
        self.bias = bias
        self.slow = len(slow_s)
        self.names = list(COEFFICIENTS)
        if bias:
            self.names.append("bias")
        for k in range(self.slow):
            self.names += [f"slow_alpha_{k}", f"slow_gain_{k}"]
            if bias:
                self.names.append(f"slow_bias_{k}")
        delta = torch.zeros(taps)
        delta[0] = 1
        self.pre_fir = nn.Parameter(delta.clone())
        self.post_fir = nn.Parameter(delta.clone())
        self.base = nn.ModuleList()
        self.contract = nn.ModuleList()
        self.direct = nn.ParameterList()
        for s in range(stages):
            raw = self._initial(branches, slow_s, sample_rate)
            self.base.append(
                nn.ParameterDict({k: nn.Parameter(raw[k]) for k in self.names})
            )
            contract = nn.Conv1d(branches, 1, 1, bias=False)
            nn.init.constant_(contract.weight, 1 / branches if s == 0 else 0.0)
            self.contract.append(contract)
            if s > 0:
                self.direct.append(nn.Parameter(torch.ones(())))
        self.head: nn.Module | None = None
        if num_controls > 0:
            self.head = nn.Sequential(
                nn.Linear(num_controls, hidden),
                nn.ReLU(),
                nn.Linear(hidden, stages * branches * len(self.names)),
            )
            nn.init.zeros_(self.head[2].weight)
            nn.init.zeros_(self.head[2].bias)
        structural = [self.pre_fir, self.post_fir, *self.direct]
        for s in range(stages):
            structural += [self.contract[s].weight, *self.base[s].values()]
        for p in structural:
            p._optim = {"weight_decay": 0.0}
        self._state: dict | None = None

    def _initial(
        self, branches: int, slow_s: tuple[float, ...], sample_rate: int
    ) -> dict[str, Tensor]:
        """Raw starting coefficients of a stage: GreyBoxAmp's, additions neutral."""
        spread = torch.linspace(0, 1, branches)
        raw = {
            "pre_pole": torch.atanh((0.5 + 0.45 * spread) / 0.9995),
            "pre_b1": torch.zeros(branches),
            "post_pole": torch.full((branches,), math.atanh(0.5)),
            "post_b1": torch.zeros(branches),
            "log_drive": math.log(8) * spread,
            "beta_in": torch.zeros(branches),
            "beta_out": torch.zeros(branches),
            "alpha": torch.full((branches,), -math.log(sample_rate * 0.005 - 1)),
            "compression": torch.zeros(branches),
            "hardness": torch.zeros(branches),
        }
        if self.bias:
            raw["bias"] = torch.zeros(branches)
        for k, tau in enumerate(slow_s):
            raw[f"slow_alpha_{k}"] = torch.full(
                (branches,), -math.log(sample_rate * tau - 1)
            )
            raw[f"slow_gain_{k}"] = torch.zeros(branches)
            if self.bias:
                raw[f"slow_bias_{k}"] = torch.zeros(branches)
        return raw

    @classmethod
    def from_greybox(
        cls, grey: GreyBoxAmp, controls: Tensor | None = None, **kwargs
    ) -> DeepGreyBoxAmp:
        """Unconditioned model holding GREY's coefficients at `controls` (raw base
        plus the head's offsets) in stage 1; FIRs zero-padded to `taps`; additions
        neutral."""
        dtype = grey.pre_fir.dtype
        model = cls(num_controls=0, branches=grey.branches, **kwargs).to(dtype)
        with torch.no_grad():
            raw = {k: v.detach().clone() for k, v in grey.base.items()}
            if grey.head is not None:
                offsets = grey.head(controls[None]).view(len(COEFFICIENTS), -1)
                raw = {k: raw[k] + offsets[i] for i, k in enumerate(COEFFICIENTS)}
            for k, v in raw.items():
                model.base[0][k].copy_(v)
            for mine, theirs in (
                (model.pre_fir, grey.pre_fir),
                (model.post_fir, grey.post_fir),
            ):
                mine.zero_()
                mine[: grey.taps].copy_(theirs)
            model.contract[0].weight.copy_(grey.contract.weight)
        return model

    def coefficients(self, controls: Tensor | None, batch: int) -> list[dict]:
        """Bounded coefficients per stage, each (batch, branches, 1)."""
        offsets = None
        if self.head is not None:
            offsets = self.head(controls).view(
                batch, self.stages, len(self.names), self.branches
            )
        out = []
        for s in range(self.stages):
            raw = {k: v.expand(batch, -1) for k, v in self.base[s].items()}
            if offsets is not None:
                raw = {k: raw[k] + offsets[:, s, i] for i, k in enumerate(self.names)}
            c = {
                "pre_pole": 0.9995 * torch.tanh(raw["pre_pole"]),
                "pre_b1": torch.tanh(raw["pre_b1"]),
                "post_pole": 0.9995 * torch.tanh(raw["post_pole"]),
                "post_b1": torch.tanh(raw["post_b1"]),
                "drive": torch.exp(raw["log_drive"].clamp(-6, 6)),
                "beta_in": raw["beta_in"].clamp(-3, 3),
                "beta_out": raw["beta_out"].clamp(-3, 3),
                "alpha": torch.sigmoid(raw["alpha"]),
                "log_retain": nn.functional.logsigmoid(-raw["alpha"]),
                "compression": nn.functional.softplus(raw["compression"]),
                "hardness": 1 + nn.functional.softplus(raw["hardness"]).clamp(max=20),
            }
            if self.bias:
                c["bias"] = raw["bias"].clamp(-3, 3)
            for k in range(self.slow):
                c[f"slow_alpha_{k}"] = torch.sigmoid(raw[f"slow_alpha_{k}"])
                c[f"slow_log_retain_{k}"] = nn.functional.logsigmoid(
                    -raw[f"slow_alpha_{k}"]
                )
                c[f"slow_gain_{k}"] = raw[f"slow_gain_{k}"]
                if self.bias:
                    c[f"slow_bias_{k}"] = raw[f"slow_bias_{k}"]
            out.append({k: v.unsqueeze(-1) for k, v in c.items()})
        return out

    def _shape(self, u: Tensor, envelopes: list[Tensor], c: dict) -> Tensor:
        """Drive, bias and clipper of one stage, given its envelopes (fast first)."""
        log_scale = sum(
            (c[f"slow_gain_{k}"] * envelopes[1 + k] for k in range(self.slow)),
            torch.zeros_like(u),
        )
        drive = (
            c["drive"] * torch.exp(-log_scale) / (1 + c["compression"] * envelopes[0])
        )
        if not self.bias:
            y = soft_clip(signed(u * drive, c["beta_in"]), c["hardness"])
            return signed(y, c["beta_out"])
        offset = c["bias"] + sum(
            (c[f"slow_bias_{k}"] * envelopes[1 + k] for k in range(self.slow)),
            torch.zeros_like(u),
        )
        y = soft_clip(signed(u * drive + offset, c["beta_in"]), c["hardness"])
        y = y - soft_clip(signed(offset, c["beta_in"]), c["hardness"])
        return signed(y, c["beta_out"])

    def forward(self, x: Tensor, controls: Tensor | None = None) -> Tensor:
        batch, _, length = x.shape
        n = torch.arange(length, device=x.device, dtype=x.dtype)
        v = causal_convolve(x, self.pre_fir.view(1, 1, -1).to(x.dtype))
        for s, c in enumerate(self.coefficients(controls, batch)):
            ones = torch.ones_like(c["pre_pole"][..., 0])
            pre = first_order_kernel(
                ones, c["pre_b1"][..., 0], c["pre_pole"][..., 0], length
            )
            u = causal_convolve(v.expand(-1, self.branches, -1), pre)
            envelopes = [
                causal_convolve(u.abs(), c["alpha"] * torch.exp(n * c["log_retain"]))
            ]
            for k in range(self.slow):
                kernel = c[f"slow_alpha_{k}"] * torch.exp(n * c[f"slow_log_retain_{k}"])
                envelopes.append(causal_convolve(u.abs(), kernel))
            y = self._shape(u, envelopes, c)
            post = first_order_kernel(
                ones, c["post_b1"][..., 0], c["post_pole"][..., 0], length
            )
            y = self.contract[s](causal_convolve(y, post))
            v = y if s == 0 else self.direct[s - 1] * v + y
        return causal_convolve(v, self.post_fir.view(1, 1, -1).to(x.dtype))

    def reset_states(self) -> None:
        self._state = None

    def step(self, x: Tensor, controls: Tensor | None = None) -> Tensor:
        """Single sample: x (batch, 1) -> (batch, 1); state kept until reset_states."""
        batch = x.shape[0]
        stages = [
            {k: v[..., 0] for k, v in c.items()}
            for c in self.coefficients(controls, batch)
        ]
        if self._state is None:
            zeros = x.new_zeros(batch, self.branches)
            self._state = {
                "pre_fir": x.new_zeros(batch, self.taps),
                "post_fir": x.new_zeros(batch, self.taps),
                "stages": [
                    {
                        "pre_x": zeros,
                        "pre_y": zeros,
                        "post_x": zeros,
                        "post_y": zeros,
                        "envelopes": [zeros] * (1 + self.slow),
                    }
                    for _ in range(self.stages)
                ],
            }
        state = self._state
        state["pre_fir"] = torch.cat([x, state["pre_fir"][:, :-1]], dim=1)
        v = (state["pre_fir"] * self.pre_fir).sum(1, keepdim=True)
        for s, (c, st) in enumerate(zip(stages, state["stages"], strict=True)):
            u_in = v.expand(-1, self.branches)
            u = u_in + c["pre_b1"] * st["pre_x"] + c["pre_pole"] * st["pre_y"]
            st["pre_x"], st["pre_y"] = u_in, u
            alphas = [(c["alpha"], c["log_retain"])] + [
                (c[f"slow_alpha_{k}"], c[f"slow_log_retain_{k}"])
                for k in range(self.slow)
            ]
            st["envelopes"] = [
                torch.exp(log_retain) * e + alpha * u.abs()
                for e, (alpha, log_retain) in zip(st["envelopes"], alphas, strict=True)
            ]
            w = self._shape(u, st["envelopes"], c)
            z = w + c["post_b1"] * st["post_x"] + c["post_pole"] * st["post_y"]
            st["post_x"], st["post_y"] = w, z
            mixed = (z * self.contract[s].weight[0, :, 0]).sum(1, keepdim=True)
            v = mixed if s == 0 else self.direct[s - 1] * v + mixed
        state["post_fir"] = torch.cat([v, state["post_fir"][:, :-1]], dim=1)
        return (state["post_fir"] * self.post_fir).sum(1, keepdim=True)
