#!/usr/bin/env python3
"""T1 checks (THEORY.md): DC group delay of a ZOH / FOH discretized diagonal mode.

Symbolic: with a = e^z, b = (a - 1)/lambda (ZOH, S4D convention) and the FOH weights
beta1 = dt*phi2(z), beta0 = dt*e^z*phi2(-z) of ssm_wavenet.py, the first moments are
  ZOH: sum_k k K_k = 2 Re(-c a / (lambda (1 - a)))
  FOH: sum_k k K_k = 2 Re(c / (lambda^2 dt))   (equal to the continuous mode)
Numeric: kernels of the real DiagonalSSMLayer (generic random complex modes, each mode
alone and whole channels with D), moments against the closed forms. Fails loudly.
"""

from __future__ import annotations

import math

import sympy as sp
import torch

from fssr_nam.models.ssm_wavenet import DiagonalSSMLayer


def symbolic() -> None:
    z, dt = sp.symbols("z dt", nonzero=True)
    a = sp.exp(z)
    lam = z / dt

    def phi2(u):
        return (sp.exp(u) - 1 - u) / u**2

    beta1, beta0 = dt * phi2(z), dt * sp.exp(z) * phi2(-z)
    # FOH weights sum to the ZOH weight (DC kept), and FOH first moment is continuous.
    assert sp.simplify(beta1 + beta0 - (a - 1) / lam) == 0
    foh_moment = (beta1 * a + beta0) / (1 - a) ** 2  # coefficient of c
    assert sp.simplify(foh_moment - 1 / (lam**2 * dt)) == 0
    # ZOH first moment coefficient and the advance phi(z) = -1/z - 1/(e^-z - 1).
    b = (a - 1) / lam
    zoh_moment = b * a / (1 - a) ** 2
    cont_moment = 1 / (lam**2 * dt)
    w_coef = -1 / lam  # w = -c/lambda: DC of the mode is 2 Re(c b/(1-a)) = 2 Re(w)
    assert sp.simplify(b / (1 - a) - w_coef) == 0
    phi = -1 / z - 1 / (sp.exp(-z) - 1)
    assert sp.simplify((cont_moment - zoh_moment) - w_coef * phi) == 0
    series = sp.series(phi, z, 0, 4).removeO()
    assert sp.simplify(series - (sp.Rational(1, 2) + z / 12 - z**3 / 720)) == 0
    print("symbolique : FOH moment = continu ; ZOH avance = Re(w phi(z))/Re(w) ;")
    print(f"  phi(z) = 1/2 + z/12 - z^3/720 + O(z^5) : {series}")
    # Real modes, g(x) = x/(e^x - 1): g' + 1/2 = e^x (sinh x - x) / (e^x - 1)^2 >= 0,
    # and g' = -(x e^x - e^x + 1) / (e^x - 1)^2 < 0, so -1/2 < g' < 0 on x > 0.
    x = sp.symbols("x", positive=True)
    g = x / (sp.exp(x) - 1)
    lhs = (sp.diff(g, x) + sp.Rational(1, 2)) * (sp.exp(x) - 1) ** 2
    assert (
        sp.simplify(sp.expand(lhs - sp.exp(x) * (sp.sinh(x) - x)).rewrite(sp.exp)) == 0
    )
    neg = sp.diff(g, x) * (sp.exp(x) - 1) ** 2 + (x * sp.exp(x) - sp.exp(x) + 1)
    assert sp.simplify(neg) == 0
    print("  modes reels : -1/2 < g'(x) < 0, g = x/(e^x - 1) : kappa_mode dans ]0 ; 1[")


def moments(kernel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.arange(kernel.shape[-1], dtype=torch.float64)
    return kernel.sum(-1), (k * kernel).sum(-1)


def numeric() -> None:
    torch.manual_seed(20261002)
    worst = 0.0
    for discretization in ("zoh", "foh"):
        for trial in range(6):
            layer = DiagonalSSMLayer(
                channels=4, state_dim=4, discretization=discretization
            ).double()
            with torch.no_grad():
                layer.log_A_real.uniform_(math.log(0.05), math.log(3.0))
                layer.A_imag.uniform_(-3.0, 3.0)
                layer.log_dt.uniform_(math.log(0.02), math.log(1.5))
                layer.C.normal_()
                layer.D.normal_()
            lam = torch.complex(-torch.exp(layer.log_A_real), layer.A_imag).to(
                torch.complex128
            )
            dt = torch.exp(layer.log_dt).to(torch.float64).unsqueeze(-1)
            z = lam * dt
            c = torch.complex(layer.C[..., 0], layer.C[..., 1]).to(torch.complex128)
            w = -c / lam
            phi = -1 / z - 1 / (torch.exp(-z) - 1)
            # Tail e^{-s N} negligible for the slowest mode (s >= 1e-3): N = 60 / 1e-3.
            length = 60000
            saved = layer.C.detach().clone()
            for m in range(4):  # each mode alone
                with torch.no_grad():
                    layer.C.zero_()
                    layer.C[:, m] = saved[:, m]
                    kernel = layer._kernel(length).to(torch.float64)
                dc, first = moments(kernel)
                tau_disc = first / dc
                tau_cont = (2 * (c[:, m] / (lam[:, m] ** 2 * dt[:, 0])).real) / dc
                advance = tau_cont - tau_disc
                predicted = (
                    (w[:, m] * phi[:, m]).real / w[:, m].real
                    if discretization == "zoh"
                    else torch.zeros_like(advance)
                )
                assert torch.allclose(dc, 2 * w[:, m].real, rtol=1e-6, atol=1e-9), (
                    "DC gain"
                )
                err = (advance - predicted).abs().max().item()
                worst = max(worst, err)
                scale = max(1.0, tau_cont.abs().max().item())
                assert err < 1e-9 * scale + 1e-9, (discretization, trial, m, err)
            with torch.no_grad():
                layer.C.copy_(saved)
                kernel = layer._kernel(length).to(torch.float64)
            dc, first = moments(kernel)
            dc = dc + layer.D.to(torch.float64)  # D: gain, zero first moment
            cont = (2 * (c / (lam**2 * dt)).real).sum(-1)
            channel_advance = (cont - first) / dc
            predicted = (
                (2 * (w * phi).real).sum(-1) / dc
                if discretization == "zoh"
                else torch.zeros_like(dc)
            )
            err = (channel_advance - predicted).abs().max().item()
            worst = max(worst, err)
            assert err < 1e-6, (discretization, trial, channel_advance, predicted)
    # Real modes rescaled by Delta -> Delta / r (rate r times higher), ZOH: the lag of
    # the first moment, in samples at the original rate, over (1 - 1/r)/2, is
    # kappa = 2 (g(s/r) - g(s)) / (s (1 - 1/r)) with s = -lambda dt, in (0, 1).
    worst_kappa = 0.0
    for r in (2, 4):
        layer = DiagonalSSMLayer(channels=4, state_dim=4, discretization="zoh").double()
        with torch.no_grad():
            layer.log_A_real.uniform_(math.log(0.05), math.log(3.0))
            layer.A_imag.zero_()
            layer.log_dt.uniform_(math.log(0.02), math.log(1.5))
            layer.C.normal_()
        s = (
            torch.exp(layer.log_A_real) * torch.exp(layer.log_dt).unsqueeze(-1)
        ).double()
        saved, base = layer.C.detach().clone(), layer.log_dt.detach().clone()
        for m in range(4):
            taus = []
            for rate, length in ((1, 60000), (r, 60000 * r)):
                with torch.no_grad():
                    layer.C.zero_()
                    layer.C[:, m] = saved[:, m]
                    layer.log_dt.copy_(base - math.log(rate))
                    dc, first = moments(layer._kernel(length).to(torch.float64))
                taus.append(first / dc / rate)  # in samples at the original rate
            kappa = (taus[1] - taus[0]) / ((1 - 1 / r) / 2)
            u = s[:, m]
            g_slow, g_fast = (u / r) / torch.expm1(u / r), u / torch.expm1(u)
            predicted = 2 * (g_slow - g_fast) / (u * (1 - 1 / r))
            err = (kappa - predicted).abs().max().item()
            worst_kappa = max(worst_kappa, err)
            assert err < 1e-6, (r, m, kappa, predicted)
            assert ((kappa > 0) & (kappa < 1)).all(), kappa
        with torch.no_grad():
            layer.C.copy_(saved)
            layer.log_dt.copy_(base)
    print(
        f"  kappa_mode reel mesure = formule, dans ]0 ; 1[, ecart max {worst_kappa:.2e}"
    )
    # Real modes: 1/s - 1/(e^s - 1) lies in (0, 1/2), decreasing in s.
    s = torch.logspace(-4, 2, 200, dtype=torch.float64)
    real_adv = 1 / s - 1 / torch.expm1(s)
    assert ((real_adv > 0) & (real_adv < 0.5)).all()
    assert (real_adv.diff() < 0).all()
    print(
        f"numerique : 96 modes et 24 canaux par discretisation, ecart max {worst:.2e}"
    )
    print("  modes reels : 1/s - 1/(e^s - 1) dans ]0 ; 1/2[, decroissante")


if __name__ == "__main__":
    symbolic()
    numeric()
