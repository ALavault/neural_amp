#!/usr/bin/env python3
"""T5 checks (THEORY.md): what the "poles" arm of SSMWaveNet conditions.

Symbolic: the DC gain of a ZOH / FOH mode, sum_k c w_k a^k, is -c/lambda and does not
depend on dt; for a real pole lambda = -e^(alpha + eps),
2 Re(w) = 2 Re(c) e^-(alpha + eps).
Numeric, on the real SSMWaveNet(cond="poles") in float64 with generic parameters, ZOH
and FOH:
  N1 path: two settings u0 != u1 give different outputs, but u1 run with the shifts of
     u0 gives exactly u0's output (no other route from the knobs);
  N2 per channel: kernel sum + D equals D + sum_m 2 Re(-c/lambda(u)); unchanged when
     only dlog_dt moves, changed when dlog_A_real moves;
  N3 network: the response to a constant input, at its last sample, is unchanged when
     only dlog_dt moves and changed when dlog_A_real moves; the same dt-only moves do
     change the response to noise (dt is not ignored).
Fails loudly.
"""

from __future__ import annotations

import math

import sympy as sp
import torch

from fssr_nam.models.ssm_wavenet import SSMWaveNet

LENGTH = 2**15
BOUND = 1.0  # random shifts in [-1, 1], so every mode dies within LENGTH samples


def symbolic() -> None:
    z, dt, alpha, eps = sp.symbols("z dt alpha epsilon")
    lam = z / dt
    a = sp.exp(z)

    def phi2(u):
        return (sp.exp(u) - 1 - u) / u**2

    zoh = ((a - 1) / lam) / (1 - a)
    foh = (dt * phi2(z) + dt * a * phi2(-z)) / (1 - a)
    for name, gain in (("ZOH", zoh), ("FOH", foh)):
        assert sp.simplify(gain + 1 / lam) == 0, name
        # At fixed lambda, dt enters only through z = lambda dt: d/d dt at fixed lambda.
        lam_s = sp.symbols("lambda", nonzero=True)
        fixed = gain.subs(z, lam_s * dt)
        assert sp.simplify(sp.diff(fixed, dt)) == 0, name
    real = sp.symbols("c_r", real=True)
    pole = -sp.exp(alpha + eps)
    assert sp.simplify(2 * (-real / pole) - 2 * real * sp.exp(-(alpha + eps))) == 0
    print("symbolique : gain continu -c/lambda, independant de dt, ZOH et FOH ;")
    print("  mode reel : 2 Re(w) = 2 Re(c) e^-(alpha+eps)")


def build(discretization: str, seed: int) -> SSMWaveNet:
    torch.manual_seed(seed)
    model = SSMWaveNet(
        num_blocks=3,
        channels=4,
        state_dim=3,
        discretization=discretization,
        num_controls=2,
        cond="poles",
        cond_head="joint",
        cond_hidden=(32,),
    ).double()
    with torch.no_grad():
        for block in model.blocks:
            ssm = block.ssm
            ssm.log_A_real.copy_(
                torch.empty_like(ssm.log_A_real).uniform_(0.2, 2).log()
            )
            ssm.A_imag.uniform_(-3, 3)
            ssm.log_dt.copy_(torch.empty_like(ssm.log_dt).uniform_(0.05, 0.5).log())
            ssm.C.normal_()
            ssm.D.normal_()
        model.pole_net[2].weight.normal_(0, 0.5)
        model.pole_net[2].bias.normal_(0, 0.5)
    return model.eval()


def random_shifts(model, move_dt: bool, move_poles: bool, gen) -> list:
    out = []
    for block in model.blocks:
        ssm = block.ssm
        dt = torch.rand(1, ssm.channels, generator=gen, dtype=torch.float64)
        poles = torch.rand(
            1, ssm.channels, ssm.state_dim, generator=gen, dtype=torch.float64
        )
        out.append(
            (
                (2 * dt - 1) * BOUND * move_dt,
                (2 * poles - 1) * BOUND * move_poles,
            )
        )
    return out


def closed_dc(ssm, shift) -> torch.Tensor:
    _, dlog_a = shift
    lam = torch.complex(-torch.exp(ssm.log_A_real + dlog_a[0]), ssm.A_imag)
    c = torch.view_as_complex(ssm.C.contiguous())
    return ssm.D + 2 * (-c / lam).sum(-1).real


def kernel_dc(ssm, shift) -> torch.Tensor:
    return ssm._shifted_kernel(LENGTH, shift)[0].sum(-1) + ssm.D


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max())


@torch.no_grad()
def numeric(discretization: str, seed: int) -> None:
    model = build(discretization, seed)
    gen = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(1, 1, 4096, generator=gen, dtype=torch.float64)
    u0, u1 = torch.rand(2, 1, 2, generator=gen, dtype=torch.float64)

    # N1: the knobs reach the output only through the pole shifts.
    y0, y1 = model(x, u0), model(x, u1)
    differ = float((y0 - y1).abs().max())
    assert differ > 1e-6, differ
    shifts0 = model.pole_shifts(u0)
    model.pole_shifts = lambda controls: shifts0
    assert torch.equal(model(x, u1), y0)
    del model.pole_shifts

    # N2: channel DC gain, closed form and dt-independence.
    worst_closed = worst_dt = 0.0
    least_poles = math.inf
    for _ in range(4):
        base = random_shifts(model, True, True, gen)
        other_dt = random_shifts(model, True, False, gen)
        other_poles = random_shifts(model, False, True, gen)
        for i, block in enumerate(model.blocks):
            ssm = block.ssm
            ref = kernel_dc(ssm, base[i])
            worst_closed = max(worst_closed, rel(ref, closed_dc(ssm, base[i])))
            moved_dt = (other_dt[i][0], base[i][1])
            worst_dt = max(worst_dt, rel(kernel_dc(ssm, moved_dt), ref))
            moved_poles = (base[i][0], other_poles[i][1])
            least_poles = min(least_poles, rel(kernel_dc(ssm, moved_poles), ref))
    assert worst_closed < 1e-9 and worst_dt < 1e-9, (worst_closed, worst_dt)
    assert least_poles > 1e-3, least_poles

    # N3: network response to a constant input, last sample.
    constant = torch.full((1, 1, LENGTH), 0.3, dtype=torch.float64)

    def last(shifts, signal=constant) -> torch.Tensor:
        model.pole_shifts = lambda controls: shifts
        out = model(signal, u0)[..., -1]
        del model.pole_shifts
        return out

    reference_shifts = random_shifts(model, True, True, gen)
    ref = last(reference_shifts)
    noise = torch.randn(1, 1, LENGTH, generator=gen, dtype=torch.float64)
    ref_noise = last(reference_shifts, noise)
    net_dt = net_poles = 0.0
    noise_dt = math.inf
    for _ in range(3):
        other = random_shifts(model, True, True, gen)
        dt_only = [(o[0], r[1]) for o, r in zip(other, reference_shifts, strict=True)]
        poles_only = [
            (r[0], o[1]) for o, r in zip(other, reference_shifts, strict=True)
        ]
        net_dt = max(net_dt, float((last(dt_only) - ref).abs().max()))
        net_poles = max(net_poles, float((last(poles_only) - ref).abs().max()))
        # dt does act away from DC: the same dt-only move changes a noise response.
        moved = float((last(dt_only, noise) - ref_noise).abs().max())
        noise_dt = min(noise_dt, moved)
    assert net_dt < 1e-9, net_dt
    assert noise_dt > 1e-4, noise_dt
    assert net_poles > 1e-4, net_poles
    print(
        f"{discretization} graine {seed} : N1 sorties distinctes ({differ:.2e}) et "
        "identiques au bit sous les decalages de u0 ; "
        f"N2 forme fermee {worst_closed:.1e}, "
        f"dt seul {worst_dt:.1e}, poles seuls >= {least_poles:.1e} ; "
        f"N3 dt seul {net_dt:.1e}, poles seuls {net_poles:.1e}, "
        f"dt seul sur bruit >= {noise_dt:.1e}"
    )


def main() -> None:
    symbolic()
    for discretization in ("zoh", "foh"):
        for seed in (0, 1, 2):
            numeric(discretization, seed)


if __name__ == "__main__":
    main()
