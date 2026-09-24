"""First-order-hold discretization of SSM-WaveNet's diagonal SSMs (FIX_foh.md).

Float64 throughout. Time is in the model's own units: sample n sits at t = n dt, with
dt = exp(log_dt) per channel.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch

from fssr_nam.models.ssm_wavenet import (
    DiagonalSSMLayer,
    SSMWaveNet,
    phi2,
    two_tap_kernel,
    zoh_to_foh,
)

CHANNELS, MODES = 3, 4


def layer(discretization: str = "foh", log_dt: float | None = None) -> DiagonalSSMLayer:
    torch.manual_seed(7)
    ssm = DiagonalSSMLayer(CHANNELS, MODES, discretization=discretization).double()
    with torch.no_grad():
        ssm.log_A_real.normal_(-1.0, 0.5)
        ssm.D.normal_()
        if log_dt is not None:
            ssm.log_dt.fill_(log_dt)
    return ssm.eval()


def lam(ssm: DiagonalSSMLayer) -> torch.Tensor:
    return torch.complex(-torch.exp(ssm.log_A_real), ssm.A_imag).detach()


def c(ssm: DiagonalSSMLayer) -> torch.Tensor:
    return torch.view_as_complex(ssm.C.detach().contiguous())


def modal_ramp_and_step(ssm: DiagonalSSMLayer, length: int) -> tuple:
    """Exact continuous responses of the modes (without D) to u(t) = t and to u(t) = 1,
    sampled at t = n dt: each mode's state is t^2 phi2(lambda t), resp. t phi1."""
    dt = torch.exp(ssm.log_dt.detach())
    t = (dt.unsqueeze(-1) * torch.arange(length, dtype=torch.float64)).unsqueeze(1)
    z = lam(ssm).unsqueeze(-1) * t
    ramp = 2 * torch.einsum("ch,chl->cl", c(ssm), t**2 * phi2(z)).real
    step = (
        2
        * torch.einsum(
            "ch,chl->cl", c(ssm), torch.expm1(z) / lam(ssm).unsqueeze(-1)
        ).real
    )
    return ramp, step


def steps(ssm: DiagonalSSMLayer, x: torch.Tensor) -> torch.Tensor:
    return torch.stack([ssm.step(x[..., n]) for n in range(x.shape[-1])], dim=-1)


def test_same_parameters_and_draws_as_zoh() -> None:
    """Pairs FOH with ZOH runs: the same seed gives the same initial model."""
    torch.manual_seed(3)
    zoh = SSMWaveNet(num_blocks=2, channels=4, state_dim=4, discretization="zoh")
    after_zoh = torch.rand(3)
    torch.manual_seed(3)
    foh = SSMWaveNet(num_blocks=2, channels=4, state_dim=4, discretization="foh")
    after_foh = torch.rand(3)
    assert torch.equal(after_zoh, after_foh)
    for (name, p), (_, q) in zip(
        zoh.state_dict().items(), foh.state_dict().items(), strict=True
    ):
        assert torch.equal(p, q), name
    y = foh(torch.randn(2, 1, 500))
    assert y.shape == (2, 1, 500)
    assert torch.isfinite(y).all()


def test_fft_matches_step_and_reset() -> None:
    ssm = layer()
    x = torch.randn(2, CHANNELS, 400, dtype=torch.float64)
    with torch.no_grad():
        full = ssm(x)
        ssm.reset_state()
        stepped = steps(ssm, x)
        # A reset must forget x[n-1] as well as the state.
        ssm.reset_state()
        again = steps(ssm, x[..., 100:250])
    torch.testing.assert_close(stepped, full, rtol=0, atol=1e-10)
    torch.testing.assert_close(again, ssm(x[..., 100:250]), rtol=0, atol=1e-10)


def test_causal() -> None:
    ssm = layer()
    x = torch.randn(1, CHANNELS, 600, dtype=torch.float64)
    changed = x.clone()
    changed[..., 300:] = torch.randn(1, CHANNELS, 300, dtype=torch.float64)
    with torch.no_grad():
        before, after = ssm(x), ssm(changed)
    # FFT round-off only (5e-16 measured).
    torch.testing.assert_close(before[..., :300], after[..., :300], rtol=0, atol=1e-12)


def test_dc_exact() -> None:
    ssm = layer(log_dt=math.log(0.05))
    with torch.no_grad():
        y = ssm(torch.ones(1, CHANNELS, 3000, dtype=torch.float64))
    expected = 2 * (-c(ssm) / lam(ssm)).sum(-1).real + ssm.D.detach()
    torch.testing.assert_close(y[0, :, -1], expected, rtol=1e-12, atol=0)


def test_piecewise_linear_input_exact_across_rates() -> None:
    """A signal linear between the 48 kHz samples is represented exactly at 96 kHz too;
    FOH samples the same continuous response at both rates, so the outputs coincide at
    the common instants. ZOH does not, which is the rate dependence FOH removes."""
    # x[-1] = 0 is implied: the first knot at t = 0 must be 0 too, or the ramp from it
    # would last dt at 48 kHz and dt / 2 at 96 kHz, two different inputs.
    x48 = torch.randn(1, CHANNELS, 300, dtype=torch.float64)
    x48[..., 0] = 0
    x96 = torch.zeros(1, CHANNELS, 600, dtype=torch.float64)
    x96[..., 0::2] = x48
    x96[..., 1:-1:2] = (x48[..., :-1] + x48[..., 1:]) / 2
    for discretization, exact in (("foh", True), ("zoh", False)):
        ssm = layer(discretization)
        with torch.no_grad():
            y48 = ssm(x48)
            ssm.log_dt -= math.log(2)
            y96 = ssm(x96)[..., 0::2]
        error = (y96[..., :-1] - y48[..., :-1]).abs().max()
        assert (error < 1e-10) == exact, (discretization, float(error))


def test_ramp_separates_zoh_from_foh() -> None:
    """On a ramp FOH gives the exact continuous response. ZOH feeds its modes the ramp
    half a sample early (x[n] held over the interval before t_n), its direct term D
    x[n] on time: that advance accounts for most of its error."""
    length = 200
    foh, zoh = layer("foh"), layer("zoh")
    dt = torch.exp(foh.log_dt.detach()).unsqueeze(-1)
    ramp = dt * torch.arange(length, dtype=torch.float64)
    modes, step = modal_ramp_and_step(foh, length)
    direct = foh.D.detach().unsqueeze(-1) * ramp
    with torch.no_grad():
        out_foh = foh(ramp.unsqueeze(0))[0]
        out_zoh = zoh(ramp.unsqueeze(0))[0]
    torch.testing.assert_close(out_foh, modes + direct, rtol=0, atol=1e-10)
    error_zoh = (out_zoh - modes - direct).abs().max()
    assert error_zoh > 1e-4 * (modes + direct).abs().max()
    advanced = modes + (dt / 2) * step + direct
    assert (out_zoh - advanced).abs().max() < error_zoh / 5


def test_zero_rate_shift_is_bitwise_neutral() -> None:
    ssm = layer()
    x = torch.randn(1, CHANNELS, 300, dtype=torch.float64)
    with torch.no_grad():
        reference = ssm(x)
        ssm.log_dt += math.log(48_000 / 48_000)
        assert torch.equal(ssm(x), reference)


@pytest.mark.parametrize("scale", [1e-14, 1e-8, 0.99e-4, 1.01e-4, 1e-2])
def test_phi2_series_and_closed_form_agree(scale: float) -> None:
    z = scale * torch.exp(1j * torch.linspace(0, 2 * math.pi, 7, dtype=torch.float64))
    reference = 0.5 + z / 6 + z**2 / 24 + z**3 / 120 + z**4 / 720
    torch.testing.assert_close(phi2(z), reference, rtol=1e-7, atol=0)


def test_gradients_finite_near_zero() -> None:
    ssm = layer(log_dt=-30.0)  # |z| ~ 1e-13 on every mode
    with torch.no_grad():
        ssm.A_imag.zero_()
    y = ssm(torch.randn(1, CHANNELS, 64, dtype=torch.float64))
    y.square().sum().backward()
    for p in (ssm.log_dt, ssm.log_A_real, ssm.A_imag, ssm.C):
        assert torch.isfinite(p.grad).all()
    z = torch.zeros(3, dtype=torch.complex128, requires_grad=True)
    phi2(z).real.sum().backward()
    assert torch.isfinite(z.grad).all()


@pytest.mark.parametrize("rate", [44_100, 88_200, 96_000, 192_000])
def test_zoh_to_foh_is_the_two_coefficient_correction(rate: int) -> None:
    """Identity at 48 kHz, and equal to two_tap_kernel (the instrument of H6) at the
    other rates: the equivalence claimed on 2026-09-24, reproduced here."""
    zoh = layer("zoh")
    foh = copy.deepcopy(zoh)
    zoh_to_foh(foh)
    assert foh.discretization == "foh"
    x = torch.randn(2, CHANNELS, 3000, dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(foh(x), zoh(x), rtol=0, atol=1e-12)
        dt0 = torch.exp(zoh.log_dt.detach().clone())
        for ssm in (zoh, foh):
            ssm.log_dt += math.log(48_000 / rate)
        kernel = two_tap_kernel(zoh, dt0, x.shape[-1]).double()
        two_tap = torch.stack(
            [
                torch.nn.functional.conv1d(
                    torch.nn.functional.pad(x[:, ch : ch + 1], (x.shape[-1] - 1, 0)),
                    kernel[ch].flip(-1).view(1, 1, -1),
                )[:, 0]
                for ch in range(CHANNELS)
            ],
            dim=1,
        ) + x * zoh.D.detach().view(1, -1, 1)
        out = foh(x)
    error = (out - two_tap).abs().max() / out.abs().max()
    # two_tap_kernel computes in complex64; the identity is analytical.
    assert error < 1e-5, float(error)


@pytest.mark.parametrize("log_dt", [math.log(0.01), -15.0])
def test_zoh_to_foh_keeps_a_float32_model(log_dt: float) -> None:
    """The conversion must keep the response the float32 layer actually computes,
    whose ZOH weight (a - 1) / lambda loses digits to cancellation on slow modes
    (Codex review, 2026-09-24: ESR 6e-3 at log_dt = -15 when b was recomputed in
    float64)."""
    torch.manual_seed(7)
    zoh = DiagonalSSMLayer(CHANNELS, MODES, discretization="zoh").eval()
    with torch.no_grad():
        zoh.log_dt.fill_(log_dt)
        zoh.D.zero_()
    foh = copy.deepcopy(zoh)
    zoh_to_foh(foh)
    x = torch.randn(2, CHANNELS, 3000)
    with torch.no_grad():
        reference, out = zoh(x), foh(x)
    esr = float(((out - reference) ** 2).sum() / (reference**2).sum())
    assert esr < 1e-10, esr
