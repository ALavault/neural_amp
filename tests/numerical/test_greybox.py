"""Grey-box amp model: finite with finite gradients, odd while its asymmetries are zero
(zero input gives zero output), asymmetric otherwise, causal, knob-controlled, the
same output whole and sample by sample, and reset."""

from __future__ import annotations

import torch

from fssr_nam.models.greybox import GreyBoxAmp

LENGTH = 2048


def model(beta: float = 0.0) -> GreyBoxAmp:
    """Conditioned grey box with a non-zero head, so the knobs act; beta sets both
    sign-dependent gains of every branch."""
    torch.manual_seed(0)
    amp = GreyBoxAmp(num_controls=2).double()
    generator = torch.Generator().manual_seed(1)
    last = amp.head[2]
    with torch.no_grad():
        last.weight.copy_(0.3 * torch.randn(last.weight.shape, generator=generator))
        last.bias.copy_(0.3 * torch.randn(last.bias.shape, generator=generator))
        # Head rows of the asymmetries (beta_in, beta_out) back to zero.
        for k in (5, 6):
            rows = slice(k * amp.branches, (k + 1) * amp.branches)
            last.weight[rows] = 0
            last.bias[rows] = 0
        amp.base["beta_in"].fill_(beta)
        amp.base["beta_out"].fill_(beta)
        amp.base["compression"].fill_(1.0)
    return amp.eval()


def signal(seed: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return 0.5 * torch.randn(2, 1, LENGTH, generator=generator, dtype=torch.float64)


CONTROLS = torch.tensor([[0.2, 0.7], [0.9, 0.1]], dtype=torch.float64)


def test_finite_shape_and_gradients() -> None:
    amp = model()
    x = signal()
    y = amp(x, CONTROLS)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    for name, p in amp.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_odd_and_zero_maps_to_zero() -> None:
    amp = model()
    x = signal()
    with torch.no_grad():
        plus, minus = amp(x, CONTROLS), amp(-x, CONTROLS)
        zero = amp(torch.zeros_like(x), CONTROLS)
    torch.testing.assert_close(minus, -plus, rtol=0, atol=1e-12)
    assert torch.equal(zero, torch.zeros_like(x))
    # Not trivially linear.
    with torch.no_grad():
        double = amp(2 * x, CONTROLS)
    assert (double - 2 * plus).abs().max() > 1e-3


def test_asymmetry_breaks_oddness() -> None:
    amp = model(beta=0.3)
    x = signal()
    with torch.no_grad():
        plus, minus = amp(x, CONTROLS), amp(-x, CONTROLS)
    assert ((plus + minus) / 2).abs().max() > 1e-2 * plus.abs().max()


def test_causal() -> None:
    amp = model()
    x = signal()
    changed = x.clone()
    changed[..., 1000:] = signal(3)[..., 1000:]
    with torch.no_grad():
        before, after = amp(x, CONTROLS), amp(changed, CONTROLS)
    torch.testing.assert_close(
        before[..., :1000], after[..., :1000], rtol=0, atol=1e-12
    )


def test_knobs_change_the_output() -> None:
    amp = model()
    x = signal()[:1].expand(2, -1, -1)
    with torch.no_grad():
        y = amp(x, CONTROLS)
    assert (y[0] - y[1]).abs().max() > 1e-3


def test_whole_matches_step_and_reset() -> None:
    amp = model(beta=0.2)
    x = signal()[..., :400]
    with torch.no_grad():
        whole = amp(x, CONTROLS)
        amp.reset_states()
        steps = torch.stack([amp.step(x[..., n], CONTROLS) for n in range(400)], -1)
        # A reset forgets every filter, the envelope and the FIR histories.
        amp.reset_states()
        again = torch.stack(
            [amp.step(x[..., n], CONTROLS) for n in range(150, 350)], -1
        )
        fresh = amp(x[..., 150:350], CONTROLS)
    torch.testing.assert_close(steps, whole, rtol=0, atol=1e-12)
    torch.testing.assert_close(again, fresh, rtol=0, atol=1e-12)


def test_random_fir_on_a_buffer_shorter_than_the_taps() -> None:
    """Non-identity FIRs on 16 samples (32 taps): causal, and whole equals step, so
    nothing wraps around and the FIR histories carry and reset."""
    amp = model(beta=0.2)
    generator = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for fir in (amp.pre_fir, amp.post_fir):
            fir.copy_(torch.randn(fir.shape, generator=generator, dtype=fir.dtype))
    impulse = torch.zeros(2, 1, 16, dtype=torch.float64)
    impulse[..., 1] = 1
    x = signal()[..., :16]
    with torch.no_grad():
        response = amp(impulse, CONTROLS)
        whole = amp(x, CONTROLS)
        amp.reset_states()
        steps = torch.stack([amp.step(x[..., n], CONTROLS) for n in range(16)], -1)
    # FFT rounding leaves ~1e-16 before the impulse; a wrap-around would give O(1).
    assert response[..., :1].abs().max() < 1e-12
    torch.testing.assert_close(steps, whole, rtol=0, atol=1e-12)


def test_reset_clears_the_fir_histories() -> None:
    """Random FIRs (32 taps) and 100 samples streamed before a reset: what follows
    equals a fresh run, while without the reset the first signal still shows."""
    amp = model(beta=0.2)
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for fir in (amp.pre_fir, amp.post_fir):
            fir.copy_(torch.randn(fir.shape, generator=generator, dtype=fir.dtype))
    first, second = signal(6)[..., :100], signal(7)[..., :100]

    def stream(x: torch.Tensor) -> torch.Tensor:
        return torch.stack([amp.step(x[..., n], CONTROLS) for n in range(100)], -1)

    with torch.no_grad():
        fresh = amp(second, CONTROLS)
        amp.reset_states()
        stream(first)
        kept = stream(second)
        amp.reset_states()
        stream(first)
        amp.reset_states()
        after = stream(second)
    torch.testing.assert_close(after, fresh, rtol=0, atol=1e-12)
    assert (kept - fresh).abs().max() > 1e-6


def test_saturated_envelope_stays_finite() -> None:
    """Envelope logits where float32 rounds alpha to 1: still finite."""
    torch.manual_seed(0)
    amp = GreyBoxAmp(num_controls=2)
    with torch.no_grad():
        amp.base["alpha"].fill_(30.0)
        amp.base["compression"].fill_(1.0)
    x = torch.randn(2, 1, 512)
    y = amp(x, torch.rand(2, 2))
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in amp.parameters())
