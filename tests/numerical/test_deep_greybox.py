"""Deeper grey-box amp model: holds GreyBoxAmp exactly while its additions are neutral;
finite with finite gradients; zero input gives zero output even with a bias; odd while
asymmetries and biases are zero, asymmetric with a bias; causal; the same output whole
and sample by sample, and reset."""

from __future__ import annotations

import pytest
import torch

from fssr_nam.models.deep_greybox import DeepGreyBoxAmp
from fssr_nam.models.greybox import GreyBoxAmp

LENGTH = 2048
CONTROLS = torch.tensor([[0.2, 0.7], [0.9, 0.1]], dtype=torch.float64)
FULL = {"stages": 3, "taps": 48, "bias": True, "slow_s": (0.002, 0.01)}


def model(bias: float = 0.0) -> DeepGreyBoxAmp:
    """Every addition on, a non-zero head so the knobs act, every stage mixed in; the
    asymmetries at zero, the static biases at `bias`, the slow bias weights at zero."""
    torch.manual_seed(0)
    amp = DeepGreyBoxAmp(num_controls=2, **FULL).double()
    generator = torch.Generator().manual_seed(1)
    last = amp.head[2]
    with torch.no_grad():
        last.weight.copy_(0.3 * torch.randn(last.weight.shape, generator=generator))
        last.bias.copy_(0.3 * torch.randn(last.bias.shape, generator=generator))
        rows = last.weight.view(amp.stages, len(amp.names), amp.branches, -1)
        offsets = last.bias.view(amp.stages, len(amp.names), amp.branches)
        for i, name in enumerate(amp.names):
            if name.startswith(("beta", "bias", "slow_bias")):
                rows[:, i] = 0
                offsets[:, i] = 0
        for s in range(amp.stages):
            amp.base[s]["compression"].fill_(1.0)
            amp.base[s]["bias"].fill_(bias)
            for k in range(amp.slow):
                amp.base[s][f"slow_gain_{k}"].fill_(0.5)
            amp.contract[s].weight.fill_(0.5)
    return amp.eval()


def signal(seed: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return 0.5 * torch.randn(2, 1, LENGTH, generator=generator, dtype=torch.float64)


@pytest.mark.parametrize(
    "kwargs", [{}, FULL, {"taps": 64}, {"bias": True, "slow_s": (0.05,)}]
)
def test_holds_the_grey_box_exactly(kwargs: dict) -> None:
    torch.manual_seed(0)
    grey = GreyBoxAmp(num_controls=2).double()
    with torch.no_grad():
        for p in grey.parameters():
            p.add_(0.1 * torch.randn_like(p))
    x = signal()[:1]
    controls = CONTROLS[0]
    with torch.no_grad():
        expected = grey(x, controls[None])
        got = DeepGreyBoxAmp.from_greybox(grey, controls, **kwargs)(x)
    torch.testing.assert_close(got, expected, rtol=0, atol=1e-12)


def test_finite_shape_and_gradients() -> None:
    amp = model(bias=0.3)
    x = signal()
    y = amp(x, CONTROLS)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    for name, p in amp.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_zero_maps_to_zero_with_a_bias() -> None:
    amp = model(bias=0.3)
    with torch.no_grad():
        zero = amp(torch.zeros(2, 1, LENGTH, dtype=torch.float64), CONTROLS)
    assert torch.equal(zero, torch.zeros_like(zero))


def test_odd_without_bias_asymmetric_with_it() -> None:
    x = signal()
    with torch.no_grad():
        amp = model()
        plus, minus = amp(x, CONTROLS), amp(-x, CONTROLS)
        torch.testing.assert_close(minus, -plus, rtol=0, atol=1e-12)
        amp = model(bias=0.3)
        plus, minus = amp(x, CONTROLS), amp(-x, CONTROLS)
    assert ((plus + minus) / 2).abs().max() > 1e-2 * plus.abs().max()


def test_causal() -> None:
    amp = model(bias=0.3)
    x = signal()
    changed = x.clone()
    changed[..., 1000:] = signal(3)[..., 1000:]
    with torch.no_grad():
        before, after = amp(x, CONTROLS), amp(changed, CONTROLS)
    torch.testing.assert_close(
        before[..., :1000], after[..., :1000], rtol=0, atol=1e-12
    )


def test_whole_matches_step_and_reset() -> None:
    amp = model(bias=0.3)
    generator = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for fir in (amp.pre_fir, amp.post_fir):
            fir.copy_(torch.randn(fir.shape, generator=generator, dtype=fir.dtype))
    x = signal()[..., :400]
    with torch.no_grad():
        whole = amp(x, CONTROLS)
        amp.reset_states()
        steps = torch.stack([amp.step(x[..., n], CONTROLS) for n in range(400)], -1)
        amp.reset_states()
        again = torch.stack(
            [amp.step(x[..., n], CONTROLS) for n in range(150, 350)], -1
        )
        fresh = amp(x[..., 150:350], CONTROLS)
    # Three stages of FFT convolutions against recursions: rounding reaches ~1e-12.
    torch.testing.assert_close(steps, whole, rtol=0, atol=1e-10)
    torch.testing.assert_close(again, fresh, rtol=0, atol=1e-10)
