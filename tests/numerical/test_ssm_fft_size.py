"""DiagonalSSMLayer.fft_size = "fast": a 2-3-5-smooth FFT length instead of the next
power of 2 must still compute the same linear (not circular) causal convolution.
Float64; the recurrence (step) is the reference, independent of any FFT."""

from __future__ import annotations

import pytest
import torch

from fssr_nam.models.ssm_wavenet import DiagonalSSMLayer, fft_size

CHANNELS, MODES = 3, 4


@pytest.fixture
def fast():
    before = DiagonalSSMLayer.fft_size
    DiagonalSSMLayer.fft_size = "fast"
    yield
    DiagonalSSMLayer.fft_size = before


def layer(discretization: str) -> DiagonalSSMLayer:
    torch.manual_seed(11)
    ssm = DiagonalSSMLayer(CHANNELS, MODES, discretization=discretization).double()
    with torch.no_grad():
        ssm.log_A_real.normal_(-1.0, 0.5)
        ssm.D.normal_()
    return ssm.eval()


def test_fft_size() -> None:
    assert fft_size(287_999, "pow2") == 524_288
    assert fft_size(287_999, "fast") == 288_000
    assert fft_size(1_993, "fast") == 2_000
    assert fft_size(1, "fast") == 1
    for bad in ((0, "fast"), (5, "other")):
        with pytest.raises(ValueError):
            fft_size(*bad)


def test_fast_is_used(fast, monkeypatch) -> None:
    lengths = []
    rfft = torch.fft.rfft

    def recording(x, n=None, dim=-1):
        lengths.append(n)
        return rfft(x, n=n, dim=dim)

    monkeypatch.setattr(torch.fft, "rfft", recording)
    with torch.no_grad():
        layer("zoh")(torch.randn(1, CHANNELS, 1000, dtype=torch.float64))
    assert set(lengths) == {2_000}


def test_fast_gradients_match_pow2(fast) -> None:
    """Same gradients in float64 (in float32 they differ: demo/fft_size/NOTE.md)."""
    x = torch.randn(2, CHANNELS, 1000, dtype=torch.float64)
    grads = {}
    for kind in ("fast", "pow2"):
        DiagonalSSMLayer.fft_size = kind
        ssm = layer("zoh")
        ssm(x).pow(2).sum().backward()
        grads[kind] = [p.grad for p in ssm.parameters()]
    for a, b in zip(grads["fast"], grads["pow2"], strict=True):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-12)


# 2L-1 = 1993 and 1999: both pad to 2000 instead of 2048.
@pytest.mark.parametrize("length", [997, 1000])
def test_fast_matches_recurrence(fast, length: int) -> None:
    """FOH: its step() runs in float64, so the recurrence is a reference independent of
    any FFT.
    (ZOH's step() runs in complex64, 6e-8 off either FFT path: compared below.)"""
    ssm = layer("foh")
    x = torch.randn(2, CHANNELS, length, dtype=torch.float64)
    with torch.no_grad():
        full = ssm(x)
        ssm.reset_state()
        stepped = torch.stack([ssm.step(x[..., n]) for n in range(length)], dim=-1)
    assert torch.isfinite(full).all() and full.shape == x.shape
    torch.testing.assert_close(full, stepped, rtol=0, atol=1e-10)


@pytest.mark.parametrize("length", [997, 1000])
def test_fast_matches_pow2_zoh(fast, length: int) -> None:
    ssm = layer("zoh")
    x = torch.randn(2, CHANNELS, length, dtype=torch.float64)
    with torch.no_grad():
        fast_out = ssm(x)
        DiagonalSSMLayer.fft_size = "pow2"
        pow2_out = ssm(x)
    torch.testing.assert_close(fast_out, pow2_out, rtol=0, atol=1e-12)


def test_fast_matches_pow2_with_pole_shifts(fast) -> None:
    ssm = layer("zoh")
    x = torch.randn(2, CHANNELS, 1000, dtype=torch.float64)
    shift = (
        0.3 * torch.randn(2, CHANNELS, dtype=torch.float64),
        0.3 * torch.randn(2, CHANNELS, MODES, dtype=torch.float64),
    )
    with torch.no_grad():
        fast_out = ssm(x, shift)
        DiagonalSSMLayer.fft_size = "pow2"
        pow2_out = ssm(x, shift)
    torch.testing.assert_close(fast_out, pow2_out, rtol=0, atol=1e-12)


def test_fast_causal(fast) -> None:
    ssm = layer("foh")
    x = torch.randn(1, CHANNELS, 1000, dtype=torch.float64)
    changed = x.clone()
    changed[..., 500:] = torch.randn(1, CHANNELS, 500, dtype=torch.float64)
    with torch.no_grad():
        before, after = ssm(x), ssm(changed)
    torch.testing.assert_close(before[..., :500], after[..., :500], rtol=0, atol=1e-12)
