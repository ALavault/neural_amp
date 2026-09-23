"""SSM-WaveNet with knob conditioning: finite, causal, controllable, rate-rescalable."""

from __future__ import annotations

import math

import torch

from fssr_nam.models.ssm_wavenet import DiagonalSSMLayer, SSMWaveNet

LENGTH = 2048


def conditioned_model() -> SSMWaveNet:
    torch.manual_seed(0)
    model = SSMWaveNet(
        num_blocks=3, channels=4, state_dim=4, discretization="zoh", num_controls=2
    ).double()
    # The FiLM head starts at zero; give it weights so the knobs act.
    torch.nn.init.normal_(model.film_net[2].weight, std=0.5)
    return model.eval()


def test_finite_shape_and_zero_init_matches_unconditioned() -> None:
    torch.manual_seed(0)
    plain = SSMWaveNet(num_blocks=3, channels=4, state_dim=4, discretization="zoh")
    torch.manual_seed(0)
    film = SSMWaveNet(
        num_blocks=3, channels=4, state_dim=4, discretization="zoh", num_controls=2
    )
    film.load_state_dict(plain.state_dict(), strict=False)
    x = torch.randn(2, 1, LENGTH)
    y = film(x, torch.rand(2, 2))
    assert y.shape == (2, 1, LENGTH)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, plain(x), rtol=0, atol=1e-6)


def test_causal() -> None:
    model = conditioned_model()
    x = torch.randn(1, 1, LENGTH, dtype=torch.float64)
    controls = torch.tensor([[0.2, 0.7]], dtype=torch.float64)
    changed = x.clone()
    changed[..., 1000:] = torch.randn(1, 1, LENGTH - 1000, dtype=torch.float64)
    with torch.no_grad():
        before, after = model(x, controls), model(changed, controls)
    torch.testing.assert_close(before[..., :1000], after[..., :1000], rtol=0, atol=1e-9)


def test_knobs_change_the_output() -> None:
    model = conditioned_model()
    x = torch.randn(1, 1, LENGTH, dtype=torch.float64)
    with torch.no_grad():
        low = model(x, torch.tensor([[0.0, 0.0]], dtype=torch.float64))
        high = model(x, torch.tensor([[1.0, 1.0]], dtype=torch.float64))
    assert (low - high).abs().max() > 1e-3


def test_delta_rescaling_stays_exact_with_knobs() -> None:
    """Rescaling dt touches only the SSMs: a zero shift is bitwise neutral for the
    conditioned model, and every conditioned model's SSM layer reproduces its 48 kHz
    step response at 96 kHz (ZOH output at t = (n + 1) dt, hence indices 2n + 1)."""
    model = conditioned_model()
    x = torch.randn(1, 1, LENGTH, dtype=torch.float64)
    controls = torch.tensor([[0.3, 0.9]], dtype=torch.float64)
    layers = [m for m in model.modules() if isinstance(m, DiagonalSSMLayer)]
    base = [layer.log_dt.data.clone() for layer in layers]
    with torch.no_grad():
        reference = model(x, controls)
        for layer, log_dt in zip(layers, base, strict=True):
            layer.log_dt.data = log_dt + math.log(48_000 / 48_000)
        assert torch.equal(model(x, controls), reference)
        for layer, log_dt in zip(layers, base, strict=True):
            step48 = layer(torch.ones(1, layer.channels, LENGTH, dtype=torch.float64))
            layer.log_dt.data = log_dt + math.log(48_000 / 96_000)
            step96 = layer(
                torch.ones(1, layer.channels, 2 * LENGTH, dtype=torch.float64)
            )
            torch.testing.assert_close(step96[..., 1::2], step48, rtol=0, atol=1e-10)
