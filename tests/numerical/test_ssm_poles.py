"""Knobs that move the SSM poles (cond="poles"; demo/param/predictions_poles.md)."""

from __future__ import annotations

import math

import pytest
import torch

from fssr_nam.models.ssm_wavenet import (
    POLE_SCALE,
    DiagonalSSMLayer,
    SSMWaveNet,
    zoh_to_foh,
)

LENGTH = 1024
# The four arms of G2.1 at the Marshall size (8 blocks, 16 channels, 4 modes, 4 knobs),
# hidden widths chosen for equal parameter counts (predictions_poles.md).
ARMS = {
    ("film", "joint", (32,)): 28_385,
    ("film", "additive", (8, 8, 8, 8)): 28_289,
    ("poles", "joint", (25,)): 28_094,
    ("poles", "additive", (7, 7, 6, 6)): 28_661,
}


def model(cond: str = "poles", head: str = "joint", hidden=(8,), disc="zoh"):
    torch.manual_seed(0)
    net = SSMWaveNet(
        num_blocks=3,
        channels=4,
        state_dim=4,
        discretization=disc,
        num_controls=2,
        cond=cond,
        cond_head=head,
        cond_hidden=hidden,
    ).double()
    last = net.pole_net[2] if head == "joint" else net.pole_net.output
    torch.nn.init.normal_(last.weight, std=0.5)  # the head starts at zero
    return net.eval()


@pytest.mark.parametrize(("arm", "count"), ARMS.items())
def test_parameter_counts_match(arm, count) -> None:
    cond, head, hidden = arm
    net = SSMWaveNet(
        num_blocks=8,
        channels=16,
        state_dim=4,
        discretization="zoh",
        num_controls=4,
        cond=cond,
        cond_head=head,
        cond_hidden=hidden,
    )
    assert net.param_count() == count
    assert abs(count / 28_385 - 1) < 0.011


def test_every_arm_starts_from_the_same_trunk() -> None:
    """At one seed, the four G2 arms share every parameter outside their head and
    leave the generator in the same state: they differ by their conditioning only."""
    trunks, after = [], []
    for cond, head, hidden in ARMS:
        torch.manual_seed(42)
        net = SSMWaveNet(
            num_blocks=8,
            channels=16,
            state_dim=4,
            discretization="zoh",
            num_controls=4,
            cond=cond,
            cond_head=head,
            cond_hidden=hidden,
        )
        after.append(torch.rand(4))
        trunks.append(
            {
                k: v
                for k, v in net.state_dict().items()
                if not k.startswith(("film_net", "pole_net"))
            }
        )
    for trunk, draws in zip(trunks[1:], after[1:], strict=True):
        assert torch.equal(draws, after[0])
        assert trunk.keys() == trunks[0].keys()
        for name, value in trunk.items():
            assert torch.equal(value, trunks[0][name]), name


def test_zero_head_is_the_unconditioned_model() -> None:
    torch.manual_seed(0)
    plain = SSMWaveNet(num_blocks=3, channels=4, state_dim=4, discretization="zoh")
    torch.manual_seed(0)
    poles = SSMWaveNet(
        num_blocks=3,
        channels=4,
        state_dim=4,
        discretization="zoh",
        num_controls=2,
        cond="poles",
        cond_hidden=(8,),
    )
    poles.load_state_dict(plain.state_dict(), strict=False)
    x = torch.randn(2, 1, LENGTH)
    y = poles(x, torch.rand(2, 2))
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, plain(x), rtol=0, atol=1e-6)


def test_shifts_bounded_and_examples_independent() -> None:
    net = model()
    controls = torch.tensor([[0.1, 0.9], [0.8, 0.2]], dtype=torch.float64)
    for dlog_dt, dlog_a in net.pole_shifts(controls):
        assert dlog_dt.abs().max() <= POLE_SCALE
        assert dlog_a.abs().max() <= POLE_SCALE
    x = torch.randn(2, 1, LENGTH, dtype=torch.float64)
    other = controls.clone()
    other[1] = torch.tensor([0.0, 1.0])
    with torch.no_grad():
        y, y_other = net(x, controls), net(x, other)
    torch.testing.assert_close(y_other[0], y[0], rtol=0, atol=1e-12)
    assert (y_other[1] - y[1]).abs().max() > 1e-4


def test_additive_head_has_no_knob_interaction() -> None:
    net = model(head="additive", hidden=(4, 3))
    grid = torch.tensor(
        [[0.1, 0.2], [0.7, 0.2], [0.1, 0.9], [0.7, 0.9]], dtype=torch.float64
    )
    with torch.no_grad():
        out = net.pole_net(grid)
    torch.testing.assert_close(out[0] + out[3], out[1] + out[2], rtol=0, atol=1e-12)
    joint = model(head="joint", hidden=(8,))
    with torch.no_grad():
        out = joint.pole_net(grid)
    assert (out[0] + out[3] - out[1] - out[2]).abs().max() > 1e-6


@pytest.mark.parametrize("disc", ["zoh", "foh"])
def test_fft_matches_step_with_shifts(disc: str) -> None:
    net = model(disc=disc)
    layer = net.blocks[1].ssm
    shift = net.pole_shifts(torch.tensor([[0.2, 0.6], [0.9, 0.1]]).double())[1]
    x = torch.randn(2, layer.channels, 300, dtype=torch.float64)
    with torch.no_grad():
        full = layer(x, shift)
        layer.reset_state()
        stepped = torch.stack(
            [layer.step(x[..., n], shift) for n in range(x.shape[-1])], dim=-1
        )
    torch.testing.assert_close(stepped, full, rtol=0, atol=1e-10)


def test_rate_rescaling_stays_exact_with_shifts() -> None:
    """ZOH is exact for a held input: with the knob shifts on, every layer still
    reproduces its 48 kHz step response at 96 kHz (output at t = (n + 1) dt)."""
    net = model()
    shifts = net.pole_shifts(torch.tensor([[0.3, 0.8]], dtype=torch.float64))
    with torch.no_grad():
        for block, shift in zip(net.blocks, shifts, strict=True):
            layer = block.ssm
            step48 = layer(torch.ones(1, 4, LENGTH, dtype=torch.float64), shift)
            layer.log_dt += math.log(48_000 / 96_000)
            step96 = layer(torch.ones(1, 4, 2 * LENGTH, dtype=torch.float64), shift)
            torch.testing.assert_close(step96[..., 1::2], step48, rtol=0, atol=1e-10)


def test_checkpointing_leaves_gradients_unchanged() -> None:
    x = torch.randn(2, 1, LENGTH, dtype=torch.float64)
    controls = torch.tensor([[0.2, 0.6], [0.9, 0.1]], dtype=torch.float64)
    grads = []
    for checkpoint in (True, False):
        net = model().train()
        for block in net.blocks:
            block.checkpoint_kernels = checkpoint
        net(x, controls).square().sum().backward()
        # The last block's residual output feeds nothing: its res_conv has no grad.
        grads.append(
            {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
        )
    assert grads[0].keys() == grads[1].keys()
    for name, grad in grads[0].items():
        torch.testing.assert_close(grad, grads[1][name], rtol=0, atol=1e-12)


def test_free_discretization_refuses_pole_shifts() -> None:
    layer = DiagonalSSMLayer(2, 2, discretization="free")
    shift = (torch.zeros(1, 2), torch.zeros(1, 2, 2))
    with pytest.raises(ValueError):
        layer(torch.zeros(1, 2, 8), shift)


def test_zoh_to_foh_refuses_knob_driven_poles() -> None:
    """Fixed FOH coefficients cannot follow knob-dependent poles (Codex review,
    2026-09-24: ESR 0.053 after conversion with nonzero shifts)."""
    net = model()
    with pytest.raises(ValueError):
        zoh_to_foh(net)
    assert all(block.ssm.discretization == "zoh" for block in net.blocks)
