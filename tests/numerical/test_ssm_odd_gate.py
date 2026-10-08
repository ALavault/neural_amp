"""SSM-WaveNet's odd gate (act_type "odd"): finite, odd while the explicit asymmetry is
zero (zero input gives zero output), asymmetric otherwise, causal, the same output
whole and sample by sample, reset, and the same random draws as the gated model."""

from __future__ import annotations

import pytest
import torch
import torch.utils.checkpoint

from fssr_nam.models.ssm_wavenet import GatedSSMBlock, SSMWaveNet

LENGTH = 1024
CHANNELS = 4


def randomise_gate(block: GatedSSMBlock, generator: torch.Generator) -> None:
    """Give the odd gate's constant-initialised parameters generic values, the
    asymmetry excepted (it stays zero)."""
    gate = block.odd_gate
    with torch.no_grad():
        for p in (gate.weight, gate.bias):
            p.copy_(torch.randn(p.shape, generator=generator, dtype=p.dtype))
        gate.alpha_logit.add_(
            torch.randn(gate.alpha_logit.shape, generator=generator, dtype=p.dtype)
        )


def odd_model() -> SSMWaveNet:
    """Conditioned odd model, knobs acting on scale and shift_b but not on beta."""
    torch.manual_seed(0)
    model = SSMWaveNet(
        num_blocks=3,
        channels=CHANNELS,
        state_dim=4,
        discretization="zoh",
        num_controls=2,
        act_type="odd",
        head_act="tanh",
    ).double()
    generator = torch.Generator().manual_seed(1)
    last = model.film_net[2]
    with torch.no_grad():
        last.weight.copy_(0.5 * torch.randn(last.weight.shape, generator=generator))
        last.bias.copy_(0.5 * torch.randn(last.bias.shape, generator=generator))
        for i in range(3):  # beta slots: [2C, 3C) of each block's 4C chunk
            rows = slice(
                i * 4 * CHANNELS + 2 * CHANNELS, i * 4 * CHANNELS + 3 * CHANNELS
            )
            last.weight[rows] = 0
            last.bias[rows] = 0
    for block in model.blocks:
        randomise_gate(block, generator)
    return model.eval()


def odd_block() -> tuple[GatedSSMBlock, torch.Tensor]:
    """A single odd block and a FiLM (scale random, beta zero, shift_b random)."""
    torch.manual_seed(0)
    block = GatedSSMBlock(CHANNELS, 4, act_type="odd", discretization="zoh").double()
    generator = torch.Generator().manual_seed(2)
    randomise_gate(block, generator)
    scale = 0.5 * torch.randn(1, 2 * CHANNELS, generator=generator, dtype=torch.float64)
    shift_b = torch.randn(1, CHANNELS, generator=generator, dtype=torch.float64)
    film = torch.cat([scale, torch.zeros(1, CHANNELS, dtype=torch.float64), shift_b], 1)
    return block.eval(), film


def test_finite_and_shape() -> None:
    model = odd_model()
    x = torch.randn(2, 1, LENGTH, dtype=torch.float64)
    with torch.no_grad():
        y = model(x, torch.rand(2, 2, dtype=torch.float64))
    assert y.shape == (2, 1, LENGTH)
    assert torch.isfinite(y).all()


def test_block_is_odd_and_zero_maps_to_zero() -> None:
    block, film = odd_block()
    x = torch.randn(1, CHANNELS, LENGTH, dtype=torch.float64)
    with torch.no_grad():
        res_plus, skip_plus = block(x, film)
        res_minus, skip_minus = block(-x, film)
        res_zero, skip_zero = block(torch.zeros_like(x), film)
    torch.testing.assert_close(res_minus, -res_plus, rtol=0, atol=1e-12)
    torch.testing.assert_close(skip_minus, -skip_plus, rtol=0, atol=1e-12)
    assert torch.equal(res_zero, torch.zeros_like(x))
    assert torch.equal(skip_zero, torch.zeros_like(x))
    # The gate is not trivially linear: doubling the input does not double the output.
    with torch.no_grad():
        _, skip_double = block(2 * x, film)
    assert (skip_double - 2 * skip_plus).abs().max() > 1e-3


def test_network_is_odd_and_zero_maps_to_zero() -> None:
    model = odd_model()
    x = torch.randn(2, 1, LENGTH, dtype=torch.float64)
    controls = torch.tensor([[0.2, 0.7], [0.9, 0.1]], dtype=torch.float64)
    with torch.no_grad():
        plus, minus = model(x, controls), model(-x, controls)
        zero = model(torch.zeros_like(x), controls)
    torch.testing.assert_close(minus, -plus, rtol=0, atol=1e-12)
    assert torch.equal(zero, torch.zeros_like(x))


def test_asymmetry_breaks_oddness() -> None:
    block, film = odd_block()
    x = torch.randn(1, CHANNELS, LENGTH, dtype=torch.float64)
    with torch.no_grad():
        block.odd_gate.asymmetry.fill_(0.3)
        _, plus = block(x, film)
        _, minus = block(-x, film)
    even = (plus + minus) / 2
    assert even.abs().max() > 1e-2 * plus.abs().max()


def test_causal() -> None:
    model = odd_model()
    x = torch.randn(1, 1, LENGTH, dtype=torch.float64)
    controls = torch.tensor([[0.4, 0.6]], dtype=torch.float64)
    changed = x.clone()
    changed[..., 600:] = torch.randn(1, 1, LENGTH - 600, dtype=torch.float64)
    with torch.no_grad():
        before, after = model(x, controls), model(changed, controls)
    torch.testing.assert_close(before[..., :600], after[..., :600], rtol=0, atol=1e-12)


def stepped(block: GatedSSMBlock, x: torch.Tensor, film: torch.Tensor) -> tuple:
    outputs = [block.step(x[..., n], film) for n in range(x.shape[-1])]
    return tuple(torch.stack(o, dim=-1) for o in zip(*outputs, strict=True))


def test_gate_whole_matches_step_and_reset() -> None:
    """The odd gate alone, all in float64: whole and sample by sample agree, and a
    reset forgets the envelope."""
    block, _ = odd_block()
    gate = block.odd_gate
    generator = torch.Generator().manual_seed(3)
    a, b = (
        torch.randn(2, CHANNELS, 300, generator=generator, dtype=torch.float64)
        for _ in range(2)
    )
    beta = 0.2 * torch.randn(2, CHANNELS, generator=generator, dtype=torch.float64)
    shift = torch.randn(2, CHANNELS, generator=generator, dtype=torch.float64)
    with torch.no_grad():
        whole = gate(a, b, beta[..., None], shift[..., None])
        gate.reset_state()
        steps = torch.stack(
            [gate.step(a[..., n], b[..., n], beta, shift) for n in range(300)], dim=-1
        )
        gate.reset_state()
        again = torch.stack(
            [gate.step(a[..., n], b[..., n], beta, shift) for n in range(100, 250)],
            dim=-1,
        )
        fresh = gate(
            a[..., 100:250], b[..., 100:250], beta[..., None], shift[..., None]
        )
    torch.testing.assert_close(steps, whole, rtol=0, atol=1e-12)
    torch.testing.assert_close(again, fresh, rtol=0, atol=1e-12)


def test_block_whole_matches_step_and_reset() -> None:
    """The whole odd block: the SSM's step path computes in complex64 (DiagonalSSMLayer
    .step), hence a float32-level tolerance; the gate itself is checked above to
    1e-12."""
    block, film = odd_block()
    x = torch.randn(2, CHANNELS, 300, dtype=torch.float64)
    film = film.expand(2, -1)
    with torch.no_grad():
        whole = block(x, film)
        block.reset_state()
        steps = stepped(block, x, film)
        # A reset forgets the SSM state and the envelope.
        block.reset_state()
        again = stepped(block, x[..., 100:250], film)
        fresh = block(x[..., 100:250], film)
    for w, s in zip(whole, steps, strict=True):
        torch.testing.assert_close(s, w, rtol=0, atol=1e-6)
    for a, f in zip(again, fresh, strict=True):
        torch.testing.assert_close(a, f, rtol=0, atol=1e-6)


def test_same_draws_as_the_gated_model() -> None:
    """Biases are removed after the layers are built: every weight the two models
    share is drawn identically, so seed-paired arms start from the same trunk."""
    kwargs = {
        "num_blocks": 3,
        "channels": CHANNELS,
        "state_dim": 4,
        "discretization": "zoh",
        "num_controls": 2,
        "head_act": "tanh",
    }
    torch.manual_seed(0)
    gated = SSMWaveNet(**kwargs).state_dict()
    torch.manual_seed(0)
    odd = SSMWaveNet(**kwargs, act_type="odd").state_dict()
    shared = [k for k in odd if k in gated]
    assert len(shared) > 20
    for k in shared:
        assert torch.equal(odd[k], gated[k]), k
    assert not [
        k
        for k in odd
        if k.endswith("bias") and "film_net" not in k and "odd_gate" not in k
    ]


def test_recomputed_gate_gives_the_same_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The odd gate is recomputed in the backward pass (activation memory): every
    parameter gradient equals that of a plain backward pass, bit for bit (CPU,
    float64)."""
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(2, 1, LENGTH, generator=generator, dtype=torch.float64)
    controls = torch.tensor([[0.2, 0.7], [0.9, 0.1]], dtype=torch.float64)

    def gradients() -> dict[str, torch.Tensor]:
        model = odd_model()
        model(x, controls).square().mean().backward()
        return {k: p.grad for k, p in model.named_parameters() if p.grad is not None}

    calls: list[int] = []
    checkpoint = torch.utils.checkpoint.checkpoint

    def counted(*args, **kwargs):
        calls.append(1)
        return checkpoint(*args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", counted)
    recomputed = gradients()
    assert len(calls) == 3  # one per block
    monkeypatch.setattr(
        torch.utils.checkpoint, "checkpoint", lambda f, *args, **kwargs: f(*args)
    )
    plain = gradients()
    assert plain.keys() == recomputed.keys() and len(plain) > 20
    for k in plain:
        assert torch.equal(recomputed[k], plain[k]), k


def test_saturated_envelope_stays_finite() -> None:
    """Envelope logits where float32 rounds alpha to 1: still finite."""
    torch.manual_seed(0)
    model = SSMWaveNet(
        num_blocks=2,
        channels=CHANNELS,
        state_dim=4,
        discretization="zoh",
        num_controls=2,
        act_type="odd",
        head_act="tanh",
    )
    with torch.no_grad():
        for block in model.blocks:
            block.odd_gate.alpha_logit.fill_(30.0)
            block.odd_gate.weight.fill_(1.0)
    y = model(torch.randn(2, 1, 512), torch.rand(2, 2))
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
