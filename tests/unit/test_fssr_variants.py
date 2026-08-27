import torch

from fssr_nam.models import S1Slow, S2Residual, S3FastSlowResidual, SlowStateController


def irregular_stream(model, signal):
    parts = []
    position = 0
    sizes = (1, 7, 64, 3, 128, 17)
    while position < len(signal):
        size = sizes[len(parts) % len(sizes)]
        parts.append(model.stream(signal[position : position + size]))
        position += size
    return torch.cat(parts)


def test_slow_controller_is_causal_and_block_equivalent():
    torch.manual_seed(10)
    controller = SlowStateController(hidden_size=8, decimation=16)
    with torch.no_grad():
        controller.gru.weight_ih.fill_(0.05)
        controller.gru.weight_hh.fill_(0.02)
        controller.gru.bias_ih.fill_(0.01)
        controller.projection.weight.fill_(0.04)
    signal = torch.cat((torch.zeros(64), torch.full((128,), 0.8)))
    expected = controller(signal)
    streamed = torch.cat(
        (
            controller.stream(signal[:11]),
            controller.stream(signal[11:100]),
            controller.stream(signal[100:]),
        ),
        dim=-1,
    )
    torch.testing.assert_close(streamed, expected, atol=1.0e-7, rtol=1.0e-7)
    assert torch.all(expected[:, :16] == torch.tensor([[1.0], [0.0], [1.0]]))
    assert not torch.equal(expected[:, 96], expected[:, 16])


def test_s1_starts_as_identity_and_streams_equivalently():
    torch.manual_seed(11)
    model = S1Slow(decimation=16)
    signal = 1.2 * torch.rand(513) - 0.6
    expected = model(signal)
    torch.testing.assert_close(expected, signal, atol=3.0e-6, rtol=3.0e-6)
    torch.testing.assert_close(
        irregular_stream(model, signal), expected, atol=3.0e-6, rtol=3.0e-6
    )


def test_s2_has_short_zero_initialized_residual_and_block_parity():
    torch.manual_seed(12)
    model = S2Residual()
    signal = torch.randn(769) * 0.2
    output, core, residual = model.forward_components(signal)
    torch.testing.assert_close(output, core)
    torch.testing.assert_close(residual, torch.zeros_like(residual))
    assert model.residual.receptive_field == 31
    assert model.residual.receptive_field < 240
    assert float(model.energy_ratio(signal).detach()) == 0.0
    torch.testing.assert_close(
        irregular_stream(model, signal), output, atol=3.0e-6, rtol=3.0e-6
    )


def test_s3_starts_as_identity_has_finite_gradients_and_block_parity():
    torch.manual_seed(13)
    model = S3FastSlowResidual(decimation=16)
    signal = (torch.rand(384) - 0.5).requires_grad_()
    expected = model(signal)
    torch.testing.assert_close(expected, signal, atol=3.0e-6, rtol=3.0e-6)
    loss = expected.square().mean()
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    model.zero_grad(set_to_none=True)
    model.reset_state()
    torch.testing.assert_close(
        irregular_stream(model, signal.detach()),
        expected.detach(),
        atol=3.0e-6,
        rtol=3.0e-6,
    )
