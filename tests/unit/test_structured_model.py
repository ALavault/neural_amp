import torch

from fssr_nam.models import CausalFIR, S0Structured


def test_causal_fir_stream_matches_full_and_reset():
    torch.manual_seed(7)
    fir = CausalFIR(17)
    with torch.no_grad():
        fir.coefficients.copy_(torch.randn(17) * 0.1)
    signal = torch.randn(997)
    expected = fir(signal)
    parts = []
    position = 0
    sizes = [1, 7, 64, 3, 128, 17]
    while position < len(signal):
        size = sizes[len(parts) % len(sizes)]
        parts.append(fir.stream(signal[position : position + size]))
        position += size
    torch.testing.assert_close(torch.cat(parts), expected, atol=1.0e-6, rtol=1.0e-6)
    fir.reset_state()
    torch.testing.assert_close(fir.stream(signal), expected, atol=1.0e-6, rtol=1.0e-6)


def test_s0_starts_as_identity_and_streams_equivalently():
    torch.manual_seed(8)
    model = S0Structured(taps=17, num_knots=17)
    signal = 1.5 * torch.rand(1021) - 0.75
    expected = model(signal)
    torch.testing.assert_close(expected, signal, atol=3.0e-6, rtol=3.0e-6)
    parts = [
        model.stream(signal[:13]),
        model.stream(signal[13:700]),
        model.stream(signal[700:]),
    ]
    torch.testing.assert_close(torch.cat(parts), expected, atol=3.0e-6, rtol=3.0e-6)


def test_s0_is_causal_and_has_finite_gradients():
    torch.manual_seed(9)
    model = S0Structured()
    first = torch.randn(512)
    second = first.clone()
    second[300:] = torch.randn_like(second[300:])
    torch.testing.assert_close(model(first)[:300], model(second)[:300])
    loss = model(first).square().mean() + model.regularization()
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
