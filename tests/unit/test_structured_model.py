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


def test_finite_fir_obeys_bibo_bound():
    torch.manual_seed(17)
    fir = CausalFIR(17)
    with torch.no_grad():
        fir.coefficients.copy_(torch.randn(17) * 0.1)
    signal = torch.rand(1024) * 2.0 - 1.0
    output = fir(signal)
    bound = signal.abs().max() * fir.coefficients.abs().sum()
    assert torch.isfinite(output).all()
    assert float(output.abs().max().detach()) <= float(bound.detach()) + 1.0e-6


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


def test_s0_relearns_identity_after_parameter_perturbation():
    torch.manual_seed(16)
    model = S0Structured(taps=7, num_knots=9)
    with torch.no_grad():
        model.drive.fill_(0.75)
        model.output_gain.fill_(1.15)
        model.offset.fill_(0.04)
    signal = torch.rand(512) - 0.5
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(150):
        optimizer.zero_grad()
        loss = (model(signal) - signal).square().mean()
        loss.backward()
        optimizer.step()
    final_error = (model(signal) - signal).square().mean()
    assert float(final_error.detach()) < 1.0e-6


def test_s0_overfits_one_short_nonlinear_segment():
    torch.manual_seed(18)
    model = S0Structured(taps=7, num_knots=17)
    signal = torch.rand(256) - 0.5
    target = torch.tanh(2.4 * signal) + 0.1 * signal.square()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    for _ in range(400):
        optimizer.zero_grad()
        loss = (model(signal) - target).square().mean()
        loss.backward()
        optimizer.step()
    final_error = (model(signal) - target).square().mean()
    assert float(final_error.detach()) < 2.0e-6
