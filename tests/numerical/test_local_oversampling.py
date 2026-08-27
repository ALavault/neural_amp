import numpy as np
import torch

from fssr_nam.data.systems import apply_system
from fssr_nam.dsp.multirate import derive_reference_rates
from fssr_nam.metrics.nonlinear import (
    complex_harmonic_error,
    known_reference_parasite_db,
)
from fssr_nam.models import LocalOversampledSpline2x, S4Antialiased


def delayed(signal, samples):
    return torch.nn.functional.pad(signal, (samples, 0))[:-samples]


def set_tanh_spline(shaper):
    spline = shaper.spline
    with torch.no_grad():
        denominator = torch.tanh(torch.tensor(2.8))
        spline.values.copy_(torch.tanh(2.8 * spline.knots) / denominator)
        spline.slopes.copy_(
            2.8 * (1.0 - torch.tanh(2.8 * spline.knots).square()) / denominator
        )


def test_local_x2_identity_has_declared_delay_and_block_parity():
    torch.manual_seed(14)
    shaper = LocalOversampledSpline2x(filter_taps=33)
    signal = torch.rand(2049) - 0.5
    expected = shaper(signal)
    torch.testing.assert_close(expected, delayed(signal, 16), atol=3.0e-6, rtol=3.0e-6)
    shaper.reset_state()
    streamed = torch.cat(
        (
            shaper.stream(signal[:1]),
            shaper.stream(signal[1:71]),
            shaper.stream(signal[71:333]),
            shaper.stream(signal[333:]),
        )
    )
    torch.testing.assert_close(streamed, expected, atol=3.0e-6, rtol=3.0e-6)


def test_s4_aligns_all_paths_and_streams_equivalently():
    torch.manual_seed(15)
    model = S4Antialiased(decimation=16)
    signal = torch.rand(1025) - 0.5
    expected = model(signal)
    torch.testing.assert_close(expected, delayed(signal, 16), atol=4.0e-6, rtol=4.0e-6)
    model.reset_state()
    streamed = torch.cat(
        (
            model.stream(signal[:9]),
            model.stream(signal[9:700]),
            model.stream(signal[700:]),
        )
    )
    torch.testing.assert_close(streamed, expected, atol=4.0e-6, rtol=4.0e-6)
    assert model.latency_samples == 16


def test_local_x2_has_finite_tanh_output_and_gradients():
    shaper = LocalOversampledSpline2x(filter_taps=33)
    set_tanh_spline(shaper)
    time = torch.arange(4096) / 48_000
    signal = 0.48 * torch.sin(2 * torch.pi * 9_000 * time)
    output = shaper(signal)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in shaper.parameters()
    )


def test_local_x2_reduces_controlled_alias_without_losing_fundamental():
    master_rate = 192_000
    count = 192_000
    index = np.arange(count, dtype=np.float64)
    master_input = 0.48 * np.sin(2.0 * np.pi * 9_000.0 * index / master_rate)
    reference = derive_reference_rates(apply_system("tanh", master_input, master_rate))[
        48_000
    ].astype(np.float32)
    low_rate_input = derive_reference_rates(master_input)[48_000].astype(np.float32)
    naive = apply_system("tanh", low_rate_input, 48_000)
    shaper = LocalOversampledSpline2x(filter_taps=33)
    set_tanh_spline(shaper)
    antialiased = shaper(torch.from_numpy(low_rate_input)).detach().numpy()
    delay_samples = shaper.latency_samples
    delayed_reference = np.pad(reference, (delay_samples, 0))[:-delay_samples]
    delayed_naive = np.pad(naive, (delay_samples, 0))[:-delay_samples]
    stable = slice(512, None)
    naive_parasite = known_reference_parasite_db(
        delayed_naive[stable], delayed_reference[stable]
    )
    antialiased_parasite = known_reference_parasite_db(
        antialiased[stable], delayed_reference[stable]
    )
    harmonic_error = complex_harmonic_error(
        antialiased[stable],
        delayed_reference[stable],
        fundamental_hz=9_000.0,
        sample_rate=48_000,
    )
    assert antialiased_parasite < naive_parasite - 3.0
    assert harmonic_error < 1.0e-5
