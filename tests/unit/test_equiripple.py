import numpy as np
import pytest
import torch
from scipy.signal import freqz

from fssr_nam.models.equiripple import (
    PolyphaseHalfbandDecimator2x,
    PolyphaseHalfbandInterpolator2x,
    SparseHalfbandFIR,
    design_equiripple_halfband,
)


def test_halfband_design_is_symmetric_sparse_and_spectrally_valid() -> None:
    coefficients = design_equiripple_halfband()
    torch.testing.assert_close(coefficients, coefficients.flip(0))
    assert int(torch.count_nonzero(coefficients)) == 25
    assert float(coefficients[len(coefficients) // 2]) == 0.5
    frequency, response = freqz(coefficients.numpy(), worN=65_536, fs=2.0)
    passband = np.abs(response[frequency <= 0.45])
    stopband = np.abs(response[frequency >= 0.55])
    assert float(np.max(np.abs(passband - 1.0))) < 0.006
    assert float(np.max(stopband)) < 0.006


def test_sparse_halfband_has_reset_and_block_parity() -> None:
    torch.manual_seed(41)
    signal = torch.randn(2, 509)
    fir = SparseHalfbandFIR(design_equiripple_halfband())
    expected = fir(signal)
    actual = torch.cat(
        (
            fir.stream(signal[:, :17]),
            fir.stream(signal[:, 17:201]),
            fir.stream(signal[:, 201:]),
        ),
        dim=-1,
    )
    torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=2.0e-6)
    assert fir.multiply_count_per_sample == 25
    fir.reset_state()
    torch.testing.assert_close(fir.stream(signal), expected)


def test_polyphase_interpolator_matches_dense_zero_insertion() -> None:
    torch.manual_seed(42)
    coefficients = design_equiripple_halfband()
    signal = torch.randn(2, 257)
    zero_inserted = signal.new_zeros((len(signal), 2 * signal.shape[-1]))
    zero_inserted[:, ::2] = signal
    expected = SparseHalfbandFIR(2.0 * coefficients)(zero_inserted)

    interpolator = PolyphaseHalfbandInterpolator2x(coefficients)
    actual = interpolator(signal)
    torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=2.0e-6)
    assert interpolator.active_multiply_count_per_input_sample == 25
    assert interpolator.active_multiply_count_per_input_sample < 2 * 25

    streamed = torch.cat(
        (
            interpolator.stream(signal[:, :17]),
            interpolator.stream(signal[:, 17:201]),
            interpolator.stream(signal[:, 201:]),
        ),
        dim=-1,
    )
    assert float((streamed - expected).abs().max()) <= 2.0e-5
    interpolator.reset_state()
    torch.testing.assert_close(interpolator.stream(signal), expected)


def test_polyphase_decimator_matches_dense_fir_across_odd_blocks() -> None:
    torch.manual_seed(43)
    coefficients = design_equiripple_halfband()
    signal = torch.randn(2, 509)
    expected = SparseHalfbandFIR(coefficients)(signal)[:, ::2]

    decimator = PolyphaseHalfbandDecimator2x(coefficients)
    actual = decimator(signal)
    torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=2.0e-6)
    assert decimator.active_multiply_count_per_output_sample == 25

    streamed = torch.cat(
        (
            decimator.stream(signal[:, :17]),
            decimator.stream(signal[:, 17:200]),
            decimator.stream(signal[:, 200:401]),
            decimator.stream(signal[:, 401:]),
        ),
        dim=-1,
    )
    assert streamed.shape == expected.shape
    assert float((streamed - expected).abs().max()) <= 2.0e-5
    decimator.reset_state()
    torch.testing.assert_close(decimator.stream(signal), expected)


def test_polyphase_pair_has_registered_group_delay_and_explicit_cost() -> None:
    coefficients = design_equiripple_halfband()
    interpolator = PolyphaseHalfbandInterpolator2x(coefficients)
    decimator = PolyphaseHalfbandDecimator2x(coefficients)
    impulse = torch.zeros(1, 97)
    impulse[:, 0] = 1.0
    response = decimator(interpolator(impulse))
    assert int(response.abs().argmax(dim=-1)) == 24
    total_cost = (
        interpolator.active_multiply_count_per_input_sample
        + decimator.active_multiply_count_per_output_sample
    )
    assert total_cost == 50


@pytest.mark.parametrize("taps", [8, 11, 15])
def test_halfband_design_rejects_invalid_tap_counts(taps: int) -> None:
    with pytest.raises(ValueError, match="1 mod 4"):
        design_equiripple_halfband(taps)
