"""Deterministic six-system qualification fixture set for FSSR-R2.

The fixture coefficients are expressed on the frozen 48 kHz time grid.  When a
fixture is rendered inside an x2/x4 full-rate island, delays and FIR taps are
zero-insertion scaled so that their physical horizons do not change.  The
optional ADAA route replaces every memoryless shaper with the same analytic
first-order divided difference used by the deployable R2 primitives.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import lfilter

R2_FIXTURES = (
    "tanh",
    "asymmetric_clipping",
    "two_clippers",
    "short_memory",
    "slow_sag",
    "rf2047_residual",
)
RF2047_DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
BASE_SAMPLE_RATE = 48_000
MECHANISM_SAMPLE_RATES = (48_000, 96_000, 192_000, 384_000, 768_000)
ADAA_THRESHOLD = 1.0e-4


class FixtureOutput(NamedTuple):
    output: NDArray[np.float32]
    residual: NDArray[np.float32]
    receptive_field: int


def _input(signal: ArrayLike, sample_rate: int) -> NDArray[np.float64]:
    samples = np.asarray(signal, dtype=np.float64)
    if samples.ndim != 1 or samples.size < 1 or not np.isfinite(samples).all():
        raise ValueError("R2 fixture input must be finite, mono, and non-empty")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return samples


def _delay(signal: NDArray[np.float64], samples: int) -> NDArray[np.float64]:
    delayed = np.zeros_like(signal)
    if samples < len(signal):
        delayed[samples:] = signal[:-samples]
    return delayed


def _rate_scale(sample_rate: int) -> int:
    if sample_rate not in MECHANISM_SAMPLE_RATES:
        raise ValueError(
            "mechanism fixtures require 48, 96, 192, 384, or 768 kHz sample rate"
        )
    return sample_rate // BASE_SAMPLE_RATE


def _scaled_fir(coefficients: ArrayLike, scale: int) -> NDArray[np.float64]:
    base = np.asarray(coefficients, dtype=np.float64)
    scaled = np.zeros((base.size - 1) * scale + 1, dtype=np.float64)
    scaled[::scale] = base
    return scaled


def _log_cosh_zeroed(inputs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Stable log(cosh(x)) with an exact zero at x=0."""
    return np.logaddexp(inputs, -inputs) - np.log(2.0)


def _tanh_pair(
    gain: float, *, multiplier: float = 1.0, normalization: float = 1.0
) -> tuple:
    factor = multiplier / normalization

    def shape(inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        return factor * np.tanh(gain * inputs)

    def primitive(inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        return factor * _log_cosh_zeroed(gain * inputs) / gain

    return shape, primitive


def _asymmetric_pair(
    *,
    positive_gain: float,
    negative_gain: float,
    negative_multiplier: float,
    normalize_branches: bool,
) -> tuple:
    positive_normalization = np.tanh(positive_gain) if normalize_branches else 1.0
    negative_normalization = np.tanh(negative_gain) if normalize_branches else 1.0
    positive, positive_primitive = _tanh_pair(
        positive_gain, normalization=float(positive_normalization)
    )
    negative, negative_primitive = _tanh_pair(
        negative_gain,
        multiplier=negative_multiplier,
        normalization=float(negative_normalization),
    )

    def shape(inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(inputs >= 0.0, positive(inputs), negative(inputs))

    def primitive(inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(
            inputs >= 0.0,
            positive_primitive(inputs),
            negative_primitive(inputs),
        )

    return shape, primitive


def _apply_shape(
    inputs: NDArray[np.float64],
    pair: tuple,
    *,
    adaa: bool,
) -> NDArray[np.float64]:
    shape, primitive = pair
    if not adaa:
        return shape(inputs)
    previous = np.empty_like(inputs)
    previous[0] = 0.0
    previous[1:] = inputs[:-1]
    difference = inputs - previous
    midpoint = 0.5 * (inputs + previous)
    small = np.abs(difference) < ADAA_THRESHOLD
    safe_difference = np.where(small, 1.0, difference)
    divided = (primitive(inputs) - primitive(previous)) / safe_difference
    return np.where(small, shape(midpoint), divided)


def _slow_sag_drive(
    samples: NDArray[np.float64], sample_rate: int
) -> NDArray[np.float64]:
    """Vectorized state update equivalent to ``systems.slow_sag``."""
    energy_coefficient = np.exp(-1.0 / (0.12 * sample_rate))
    bias_coefficient = np.exp(-1.0 / (0.28 * sample_rate))
    energy_state = lfilter(
        [1.0 - energy_coefficient], [1.0, -energy_coefficient], samples * samples
    )
    bias_state = lfilter([1.0 - bias_coefficient], [1.0, -bias_coefficient], samples)
    gain = 1.0 / (1.0 + 3.2 * energy_state)
    return 3.0 * gain * samples + 0.8 * bias_state


def apply_r2_fixture(
    name: str,
    signal: ArrayLike,
    sample_rate: int,
    *,
    adaa: bool = False,
) -> FixtureOutput:
    """Apply one preregistered mechanism fixture without random state."""
    samples = _input(signal, sample_rate)
    scale = _rate_scale(sample_rate)
    residual = np.zeros_like(samples)
    receptive_field = 1
    if name == "tanh":
        output = _apply_shape(
            samples,
            _tanh_pair(2.8, normalization=float(np.tanh(2.8))),
            adaa=adaa,
        )
    elif name == "asymmetric_clipping":
        output = _apply_shape(
            samples,
            _asymmetric_pair(
                positive_gain=2.0,
                negative_gain=4.2,
                negative_multiplier=0.72,
                normalize_branches=True,
            ),
            adaa=adaa,
        )
    elif name == "two_clippers":
        first_input = lfilter(_scaled_fir([0.18, 0.31, 0.51], scale), [1.0], samples)
        first = _apply_shape(first_input, _tanh_pair(3.0), adaa=adaa)
        middle = lfilter(
            _scaled_fir([0.12, 0.22, 0.32, 0.22, 0.12], scale),
            [1.0],
            first,
        )
        second = _apply_shape(
            middle,
            _asymmetric_pair(
                positive_gain=2.4,
                negative_gain=4.1,
                negative_multiplier=0.78,
                normalize_branches=False,
            ),
            adaa=adaa,
        )
        output = lfilter(_scaled_fir([0.2, 0.3, 0.5], scale), [1.0], second)
        receptive_field = 8 * scale + 1
    elif name == "short_memory":
        pair = _tanh_pair(2.8, normalization=float(np.tanh(2.8)))
        output = 0.60 * _apply_shape(samples, pair, adaa=adaa)
        output += 0.25 * _apply_shape(
            samples + 0.9 * _delay(samples, 3 * scale), pair, adaa=adaa
        )
        output += 0.15 * _apply_shape(
            samples - 0.8 * _delay(samples, 17 * scale), pair, adaa=adaa
        )
        receptive_field = 17 * scale + 1
    elif name == "slow_sag":
        drive = _slow_sag_drive(samples, sample_rate)
        output = _apply_shape(drive, _tanh_pair(1.0, multiplier=0.9), adaa=adaa)
        receptive_field = 1
    elif name == "rf2047_residual":
        base = _apply_shape(samples, _tanh_pair(2.6), adaa=adaa)
        residual = 0.60 * _apply_shape(
            _delay(samples, 2046 * scale), _tanh_pair(2.2), adaa=adaa
        )
        residual += 0.15 * _apply_shape(
            _delay(samples, 682 * scale), _tanh_pair(1.7), adaa=adaa
        )
        output = base + residual
        receptive_field = 2046 * scale + 1
    else:
        raise ValueError(f"unknown R2 fixture: {name}")
    if not np.isfinite(output).all():
        raise RuntimeError("R2 fixture produced non-finite output")
    return FixtureOutput(
        np.asarray(output, dtype=np.float32),
        np.asarray(residual, dtype=np.float32),
        receptive_field,
    )


def residual_energy_ratio(result: FixtureOutput, epsilon: float = 1.0e-12) -> float:
    """Report the RF2047 residual/output energy guard without pseudo-samples."""
    numerator = float(np.sum(result.residual.astype(np.float64) ** 2))
    denominator = float(np.sum(result.output.astype(np.float64) ** 2))
    return numerator / (denominator + epsilon)
