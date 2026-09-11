"""Frozen causal 192 -> 96 -> 48 kHz capture derivation for FSSR-R2."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import lfilter

FILTER_ID = "r2-causal-kaiser65-beta8p6-v1"
FILTER_TAPS = 65
FILTER_BETA = 8.6
DERIVATION_TOLERANCE = 2.0e-6


def frozen_decimation_fir() -> NDArray[np.float64]:
    """Return the versioned unity-DC factor-two anti-alias FIR."""
    index = np.arange(FILTER_TAPS, dtype=np.float64) - (FILTER_TAPS - 1) / 2
    cutoff = 0.25
    impulse = 2.0 * cutoff * np.sinc(2.0 * cutoff * index)
    impulse *= np.kaiser(FILTER_TAPS, FILTER_BETA)
    return impulse / np.sum(impulse)


class FrozenDecimator2:
    """Streaming causal factor-two decimator with phase zero at file start."""

    def __init__(self) -> None:
        coefficients = frozen_decimation_fir()
        self.coefficients = coefficients
        self.state = np.zeros(len(coefficients) - 1, dtype=np.float64)
        self.input_samples = 0

    def reset(self) -> None:
        self.state.fill(0.0)
        self.input_samples = 0

    def process(self, samples: ArrayLike) -> NDArray[np.float64]:
        signal = np.asarray(samples, dtype=np.float64)
        if signal.ndim != 1 or not np.isfinite(signal).all():
            raise ValueError("capture decimator input must be finite mono audio")
        filtered, self.state = lfilter(self.coefficients, [1.0], signal, zi=self.state)
        start = (-self.input_samples) % 2
        self.input_samples += len(signal)
        return np.asarray(filtered[start::2], dtype=np.float64)


def derive_capture_rates(
    native_192khz: ArrayLike,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Derive 96 and then 48 kHz without gain normalization or delay removal."""
    first = FrozenDecimator2()
    second = FrozenDecimator2()
    rate_96 = first.process(native_192khz)
    rate_48 = second.process(rate_96)
    return rate_96.astype(np.float32), rate_48.astype(np.float32)
