"""Known synthetic nonlinear systems used as M1 reference targets."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import lfilter

FloatArray = NDArray[np.float32]
System = Callable[[ArrayLike, int], FloatArray]

_PRE_FILTER = np.array([0.05, 0.12, 0.21, 0.29, 0.21, 0.09, 0.03])
_POST_FILTER = np.array([0.04, 0.1, 0.18, 0.24, 0.22, 0.14, 0.08])


def _input(signal: ArrayLike) -> NDArray[np.float64]:
    samples = np.asarray(signal, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("synthetic systems require mono one-dimensional input")
    if not np.all(np.isfinite(samples)):
        raise ValueError("synthetic system input must be finite")
    return samples


def _output(samples: NDArray[np.float64]) -> FloatArray:
    if not np.all(np.isfinite(samples)):
        raise RuntimeError("synthetic system produced non-finite output")
    return np.asarray(samples, dtype=np.float32)


def polynomial(signal: ArrayLike, sample_rate: int) -> FloatArray:
    del sample_rate
    samples = _input(signal)
    return _output(0.82 * samples + 0.14 * samples**2 - 0.22 * samples**3)


def tanh_memoryless(signal: ArrayLike, sample_rate: int) -> FloatArray:
    del sample_rate
    samples = _input(signal)
    return _output(np.tanh(2.8 * samples) / np.tanh(2.8))


def asymmetric_waveshaper(signal: ArrayLike, sample_rate: int) -> FloatArray:
    del sample_rate
    samples = _input(signal)
    positive = np.tanh(2.0 * samples) / np.tanh(2.0)
    negative = 0.72 * np.tanh(4.2 * samples) / np.tanh(4.2)
    return _output(np.where(samples >= 0.0, positive, negative))


def wiener(signal: ArrayLike, sample_rate: int) -> FloatArray:
    del sample_rate
    filtered = lfilter(_PRE_FILTER, [1.0], _input(signal))
    return _output(np.tanh(3.1 * filtered) / np.tanh(3.1))


def hammerstein(signal: ArrayLike, sample_rate: int) -> FloatArray:
    del sample_rate
    shaped = np.tanh(3.1 * _input(signal)) / np.tanh(3.1)
    return _output(lfilter(_POST_FILTER, [1.0], shaped))


def wiener_hammerstein(signal: ArrayLike, sample_rate: int) -> FloatArray:
    del sample_rate
    filtered = lfilter(_PRE_FILTER, [1.0], _input(signal))
    shaped = np.where(
        filtered >= 0.0, np.tanh(2.4 * filtered), 0.8 * np.tanh(4.0 * filtered)
    )
    return _output(lfilter(_POST_FILTER, [1.0], shaped))


def memory_clipper(signal: ArrayLike, sample_rate: int) -> FloatArray:
    samples = _input(signal)
    envelope = 0.0
    attack = np.exp(-1.0 / (0.003 * sample_rate))
    release = np.exp(-1.0 / (0.06 * sample_rate))
    output = np.empty_like(samples)
    for index, sample in enumerate(samples):
        coefficient = attack if abs(sample) > envelope else release
        envelope = coefficient * envelope + (1.0 - coefficient) * abs(sample)
        threshold = max(0.16, 0.52 - 0.45 * envelope)
        output[index] = threshold * np.tanh(2.2 * sample / threshold)
    return _output(output)


def slow_sag(signal: ArrayLike, sample_rate: int) -> FloatArray:
    samples = _input(signal)
    energy_state = 0.0
    bias_state = 0.0
    energy_coefficient = np.exp(-1.0 / (0.12 * sample_rate))
    bias_coefficient = np.exp(-1.0 / (0.28 * sample_rate))
    output = np.empty_like(samples)
    for index, sample in enumerate(samples):
        energy_state = (
            energy_coefficient * energy_state
            + (1.0 - energy_coefficient) * sample * sample
        )
        bias_state = bias_coefficient * bias_state + (1.0 - bias_coefficient) * sample
        gain = 1.0 / (1.0 + 3.2 * energy_state)
        output[index] = 0.9 * np.tanh(3.0 * gain * sample + 0.8 * bias_state)
    return _output(output)


SYSTEMS: dict[str, System] = {
    "polynomial": polynomial,
    "tanh": tanh_memoryless,
    "asymmetric_waveshaper": asymmetric_waveshaper,
    "wiener": wiener,
    "hammerstein": hammerstein,
    "wiener_hammerstein": wiener_hammerstein,
    "memory_clipper": memory_clipper,
    "slow_sag": slow_sag,
}


def apply_system(name: str, signal: ArrayLike, sample_rate: int) -> FloatArray:
    """Apply one named, deterministic system."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    try:
        system = SYSTEMS[name]
    except KeyError as error:
        raise ValueError(f"unknown synthetic system: {name}") from error
    return system(signal, sample_rate)
