"""Deterministic diagnostic excitations for nonlinear audio systems."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.signal import chirp, lfilter

FloatArray = NDArray[np.float32]


def _time(sample_rate: int, duration_seconds: float) -> NDArray[np.float64]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive")
    return (
        np.arange(round(sample_rate * duration_seconds), dtype=np.float64) / sample_rate
    )


def _bounded(signal: NDArray[np.float64], peak: float = 0.5) -> FloatArray:
    maximum = float(np.max(np.abs(signal), initial=0.0))
    if maximum == 0.0:
        return signal.astype(np.float32)
    return np.asarray(signal * (peak / maximum), dtype=np.float32)


def fixed_sine(sample_rate: int, duration_seconds: float, seed: int) -> FloatArray:
    del seed
    time = _time(sample_rate, duration_seconds)
    return np.asarray(0.45 * np.sin(2.0 * np.pi * 997.0 * time), dtype=np.float32)


def logarithmic_sweep(
    sample_rate: int, duration_seconds: float, seed: int
) -> FloatArray:
    del seed
    time = _time(sample_rate, duration_seconds)
    stop_frequency = min(30_000.0, 0.45 * sample_rate)
    return np.asarray(
        0.45
        * chirp(
            time,
            f0=20.0,
            f1=stop_frequency,
            t1=duration_seconds,
            method="logarithmic",
        ),
        dtype=np.float32,
    )


def two_tone(sample_rate: int, duration_seconds: float, seed: int) -> FloatArray:
    del seed
    time = _time(sample_rate, duration_seconds)
    signal = np.sin(2.0 * np.pi * 997.0 * time) + np.sin(
        2.0 * np.pi * 1_531.0 * time + 0.2
    )
    return _bounded(signal, 0.45)


def multisine(sample_rate: int, duration_seconds: float, seed: int) -> FloatArray:
    time = _time(sample_rate, duration_seconds)
    frequencies = np.array([73, 191, 487, 997, 3_001, 7_001, 13_007], dtype=float)
    frequencies = frequencies[frequencies < 0.4 * sample_rate]
    phases = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, frequencies.size)
    signal = np.sum(
        np.sin(2.0 * np.pi * frequencies[:, None] * time + phases[:, None]), axis=0
    )
    return _bounded(signal, 0.45)


def bursts(sample_rate: int, duration_seconds: float, seed: int) -> FloatArray:
    del seed
    time = _time(sample_rate, duration_seconds)
    period_samples = max(1, round(0.1 * sample_rate))
    active_samples = max(1, round(0.025 * sample_rate))
    gate = (np.arange(time.size) % period_samples) < active_samples
    return np.asarray(
        0.45 * gate * np.sin(2.0 * np.pi * 1_997.0 * time), dtype=np.float32
    )


def colored_noise(sample_rate: int, duration_seconds: float, seed: int) -> FloatArray:
    time = _time(sample_rate, duration_seconds)
    white = np.random.default_rng(seed).standard_normal(time.size)
    colored = lfilter([1.0], [1.0, -0.96], white)
    colored -= np.mean(colored)
    return _bounded(colored, 0.45)


def impulses(sample_rate: int, duration_seconds: float, seed: int) -> FloatArray:
    del seed
    time = _time(sample_rate, duration_seconds)
    signal = np.zeros(time.size, dtype=np.float32)
    spacing = max(1, round(0.05 * sample_rate))
    signal[::spacing] = 0.5
    signal[spacing // 2 :: spacing] = -0.5
    return signal


def level_changes(sample_rate: int, duration_seconds: float, seed: int) -> FloatArray:
    del seed
    time = _time(sample_rate, duration_seconds)
    block = np.floor(8.0 * time / duration_seconds).astype(int)
    levels = np.array([0.03, 0.08, 0.16, 0.32, 0.48, 0.24, 0.1, 0.04])
    envelope = levels[np.minimum(block, levels.size - 1)]
    return np.asarray(envelope * np.sin(2.0 * np.pi * 220.0 * time), dtype=np.float32)


def procedural_plucks(
    sample_rate: int, duration_seconds: float, seed: int
) -> FloatArray:
    """Generate a rights-free guitar-like signal; this is not a recorded DI."""
    time = _time(sample_rate, duration_seconds)
    rng = np.random.default_rng(seed)
    signal = np.zeros(time.size, dtype=np.float64)
    note_starts = np.arange(0.0, duration_seconds, 0.18)
    frequencies = np.array([82.41, 110.0, 146.83, 196.0, 246.94, 329.63])
    for index, start in enumerate(note_starts):
        offset = round(start * sample_rate)
        local_time = time[: time.size - offset]
        frequency = frequencies[index % frequencies.size]
        phase = rng.uniform(0.0, 2.0 * np.pi)
        pluck = np.zeros_like(local_time)
        for harmonic in range(1, 9):
            pluck += (
                np.sin(2.0 * np.pi * harmonic * frequency * local_time + phase)
                / harmonic**1.3
            )
        pluck *= np.exp(-local_time * (2.8 + 0.25 * index))
        signal[offset:] += pluck
    return _bounded(signal, 0.45)


EXCITATIONS: dict[str, Callable[[int, float, int], FloatArray]] = {
    "fixed_sine": fixed_sine,
    "logarithmic_sweep": logarithmic_sweep,
    "two_tone": two_tone,
    "multisine": multisine,
    "bursts": bursts,
    "colored_noise": colored_noise,
    "impulses": impulses,
    "level_changes": level_changes,
    "procedural_plucks": procedural_plucks,
}


def generate_excitation(
    name: str, *, sample_rate: int, duration_seconds: float, seed: int
) -> FloatArray:
    """Generate one named excitation with explicit rate, duration, and seed."""
    try:
        generator = EXCITATIONS[name]
    except KeyError as error:
        raise ValueError(f"unknown excitation: {name}") from error
    output = generator(sample_rate, duration_seconds, seed)
    if output.ndim != 1 or not np.all(np.isfinite(output)):
        raise RuntimeError(f"excitation {name} produced invalid samples")
    return output
