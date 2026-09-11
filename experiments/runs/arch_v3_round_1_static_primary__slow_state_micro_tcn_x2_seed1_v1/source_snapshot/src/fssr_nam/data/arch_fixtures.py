"""Deterministic architecture-mechanism fixtures for AMP-QUALITY-ARCH-v1."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

from fssr_nam.training.r1_diagnostic import two_clippers_with_interstage_filter

SAMPLE_RATE = 48_000
ARCH_FIXTURES = (
    "tanh",
    "asymmetric_clip",
    "two_clippers",
    "rf2047_memory",
    "slow_sag",
    "attack_release",
    "blocking_distortion",
    "level_transition",
)
STATIC_FIXTURES = ("tanh", "asymmetric_clip")
DYNAMIC_FIXTURES = (
    "rf2047_memory",
    "slow_sag",
    "attack_release",
    "blocking_distortion",
    "level_transition",
)
MECHANISM_SYSTEMS = ("static_composite", "dynamic_composite", "two_clippers")

FloatArray = NDArray[np.float32]


def _validate_signal(signal: NDArray[np.floating]) -> NDArray[np.float64]:
    value = np.asarray(signal, dtype=np.float64)
    if value.ndim != 1 or value.size < 1 or not np.isfinite(value).all():
        raise ValueError("architecture fixture input must be finite mono audio")
    return value


def _delay(signal: NDArray[np.float64], samples: int) -> NDArray[np.float64]:
    output = np.zeros_like(signal)
    if samples < len(signal):
        output[samples:] = signal[:-samples]
    return output


def _asymmetric_clip(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    positive = np.tanh(2.6 * (signal + 0.018)) / np.tanh(2.6)
    negative = 0.76 * np.tanh(4.4 * (signal + 0.018)) / np.tanh(4.4)
    return np.where(signal >= 0.0, positive, negative)


def _rf2047_memory(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    memory = (
        signal
        + 0.31 * _delay(signal, 31)
        - 0.24 * _delay(signal, 257)
        + 0.19 * _delay(signal, 2_046)
    )
    return np.tanh(2.2 * memory) / np.tanh(2.2)


def _slow_sag(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    energy_decay = np.exp(-1.0 / (0.12 * SAMPLE_RATE))
    bias_decay = np.exp(-1.0 / (0.24 * SAMPLE_RATE))
    energy = lfilter([1.0 - energy_decay], [1.0, -energy_decay], signal * signal)
    bias = lfilter([1.0 - bias_decay], [1.0, -bias_decay], signal)
    drive = (3.2 / (1.0 + 4.0 * energy)) * signal + 0.55 * bias
    return np.tanh(drive)


def _attack_release(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    attack = np.exp(-1.0 / (0.004 * SAMPLE_RATE))
    release = np.exp(-1.0 / (0.09 * SAMPLE_RATE))
    envelope = np.zeros_like(signal)
    state = 0.0
    for index, sample in enumerate(signal):
        absolute = abs(sample)
        coefficient = attack if absolute > state else release
        state = coefficient * state + (1.0 - coefficient) * absolute
        envelope[index] = state
    gain = 1.0 / (1.0 + 2.8 * envelope)
    return np.tanh(3.0 * gain * signal)


def _blocking_distortion(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    decay = np.exp(-1.0 / (0.035 * SAMPLE_RATE))
    dc_state = lfilter([1.0 - decay], [1.0, -decay], signal)
    blocked = signal - 0.88 * dc_state
    return _asymmetric_clip(1.15 * blocked)


def _level_transition(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    decay = np.exp(-1.0 / (0.16 * SAMPLE_RATE))
    level = lfilter([1.0 - decay], [1.0, -decay], signal * signal)
    drive = 2.0 + 2.4 * np.tanh(7.0 * level)
    normalization = np.maximum(np.tanh(drive), 1.0e-6)
    return np.tanh(drive * signal) / normalization


def apply_arch_fixture(name: str, signal: NDArray[np.floating]) -> FloatArray:
    """Render one named causal fixture on the frozen 48 kHz grid."""
    source = _validate_signal(signal)
    if name == "tanh":
        output = np.tanh(3.1 * source) / np.tanh(3.1)
    elif name == "asymmetric_clip":
        output = _asymmetric_clip(source)
    elif name == "two_clippers":
        output = two_clippers_with_interstage_filter(source)
    elif name == "rf2047_memory":
        output = _rf2047_memory(source)
    elif name == "slow_sag":
        output = _slow_sag(source)
    elif name == "attack_release":
        output = _attack_release(source)
    elif name == "blocking_distortion":
        output = _blocking_distortion(source)
    elif name == "level_transition":
        output = _level_transition(source)
    else:
        raise ValueError(f"unknown architecture fixture: {name}")
    result = np.asarray(output, dtype=np.float32)
    if result.shape != source.shape or not np.isfinite(result).all():
        raise RuntimeError(f"architecture fixture produced invalid output: {name}")
    return result


def apply_mechanism_system(name: str, signal: NDArray[np.floating]) -> FloatArray:
    """Render one preregistered trainable composite system."""
    source = _validate_signal(signal)
    if name == "static_composite":
        tanh = apply_arch_fixture("tanh", source).astype(np.float64)
        asymmetric = apply_arch_fixture("asymmetric_clip", source).astype(np.float64)
        output = 0.55 * tanh + 0.45 * asymmetric
    elif name == "dynamic_composite":
        output = source
        for fixture in DYNAMIC_FIXTURES:
            output = apply_arch_fixture(fixture, output).astype(np.float64)
        output = 0.78 * output
    elif name == "two_clippers":
        output = apply_arch_fixture(name, source)
    else:
        raise ValueError(f"unknown architecture mechanism system: {name}")
    result = np.asarray(output, dtype=np.float32)
    if not np.isfinite(result).all() or np.var(result) < 1.0e-6:
        raise RuntimeError(f"mechanism system is invalid or silent: {name}")
    return result


def generate_mechanism_source(seed: int, samples: int) -> FloatArray:
    """Generate one rights-free, level-varying excitation episode."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("mechanism seed must be an integer")
    if samples < 4_096:
        raise ValueError("mechanism episodes must contain at least 4096 samples")
    generator = np.random.default_rng(seed)
    noise = generator.standard_normal(samples)
    colored = lfilter([0.08, 0.16, 0.24, 0.16, 0.08], [1.0, -0.22], noise)
    time = np.arange(samples, dtype=np.float64) / SAMPLE_RATE
    frequencies = generator.choice(
        np.array([82.41, 110.0, 146.83, 196.0, 246.94, 329.63]),
        size=3,
        replace=False,
    )
    tonal = sum(
        np.sin(2.0 * np.pi * frequency * time + generator.uniform(-np.pi, np.pi))
        for frequency in frequencies
    ) / len(frequencies)
    levels = generator.uniform(0.08, 0.48, size=(samples + 511) // 512)
    envelope = np.repeat(levels, 512)[:samples]
    source = envelope * (0.62 * colored / (np.std(colored) + 1.0e-8) + 0.38 * tonal)
    source[:256] = 0.0
    source = np.clip(source, -0.92, 0.92)
    result = np.asarray(source, dtype=np.float32)
    if not np.isfinite(result).all() or np.var(result) < 1.0e-5:
        raise RuntimeError("mechanism source generation failed")
    return result


def build_mechanism_episodes(
    *, split: str, episodes: int, samples: int
) -> dict[str, tuple[FloatArray, FloatArray]]:
    """Build seed-disjoint episode matrices for every trainable system."""
    if split not in {"train", "validation"}:
        raise ValueError("mechanism split must be train or validation")
    if episodes < 1:
        raise ValueError("mechanism episode count must be positive")
    base_seed = 20_260_828 if split == "train" else 20_261_828
    inputs = np.stack(
        [
            generate_mechanism_source(base_seed + index, samples)
            for index in range(episodes)
        ]
    )
    return {
        system: (
            inputs.copy(),
            np.stack([apply_mechanism_system(system, episode) for episode in inputs]),
        )
        for system in MECHANISM_SYSTEMS
    }
