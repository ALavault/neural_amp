"""Long, progressive synthetic fixtures for AMP-QUALITY-ARCH-v3."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .arch_fixtures import (
    apply_arch_fixture,
    apply_mechanism_system,
    generate_mechanism_source,
)

SAMPLE_RATE = 48_000
PRIMARY_SYSTEMS = (
    "static_primary",
    "dynamic_primary",
    "two_clippers_primary",
)
DIAGNOSTIC_SYSTEMS = (
    "tanh_component",
    "asymmetric_component",
    "memory_component",
    "sag_component",
    "envelope_component",
    "blocking_component",
    "level_component",
)
STRESS_SYSTEMS = ("v2_dynamic_rail_stress",)
ALL_SYSTEMS = (*PRIMARY_SYSTEMS, *DIAGNOSTIC_SYSTEMS, *STRESS_SYSTEMS)

_COMPONENT_FIXTURE = {
    "tanh_component": "tanh",
    "asymmetric_component": "asymmetric_clip",
    "memory_component": "rf2047_memory",
    "sag_component": "slow_sag",
    "envelope_component": "attack_release",
    "blocking_component": "blocking_distortion",
    "level_component": "level_transition",
}

FloatArray = NDArray[np.float32]


def _validate_signal(signal: NDArray[np.floating]) -> NDArray[np.float64]:
    value = np.asarray(signal, dtype=np.float64)
    if value.ndim != 1 or value.size < 1 or not np.isfinite(value).all():
        raise ValueError("architecture v3 fixture input must be finite mono audio")
    return value


def _dynamic_primary(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mix complementary stateful mechanisms without the v2 rail collapse."""
    components = {
        name: apply_arch_fixture(name, signal).astype(np.float64)
        for name in (
            "rf2047_memory",
            "slow_sag",
            "attack_release",
            "blocking_distortion",
            "level_transition",
        )
    }
    return (
        0.15 * signal
        + 0.25 * components["rf2047_memory"]
        + 0.20 * components["slow_sag"]
        + 0.15 * components["attack_release"]
        + 0.15 * components["blocking_distortion"]
        + 0.10 * components["level_transition"]
    )


def apply_arch_v3_system(name: str, signal: NDArray[np.floating]) -> FloatArray:
    """Render one v3 primary, component, or explicitly isolated stress system."""
    source = _validate_signal(signal)
    if name == "static_primary":
        output = apply_mechanism_system("static_composite", source)
    elif name == "dynamic_primary":
        output = _dynamic_primary(source)
    elif name == "two_clippers_primary":
        output = apply_mechanism_system("two_clippers", source)
    elif name == "v2_dynamic_rail_stress":
        output = apply_mechanism_system("dynamic_composite", source)
    elif name in _COMPONENT_FIXTURE:
        output = apply_arch_fixture(_COMPONENT_FIXTURE[name], source)
    else:
        raise ValueError(f"unknown architecture v3 system: {name}")
    result = np.asarray(output, dtype=np.float32)
    if result.shape != source.shape or not np.isfinite(result).all():
        raise RuntimeError(f"architecture v3 system produced invalid output: {name}")
    if np.var(result) < 1.0e-6:
        raise RuntimeError(f"architecture v3 system is effectively silent: {name}")
    return result


def build_arch_v3_episodes(
    *, source_seed: int, episodes: int, samples: int = 72_000
) -> dict[str, tuple[FloatArray, FloatArray]]:
    """Build a deterministic long split without normalization or split reuse."""
    if not isinstance(source_seed, int) or isinstance(source_seed, bool):
        raise ValueError("v3 source seed must be an integer")
    if episodes < 1:
        raise ValueError("v3 episode count must be positive")
    if samples < 57_600:
        raise ValueError("v3 episodes must span at least 1.2 seconds")
    inputs = np.stack(
        [
            generate_mechanism_source(source_seed + index, samples)
            for index in range(episodes)
        ]
    )
    return {
        system: (
            inputs.copy(),
            np.stack([apply_arch_v3_system(system, episode) for episode in inputs]),
        )
        for system in ALL_SYSTEMS
    }


def history_collision_pair(
    *, receptive_field_samples: int, probe_samples: int = 128
) -> tuple[FloatArray, FloatArray]:
    """Create equal local contexts with distinct older excitation histories."""
    if receptive_field_samples < 1:
        raise ValueError("receptive field must be positive")
    if probe_samples < 1:
        raise ValueError("probe length must be positive")
    common_samples = receptive_field_samples + probe_samples
    samples = max(72_000, 2 * common_samples)
    time = np.arange(samples, dtype=np.float64) / SAMPLE_RATE
    quiet_history = np.zeros(samples, dtype=np.float64)
    excited_history = np.zeros(samples, dtype=np.float64)
    excited_history[:-common_samples] = 0.72 * np.sin(
        2.0 * np.pi * 91.0 * time[:-common_samples]
    )
    probe_time = np.arange(probe_samples, dtype=np.float64) / SAMPLE_RATE
    probe = 0.28 * np.sin(2.0 * np.pi * 440.0 * probe_time)
    quiet_history[-probe_samples:] = probe
    excited_history[-probe_samples:] = probe
    return (
        np.asarray(quiet_history, dtype=np.float32),
        np.asarray(excited_history, dtype=np.float32),
    )
