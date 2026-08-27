"""Deterministic synthetic signals used to validate the experimental stack."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class IdentityFixture:
    """Paired mono input/output samples for an exact identity system."""

    input: NDArray[np.float32]
    output: NDArray[np.float32]
    sample_rate: int
    seed: int


def generate_identity_fixture(
    *, sample_rate: int = 48_000, duration_seconds: float = 0.25, seed: int = 0
) -> IdentityFixture:
    """Generate a bounded, deterministic excitation and its identity target."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive")

    sample_count = round(sample_rate * duration_seconds)
    rng = np.random.default_rng(seed)
    time = np.arange(sample_count, dtype=np.float64) / sample_rate
    signal = (
        0.12 * np.sin(2.0 * np.pi * 97.0 * time)
        + 0.08 * np.sin(2.0 * np.pi * 997.0 * time + 0.3)
        + 0.03 * rng.standard_normal(sample_count)
    )
    input_signal = np.asarray(np.clip(signal, -0.5, 0.5), dtype=np.float32)
    return IdentityFixture(
        input=input_signal,
        output=input_signal.copy(),
        sample_rate=sample_rate,
        seed=seed,
    )
