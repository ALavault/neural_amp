"""Seed-explicit synthetic data for AMP-COMPETENCE-ARCH-v2."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .arch_fixtures import (
    MECHANISM_SYSTEMS,
    apply_mechanism_system,
    generate_mechanism_source,
)

FloatArray = NDArray[np.float32]


def build_arch_v2_episodes(
    *, source_seed: int, episodes: int, samples: int
) -> dict[str, tuple[FloatArray, FloatArray]]:
    """Build one frozen split without reusing the v1 source seeds."""
    if not isinstance(source_seed, int) or isinstance(source_seed, bool):
        raise ValueError("v2 source seed must be an integer")
    if episodes < 1:
        raise ValueError("v2 episode count must be positive")
    inputs = np.stack(
        [
            generate_mechanism_source(source_seed + index, samples)
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
