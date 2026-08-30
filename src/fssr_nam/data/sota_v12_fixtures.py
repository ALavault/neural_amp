"""Single-system synthetic episodes for AMP-SOTA-PROTOTYPE-v1.2."""

from __future__ import annotations

import numpy as np

from .arch_fixtures import generate_mechanism_source
from .arch_v3_fixtures import PRIMARY_SYSTEMS, apply_arch_v3_system


def build_sota_v12_system_episodes(
    *, system: str, source_seed: int, episodes: int, samples: int = 72_000
) -> tuple[np.ndarray, np.ndarray]:
    """Generate only the currently authorized system target matrix."""
    if system not in PRIMARY_SYSTEMS:
        raise ValueError("unknown v1.2 primary system")
    if not isinstance(source_seed, int) or isinstance(source_seed, bool):
        raise ValueError("v1.2 source seed must be an integer")
    if episodes < 1:
        raise ValueError("v1.2 episode count must be positive")
    if samples < 57_600:
        raise ValueError("v1.2 episodes must span at least 1.2 seconds")
    inputs = np.stack(
        [
            generate_mechanism_source(source_seed + index, samples)
            for index in range(episodes)
        ]
    )
    targets = np.stack([apply_arch_v3_system(system, episode) for episode in inputs])
    if inputs.shape != targets.shape or inputs.dtype != np.float32:
        raise RuntimeError("v1.2 fixture matrices changed shape or dtype")
    if not np.isfinite(targets).all():
        raise RuntimeError("v1.2 fixture targets are non-finite")
    return inputs, np.asarray(targets, dtype=np.float32)
