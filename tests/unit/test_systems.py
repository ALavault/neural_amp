from __future__ import annotations

import numpy as np
import pytest

from fssr_nam.data.excitations import generate_excitation
from fssr_nam.data.systems import SYSTEMS, apply_system


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_systems_are_deterministic_finite_and_zero_stable(name: str) -> None:
    signal = generate_excitation(
        "level_changes", sample_rate=48_000, duration_seconds=0.1, seed=2
    )
    first = apply_system(name, signal, 48_000)
    second = apply_system(name, signal, 48_000)

    np.testing.assert_array_equal(first, second)
    assert first.shape == signal.shape
    assert first.dtype == np.float32
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(
        apply_system(name, np.zeros(64, dtype=np.float32), 48_000),
        np.zeros(64, dtype=np.float32),
    )


def test_all_systems_produce_distinct_targets() -> None:
    signal = generate_excitation(
        "multisine", sample_rate=48_000, duration_seconds=0.1, seed=2
    )
    outputs = [apply_system(name, signal, 48_000) for name in sorted(SYSTEMS)]
    for left_index, left in enumerate(outputs):
        for right in outputs[left_index + 1 :]:
            assert not np.allclose(left, right, rtol=1.0e-5, atol=1.0e-6)
