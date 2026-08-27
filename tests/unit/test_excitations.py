from __future__ import annotations

import numpy as np
import pytest

from fssr_nam.data.excitations import EXCITATIONS, generate_excitation


@pytest.mark.parametrize("name", sorted(EXCITATIONS))
def test_excitation_is_deterministic_finite_and_bounded(name: str) -> None:
    first = generate_excitation(name, sample_rate=48_000, duration_seconds=0.1, seed=7)
    second = generate_excitation(name, sample_rate=48_000, duration_seconds=0.1, seed=7)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (4_800,)
    assert first.dtype == np.float32
    assert np.all(np.isfinite(first))
    assert np.max(np.abs(first)) <= 0.500001
