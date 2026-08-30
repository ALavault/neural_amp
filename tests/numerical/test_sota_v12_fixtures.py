from __future__ import annotations

import numpy as np
import pytest

from fssr_nam.data.sota_v12_fixtures import build_sota_v12_system_episodes


def test_v12_fixture_builds_only_one_deterministic_finite_system() -> None:
    first = build_sota_v12_system_episodes(
        system="dynamic_primary", source_seed=91, episodes=2, samples=57_600
    )
    second = build_sota_v12_system_episodes(
        system="dynamic_primary", source_seed=91, episodes=2, samples=57_600
    )
    assert first[0].shape == first[1].shape == (2, 57_600)
    assert first[0].dtype == first[1].dtype == np.float32
    assert np.isfinite(first[0]).all() and np.isfinite(first[1]).all()
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_v12_fixture_rejects_unregistered_or_short_systems() -> None:
    with pytest.raises(ValueError, match="primary system"):
        build_sota_v12_system_episodes(
            system="v2_dynamic_rail_stress", source_seed=1, episodes=1
        )
    with pytest.raises(ValueError, match=r"1\.2 seconds"):
        build_sota_v12_system_episodes(
            system="dynamic_primary", source_seed=1, episodes=1, samples=4_096
        )
