from __future__ import annotations

import numpy as np

from fssr_nam.data.arch_v2_fixtures import build_arch_v2_episodes


def test_v2_fixture_splits_are_deterministic_finite_and_seed_disjoint() -> None:
    first = build_arch_v2_episodes(source_seed=20360828, episodes=2, samples=4096)
    repeat = build_arch_v2_episodes(source_seed=20360828, episodes=2, samples=4096)
    other = build_arch_v2_episodes(source_seed=20361828, episodes=2, samples=4096)
    assert set(first) == {"static_composite", "dynamic_composite", "two_clippers"}
    for system, (inputs, targets) in first.items():
        np.testing.assert_array_equal(inputs, repeat[system][0])
        np.testing.assert_array_equal(targets, repeat[system][1])
        assert inputs.shape == targets.shape == (2, 4096)
        assert np.isfinite(inputs).all() and np.isfinite(targets).all()
        assert not np.array_equal(inputs, other[system][0])
