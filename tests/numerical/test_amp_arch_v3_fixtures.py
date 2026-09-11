from __future__ import annotations

import numpy as np

from fssr_nam.data.arch_v3_fixtures import (
    ALL_SYSTEMS,
    PRIMARY_SYSTEMS,
    apply_arch_v3_system,
    build_arch_v3_episodes,
    history_collision_pair,
)


def test_v3_systems_are_finite_causal_and_nonconstant() -> None:
    generator = np.random.default_rng(17)
    prefix = generator.uniform(-0.5, 0.5, 4096).astype(np.float32)
    suffix_a = generator.uniform(-0.5, 0.5, 512).astype(np.float32)
    suffix_b = generator.uniform(-0.5, 0.5, 512).astype(np.float32)
    first = np.concatenate((prefix, suffix_a))
    second = np.concatenate((prefix, suffix_b))
    for system in ALL_SYSTEMS:
        first_output = apply_arch_v3_system(system, first)
        second_output = apply_arch_v3_system(system, second)
        assert first_output.shape == first.shape
        assert np.isfinite(first_output).all()
        assert np.var(first_output) > 1.0e-6
        np.testing.assert_array_equal(
            first_output[: len(prefix)], second_output[: len(prefix)]
        )


def test_v3_long_train_split_is_deterministic_and_not_rail_collapsed() -> None:
    first = build_arch_v3_episodes(source_seed=30360828, episodes=1)
    repeat = build_arch_v3_episodes(source_seed=30360828, episodes=1)
    for system in ALL_SYSTEMS:
        np.testing.assert_array_equal(first[system][0], repeat[system][0])
        np.testing.assert_array_equal(first[system][1], repeat[system][1])
    for system in PRIMARY_SYSTEMS:
        target = first[system][1][:, 14_400:].reshape(-1)
        peak = np.max(np.abs(target))
        assert 0.05 <= np.sqrt(np.mean(np.square(target))) <= 0.8
        assert np.mean(np.abs(target) >= 0.95 * peak) <= 0.5


def test_v3_history_probe_exposes_short_rf_but_not_long_rf() -> None:
    collision_rms = {}
    for receptive_field in (2047, 12283):
        first, second = history_collision_pair(receptive_field_samples=receptive_field)
        assert np.array_equal(first[-receptive_field:], second[-receptive_field:])
        first_target = apply_arch_v3_system("dynamic_primary", first)
        second_target = apply_arch_v3_system("dynamic_primary", second)
        difference = first_target[-128:] - second_target[-128:]
        collision_rms[receptive_field] = float(np.sqrt(np.mean(np.square(difference))))
    assert collision_rms[2047] >= 0.005
    assert collision_rms[12283] < 0.005
