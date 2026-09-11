from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from fssr_nam.data.arch_fixtures import (
    ARCH_FIXTURES,
    MECHANISM_SYSTEMS,
    apply_arch_fixture,
    apply_mechanism_system,
    build_mechanism_episodes,
    generate_mechanism_source,
)


@given(
    signal=npst.arrays(
        dtype=np.float32,
        shape=st.integers(min_value=1, max_value=257),
        elements=st.floats(
            min_value=-1.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
    )
)
@settings(max_examples=8, deadline=None)
def test_named_architecture_fixtures_are_finite_and_shape_preserving(
    signal: np.ndarray,
) -> None:
    for fixture in ARCH_FIXTURES:
        output = apply_arch_fixture(fixture, signal)
        assert output.shape == signal.shape
        assert output.dtype == np.float32
        assert np.isfinite(output).all()


@pytest.mark.parametrize("fixture", ARCH_FIXTURES)
def test_architecture_fixtures_are_causal(fixture: str) -> None:
    source = generate_mechanism_source(31, 4_096)
    changed = source.copy()
    changed[2_731:] = np.linspace(-0.8, 0.8, len(changed) - 2_731)
    np.testing.assert_allclose(
        apply_arch_fixture(fixture, source)[:2_731],
        apply_arch_fixture(fixture, changed)[:2_731],
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("system", MECHANISM_SYSTEMS)
def test_mechanism_composites_are_deterministic_non_silent_and_causal(
    system: str,
) -> None:
    source = generate_mechanism_source(42, 8_192)
    output = apply_mechanism_system(system, source)
    repeated = apply_mechanism_system(system, source)
    assert np.var(output) > 1.0e-6
    np.testing.assert_array_equal(output, repeated)
    changed = source.copy()
    changed[6_000:] *= -0.7
    np.testing.assert_array_equal(
        output[:6_000], apply_mechanism_system(system, changed)[:6_000]
    )


def test_dynamic_composite_has_memory_beyond_static_composite() -> None:
    first = np.zeros(8_192, dtype=np.float32)
    second = first.copy()
    first[512:2_560] = 0.7
    second[512:2_560] = 0.1
    first[4_096:] = second[4_096:] = 0.2
    dynamic_difference = np.mean(
        np.abs(
            apply_mechanism_system("dynamic_composite", first)[4_096:]
            - apply_mechanism_system("dynamic_composite", second)[4_096:]
        )
    )
    static_difference = np.mean(
        np.abs(
            apply_mechanism_system("static_composite", first)[4_096:]
            - apply_mechanism_system("static_composite", second)[4_096:]
        )
    )
    assert dynamic_difference > 1.0e-4
    assert static_difference == 0.0


def test_mechanism_episode_splits_are_seed_disjoint_and_reproducible() -> None:
    train = build_mechanism_episodes(split="train", episodes=2, samples=4_096)
    repeated = build_mechanism_episodes(split="train", episodes=2, samples=4_096)
    validation = build_mechanism_episodes(split="validation", episodes=2, samples=4_096)
    assert set(train) == set(MECHANISM_SYSTEMS)
    for system in MECHANISM_SYSTEMS:
        np.testing.assert_array_equal(train[system][0], repeated[system][0])
        np.testing.assert_array_equal(train[system][1], repeated[system][1])
        assert not np.array_equal(train[system][0], validation[system][0])


def test_fixture_interfaces_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="unknown architecture fixture"):
        apply_arch_fixture("post_result_fixture", np.zeros(32, dtype=np.float32))
    with pytest.raises(ValueError, match="unknown architecture mechanism"):
        apply_mechanism_system("post_result_system", np.zeros(32, dtype=np.float32))
    with pytest.raises(ValueError, match="at least 4096"):
        generate_mechanism_source(0, 100)
    with pytest.raises(ValueError, match="split"):
        build_mechanism_episodes(split="test", episodes=1, samples=4_096)
