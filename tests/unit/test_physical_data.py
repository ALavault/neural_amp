import numpy as np
import pytest

from fssr_nam.data.physical import prepare_pair


def test_prepare_pair_preserves_level_and_relative_pairing() -> None:
    x = np.linspace(-0.4, 0.4, 4_410, dtype=np.float32)
    y = 2.0 * x + 0.1
    pair = prepare_pair(
        x,
        y,
        source_rate=44_100,
        target_rate=48_000,
        trim_start_seconds=0.01,
        trim_end_seconds=0.01,
        maximum_seconds=0.05,
    )
    assert pair.sample_rate == 48_000
    assert pair.input.shape == pair.target.shape == (2_400,)
    interior = slice(64, -64)
    assert (
        np.max(np.abs(pair.target[interior] - (2.0 * pair.input[interior] + 0.1)))
        < 2.0e-3
    )
    assert np.max(np.abs(pair.input)) < 0.4


def test_prepare_pair_rejects_unpaired_shapes() -> None:
    with pytest.raises(ValueError, match="equal-length mono"):
        prepare_pair(np.zeros(10), np.zeros(9), source_rate=48_000)
