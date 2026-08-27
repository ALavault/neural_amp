from __future__ import annotations

import numpy as np
import pytest

from fssr_nam.data.synthetic import generate_identity_fixture
from fssr_nam.metrics.time import error_to_signal_ratio


def test_identity_fixture_is_deterministic_and_exact() -> None:
    first = generate_identity_fixture(seed=17)
    second = generate_identity_fixture(seed=17)

    np.testing.assert_array_equal(first.input, second.input)
    np.testing.assert_array_equal(first.output, first.input)
    assert error_to_signal_ratio(first.output, first.input) == pytest.approx(0.0)


@pytest.mark.parametrize("sample_rate", [0, -1])
def test_identity_fixture_rejects_invalid_sample_rate(sample_rate: int) -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        generate_identity_fixture(sample_rate=sample_rate)
