from __future__ import annotations

import numpy as np
import pytest

from fssr_nam.metrics.time import error_to_signal_ratio


def test_esr_detects_gain_error() -> None:
    target = np.array([1.0, -1.0], dtype=np.float32)
    assert error_to_signal_ratio(0.5 * target, target) == pytest.approx(0.25)


def test_esr_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        error_to_signal_ratio([np.nan], [0.0])
