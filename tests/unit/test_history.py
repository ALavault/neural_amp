"""History-conditioned gain: the identity has no history slope (the first definition
gave -0.5 to -0.9 on noise), a slow gain reduction gives a negative one, and too little
signal gives NaN rather than a perfect-looking fit."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

from fssr_nam.product.history import HISTORY_S, history_slopes

FS = 48_000


def noise(seconds: float = 8.0) -> np.ndarray:
    return np.random.default_rng(7).standard_normal(int(seconds * FS))


def test_identity_has_no_history_slope() -> None:
    x = noise()
    slopes = history_slopes(x, x, FS)
    for t in HISTORY_S:
        assert abs(slopes[t]["gamma"]) < 1e-9


def test_slow_gain_reduction_gives_a_negative_slope() -> None:
    x = noise() * np.repeat(np.random.default_rng(8).uniform(0.05, 1, 80), FS // 10)
    a = math.exp(-1 / (0.1 * FS))
    envelope = lfilter([1 - a], [1, -a], np.abs(x))
    y = x / (1 + 4 * envelope)
    assert history_slopes(x, y, FS)[0.1]["gamma"] < -0.1


def test_too_little_signal_is_nan() -> None:
    x = noise(0.4)
    slopes = history_slopes(x, x, FS)
    assert slopes["frames"] == 0
    assert all(math.isnan(slopes[t]["gamma"]) for t in HISTORY_S)
