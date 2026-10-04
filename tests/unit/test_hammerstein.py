"""The sweep-based parallel Hammerstein identification recovers known filters."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, lfilter, resample_poly

from fssr_nam.product.hammerstein import Sweep, identify

RATE = 48_000


def sweep_signal(rate: int, sweep: Sweep, peak: float) -> np.ndarray:
    t = np.arange(int(sweep.seconds * rate)) / rate
    k = math.log(sweep.f1 / sweep.f0)
    phase = (
        2 * np.pi * sweep.f0 * sweep.seconds / k * (np.exp(t * k / sweep.seconds) - 1)
    )
    return peak * np.sin(phase)


def filters() -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        butter(2, 4_000, fs=RATE),
        butter(2, 300, btype="high", fs=RATE),
        butter(2, (500, 3_000), btype="band", fs=RATE),
    ]


def response(b: np.ndarray, a: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    z = np.exp(-2j * np.pi * freqs / RATE)
    return np.polyval(b[::-1], z) / np.polyval(a[::-1], z)


def test_recovers_known_filters() -> None:
    sweep = Sweep()
    x = sweep_signal(RATE, sweep, 0.5)
    y = sum(lfilter(b, a, x ** (n + 1)) for n, (b, a) in enumerate(filters()))
    model = identify(y, x, lambda s: s, RATE, sweep, order=3)
    band = (model.freqs > 100) & (model.freqs < 6_000)
    for n, (b, a) in enumerate(filters()):
        true = response(b, a, model.freqs[band])
        err = np.abs(model.g[n][band] - true) / np.abs(true).max()
        assert np.median(err) < 1e-2 and err.max() < 5e-2, (
            n + 1,
            np.median(err),
            err.max(),
        )


def test_recovers_through_decimation() -> None:
    """Input and output at 192 kHz, analysed at 48 kHz after the same decimation."""
    sweep = Sweep()
    x192 = sweep_signal(4 * RATE, sweep, 0.5)
    b, a = butter(2, 2_000, fs=4 * RATE)
    y192 = lfilter(b, a, x192) + 0.3 * x192**3
    model = identify(
        resample_poly(y192, 1, 4),
        x192,
        lambda s: resample_poly(s, 1, 4),
        RATE,
        sweep,
        order=3,
    )
    band = (model.freqs > 100) & (model.freqs < 6_000)
    z = np.exp(-2j * np.pi * model.freqs[band] / (4 * RATE))
    true1 = np.polyval(b[::-1], z) / np.polyval(a[::-1], z)
    assert np.median(np.abs(model.g[0][band] - true1)) < 1e-2
    assert np.median(np.abs(model.g[2][band] - 0.3)) < 1e-2
    assert np.median(np.abs(model.g[1][band])) < 1e-2
