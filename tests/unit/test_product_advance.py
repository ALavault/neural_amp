"""Latency compensation of rendered targets (FIX_ffuzz.md)."""

from __future__ import annotations

import numpy as np

from fssr_nam.product.data import advance


def test_advance_moves_earlier_and_keeps_length() -> None:
    audio = np.arange(20, dtype=np.float32).reshape(10, 2)
    moved = advance(audio, 4)
    assert moved.shape == audio.shape
    assert moved.dtype == audio.dtype
    np.testing.assert_array_equal(moved[:6], audio[4:])
    np.testing.assert_array_equal(moved[6:], 0)
    assert advance(audio, 0) is audio
