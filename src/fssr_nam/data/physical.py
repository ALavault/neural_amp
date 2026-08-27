"""Conservative preparation helpers for paired physical-device audio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import resample_poly


@dataclass(frozen=True)
class PreparedPair:
    input: NDArray[np.float32]
    target: NDArray[np.float32]
    sample_rate: int


def prepare_pair(
    input_signal: ArrayLike,
    target_signal: ArrayLike,
    *,
    source_rate: int,
    target_rate: int = 48_000,
    trim_start_seconds: float = 0.0,
    trim_end_seconds: float = 0.0,
    maximum_seconds: float | None = None,
) -> PreparedPair:
    """Trim and jointly resample an already paired mono recording.

    No gain normalization or independent alignment is applied. The same
    deterministic polyphase operation is used for both channels so their
    relative timing is preserved.
    """
    x = np.asarray(input_signal, dtype=np.float32)
    y = np.asarray(target_signal, dtype=np.float32)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("input and target must be equal-length mono signals")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("input and target must contain only finite samples")
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if trim_start_seconds < 0.0 or trim_end_seconds < 0.0:
        raise ValueError("trim durations must be non-negative")
    if maximum_seconds is not None and maximum_seconds <= 0.0:
        raise ValueError("maximum_seconds must be positive when provided")

    start = round(trim_start_seconds * source_rate)
    stop = x.size - round(trim_end_seconds * source_rate)
    if stop <= start:
        raise ValueError("trimming would remove the complete recording")
    if maximum_seconds is not None:
        stop = min(stop, start + round(maximum_seconds * source_rate))
    x = x[start:stop]
    y = y[start:stop]

    if source_rate != target_rate:
        divisor = int(np.gcd(source_rate, target_rate))
        up = target_rate // divisor
        down = source_rate // divisor
        window = ("kaiser", 8.6)
        x = resample_poly(x, up, down, window=window)
        y = resample_poly(y, up, down, window=window)
    return PreparedPair(
        input=np.asarray(x, dtype=np.float32),
        target=np.asarray(y, dtype=np.float32),
        sample_rate=target_rate,
    )
