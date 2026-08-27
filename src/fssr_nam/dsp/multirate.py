"""Explicit linear-phase FIR decimation for high-rate reference generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import firwin


@dataclass(frozen=True)
class DecimationConfig:
    """A documented low-pass and integer decimation configuration."""

    factor: int = 2
    taps: int = 255
    cutoff_fraction_of_target_nyquist: float = 0.9
    kaiser_beta: float = 8.6

    def validate(self) -> None:
        if self.factor < 2:
            raise ValueError("factor must be at least two")
        if self.taps < 3 or self.taps % 2 == 0:
            raise ValueError("taps must be an odd integer of at least three")
        if not 0.0 < self.cutoff_fraction_of_target_nyquist < 1.0:
            raise ValueError("cutoff fraction must lie strictly between zero and one")


DEFAULT_DECIMATION_CONFIG = DecimationConfig()


def design_decimation_filter(config: DecimationConfig) -> NDArray[np.float64]:
    """Design the exact FIR used before downsampling."""
    config.validate()
    normalized_cutoff = config.cutoff_fraction_of_target_nyquist / config.factor
    return firwin(
        config.taps,
        normalized_cutoff,
        window=("kaiser", config.kaiser_beta),
        pass_zero="lowpass",
        scale=True,
    )


def controlled_decimate(
    signal: ArrayLike, config: DecimationConfig = DEFAULT_DECIMATION_CONFIG
) -> NDArray[np.float32]:
    """Low-pass, compensate the known FIR delay, and downsample.

    This is an offline reference operation. The symmetric FIR group delay is
    explicitly removed before sampling; zero extension defines the boundaries.
    """
    samples = np.asarray(signal, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("decimation requires a one-dimensional mono signal")
    if not np.all(np.isfinite(samples)):
        raise ValueError("decimation input must be finite")
    kernel = design_decimation_filter(config)
    group_delay = (kernel.size - 1) // 2
    filtered_full = np.convolve(samples, kernel, mode="full")
    compensated = filtered_full[group_delay : group_delay + samples.size]
    return np.asarray(compensated[:: config.factor], dtype=np.float32)


def derive_reference_rates(
    master_192k: ArrayLike,
) -> dict[int, NDArray[np.float32]]:
    """Derive 96 and 48 kHz references by two documented x2 stages."""
    master = np.asarray(master_192k, dtype=np.float32)
    at_96k = controlled_decimate(master)
    at_48k = controlled_decimate(at_96k)
    return {192_000: master, 96_000: at_96k, 48_000: at_48k}
