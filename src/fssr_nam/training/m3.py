"""Reusable model and target factories for M3 synthetic diagnostics."""

from __future__ import annotations

import numpy as np
import torch

from fssr_nam.data.systems import apply_system
from fssr_nam.models import (
    S0Structured,
    S1Slow,
    S2Residual,
    S3FastSlowResidual,
    S4Antialiased,
)


def short_nonlinear_memory(signal: np.ndarray) -> np.ndarray:
    delay_3 = np.zeros_like(signal)
    delay_17 = np.zeros_like(signal)
    delay_3[3:] = signal[:-3]
    delay_17[17:] = signal[:-17]
    denominator = np.tanh(2.8)
    target = (
        0.60 * np.tanh(2.8 * signal)
        + 0.25 * np.tanh(2.8 * (signal + 0.9 * delay_3))
        + 0.15 * np.tanh(2.8 * (signal - 0.8 * delay_17))
    ) / denominator
    return np.asarray(target, dtype=np.float32)


def synthetic_target(case: str, signal: np.ndarray, sample_rate: int) -> np.ndarray:
    if case in {"tanh", "slow_sag"}:
        return apply_system(case, signal, sample_rate)
    if case == "short_nonlinear_memory":
        return short_nonlinear_memory(signal)
    raise ValueError(f"unknown synthetic case: {case}")


def model_factory(variant: str, config: dict):
    common = {
        "taps": int(config["taps"]),
        "num_knots": int(config["num_knots"]),
    }
    spline_range = float(config.get("spline_range", 2.0))
    if variant == "S0":
        return S0Structured(**common, spline_range=spline_range)
    if variant == "S1":
        return S1Slow(
            **common,
            hidden_size=int(config["slow_hidden_size"]),
            decimation=int(config["slow_decimation"]),
            spline_range=spline_range,
        )
    if variant == "S2":
        return S2Residual(
            **common,
            residual_channels=int(config["residual_channels"]),
            spline_range=spline_range,
        )
    if variant == "S3":
        return S3FastSlowResidual(
            **common,
            hidden_size=int(config["slow_hidden_size"]),
            decimation=int(config["slow_decimation"]),
            residual_channels=int(config["residual_channels"]),
            spline_range=spline_range,
        )
    if variant == "S4":
        return S4Antialiased(
            **common,
            hidden_size=int(config["slow_hidden_size"]),
            decimation=int(config["slow_decimation"]),
            residual_channels=int(config["residual_channels"]),
            oversampling_filter_taps=int(config["oversampling_filter_taps"]),
        )
    raise ValueError(f"unknown variant: {variant}")


def delay_target(target_signal: torch.Tensor, latency: int) -> torch.Tensor:
    if latency == 0:
        return target_signal
    return torch.nn.functional.pad(target_signal, (latency, 0))[:-latency]
