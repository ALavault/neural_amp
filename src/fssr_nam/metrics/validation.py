"""Controlled perturbations used to validate metric sensitivity and limits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt

from fssr_nam.alignment.delay import apply_fractional_delay, estimate_delay
from fssr_nam.data.systems import apply_system
from fssr_nam.dsp.multirate import controlled_decimate, derive_reference_rates
from fssr_nam.metrics.nonlinear import (
    complex_harmonic_error,
    inharmonic_energy_ratio,
    known_reference_parasite_db,
)
from fssr_nam.metrics.spectral import spectral_metrics
from fssr_nam.metrics.time import time_metrics
from fssr_nam.metrics.transient import envelope_error

SAMPLE_RATE = 48_000
SAMPLE_COUNT = 48_000
FUNDAMENTAL_HZ = 250.0
PERMITTED_FREQUENCIES_HZ = (250.0, 750.0, 1_250.0, 7_000.0, 11_000.0)


@dataclass(frozen=True)
class Perturbation:
    prediction: NDArray[np.float64]
    target: NDArray[np.float64]
    fundamental_hz: float = FUNDAMENTAL_HZ
    permitted_frequencies_hz: tuple[float, ...] = PERMITTED_FREQUENCIES_HZ


def _tone(frequency_hz: float, sample_rate: int, sample_count: int) -> np.ndarray:
    indices = np.arange(sample_count, dtype=np.float64)
    return np.sin(2.0 * np.pi * frequency_hz * indices / sample_rate)


def _base_target() -> NDArray[np.float64]:
    return np.asarray(
        0.50 * _tone(250.0, SAMPLE_RATE, SAMPLE_COUNT)
        + 0.18 * _tone(750.0, SAMPLE_RATE, SAMPLE_COUNT)
        + 0.09 * _tone(1_250.0, SAMPLE_RATE, SAMPLE_COUNT)
        + 0.08 * _tone(7_000.0, SAMPLE_RATE, SAMPLE_COUNT)
        + 0.05 * _tone(11_000.0, SAMPLE_RATE, SAMPLE_COUNT),
        dtype=np.float64,
    )


def _zero_shift(signal: np.ndarray, samples: int) -> np.ndarray:
    shifted = np.zeros_like(signal)
    shifted[samples:] = signal[:-samples]
    return shifted


def _controlled_alias_pairs() -> dict[str, Perturbation]:
    master_rate = 192_000
    master_count = 4 * SAMPLE_COUNT
    master_input = 0.48 * _tone(9_000.0, master_rate, master_count)
    master_target = apply_system("tanh", master_input, master_rate)
    reference = derive_reference_rates(master_target)[SAMPLE_RATE].astype(np.float64)
    inputs_by_rate = derive_reference_rates(master_input)
    low_rate_input = inputs_by_rate[SAMPLE_RATE]
    naive = apply_system("tanh", low_rate_input, SAMPLE_RATE).astype(np.float64)
    at_96k = apply_system("tanh", inputs_by_rate[96_000], 96_000)
    local_x2 = controlled_decimate(at_96k).astype(np.float64)
    common = {
        "target": reference,
        "fundamental_hz": 9_000.0,
        "permitted_frequencies_hz": (9_000.0,),
    }
    return {
        "controlled_aliasing": Perturbation(prediction=naive, **common),
        "controlled_aliasing_x2": Perturbation(prediction=local_x2, **common),
    }


def build_perturbations() -> dict[str, Perturbation]:
    target = _base_target()
    rng = np.random.default_rng(91)
    harmonic_suppressed = target - 0.18 * _tone(750.0, SAMPLE_RATE, SAMPLE_COUNT)
    lowpass = sosfiltfilt(
        butter(6, 4_000.0, btype="lowpass", fs=SAMPLE_RATE, output="sos"), target
    )
    ringing = (
        0.06
        * np.exp(-np.arange(SAMPLE_COUNT) / (0.035 * SAMPLE_RATE))
        * _tone(6_000.0, SAMPLE_RATE, SAMPLE_COUNT)
    )
    ringing = np.roll(ringing, SAMPLE_COUNT // 2)
    ringing[: SAMPLE_COUNT // 2] = 0.0
    spectrum = np.fft.rfft(target)
    angular_frequency = 2.0 * np.pi * np.fft.rfftfreq(target.size)
    phase_only = np.fft.irfft(
        spectrum * np.exp(-0.6j * angular_frequency), n=target.size
    )
    perturbations = {
        "identity": Perturbation(target.copy(), target),
        "gain_error": Perturbation(0.8 * target, target),
        "integer_delay": Perturbation(_zero_shift(target, 8), target),
        "fractional_delay": Perturbation(apply_fractional_delay(target, 0.35), target),
        "polarity_inversion": Perturbation(-target, target),
        "dc_offset": Perturbation(target + 0.05, target),
        "lowpass": Perturbation(lowpass, target),
        "added_noise": Perturbation(
            target + 0.01 * rng.standard_normal(target.size), target
        ),
        "harmonic_suppression": Perturbation(harmonic_suppressed, target),
        "inharmonic_addition": Perturbation(
            target + 0.05 * _tone(1_337.0, SAMPLE_RATE, SAMPLE_COUNT), target
        ),
        "ringing": Perturbation(target + ringing, target),
        "phase_only": Perturbation(phase_only, target),
    }
    perturbations.update(_controlled_alias_pairs())
    return perturbations


def evaluate_perturbation(perturbation: Perturbation) -> dict[str, float]:
    prediction = perturbation.prediction
    target = perturbation.target
    metrics = time_metrics(prediction, target)
    metrics.update(spectral_metrics(prediction, target))
    metrics["envelope_error"] = envelope_error(
        prediction, target, sample_rate=SAMPLE_RATE
    )
    metrics["complex_harmonic_error"] = complex_harmonic_error(
        prediction,
        target,
        fundamental_hz=perturbation.fundamental_hz,
        sample_rate=SAMPLE_RATE,
    )
    metrics["inharmonic_energy_ratio"] = inharmonic_energy_ratio(
        prediction,
        permitted_frequencies_hz=perturbation.permitted_frequencies_hz,
        sample_rate=SAMPLE_RATE,
    )
    metrics["known_reference_parasite_db"] = known_reference_parasite_db(
        prediction, target
    )
    delay = estimate_delay(target, prediction, max_lag=32)
    metrics["estimated_delay_samples"] = delay.total_samples
    metrics["alignment_peak"] = delay.normalized_peak
    return metrics


def detection_checks(results: dict[str, dict[str, float]]) -> dict[str, bool]:
    """Apply broad, preregistered sanity thresholds to each intended response."""
    return {
        "identity_near_zero": results["identity"]["esr"] < 1.0e-15,
        "gain_detected": abs(results["gain_error"]["gain_error"] + 0.2) < 1.0e-3,
        "integer_delay_detected": abs(
            results["integer_delay"]["estimated_delay_samples"] - 8.0
        )
        < 0.1,
        "fractional_delay_detected": abs(
            results["fractional_delay"]["estimated_delay_samples"] - 0.35
        )
        < 0.12,
        "polarity_detected": results["polarity_inversion"]["correlation"] < -0.99,
        "dc_detected": abs(results["dc_offset"]["dc_error"] - 0.05) < 1.0e-3,
        "lowpass_detected": results["lowpass"]["magnitude_error"] > 0.1,
        "noise_detected": results["added_noise"]["inharmonic_energy_ratio"]
        > results["identity"]["inharmonic_energy_ratio"] + 1.0e-4,
        "harmonic_loss_detected": results["harmonic_suppression"][
            "complex_harmonic_error"
        ]
        > 0.05,
        "inharmonic_addition_detected": results["inharmonic_addition"][
            "inharmonic_energy_ratio"
        ]
        > 1.0e-3,
        "controlled_alias_detected": results["controlled_aliasing"][
            "known_reference_parasite_db"
        ]
        > -40.0,
        "oversampling_reduces_controlled_alias": results["controlled_aliasing_x2"][
            "known_reference_parasite_db"
        ]
        < results["controlled_aliasing"]["known_reference_parasite_db"] - 3.0,
        "ringing_detected": results["ringing"]["mrstft"] > 0.1,
        "phase_without_magnitude_detected": results["phase_only"]["phase_error_radians"]
        > 0.02
        and results["phase_only"]["magnitude_error"] < 1.0e-5,
    }


def validate_metric_suite() -> dict[str, object]:
    results = {
        name: evaluate_perturbation(perturbation)
        for name, perturbation in build_perturbations().items()
    }
    checks = detection_checks(results)
    return {
        "schema_version": 1,
        "sample_rate": SAMPLE_RATE,
        "sample_count": SAMPLE_COUNT,
        "results": results,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "aliasing_qualification": (
            "Only controlled_aliasing and controlled_aliasing_x2 use that label "
            "because their target is a 192 kHz nonlinear reference decimated "
            "through the documented FIR."
        ),
    }
