"""Executable synthetic mechanism qualification for FSSR-R2-48K.

This module turns the frozen analytic fixtures into the four preregistered AA
routes.  It deliberately does not read physical audio: the 192 kHz signal is a
directly generated synthetic reference, never an upsampled hardware target.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import lfilter

from fssr_nam.data.r2_fixtures import (
    R2_FIXTURES,
    apply_r2_fixture,
    residual_energy_ratio,
)
from fssr_nam.dsp.multirate import derive_reference_rates
from fssr_nam.metrics.nonlinear import (
    complex_harmonic_error,
    known_reference_parasite_db,
)
from fssr_nam.models.oversampling import design_resampling_lowpass

from .r2_aliasing import (
    AMPLITUDES,
    DFT_SAMPLES,
    FRAMES,
    K0_VALUES,
    SAMPLE_RATE_HZ,
    aggregate_asr_grid,
    anti_silence_guard,
    coherent_sine_probe,
    sato_smith_asr,
)

MECHANISM_MODES = ("off", "full_island_x2", "adaa1", "teacher_x4")
MODE_FACTORS = {
    "off": 1,
    "full_island_x2": 2,
    "adaa1": 1,
    "teacher_x4": 4,
}
MODE_LATENCY_SAMPLES = {
    "off": 0,
    "full_island_x2": 16,
    "adaa1": 1,
    "teacher_x4": 16,
}
REFERENCE_RATE_HZ = 192_000
REFERENCE_TAIL_FRAMES = 1
RESIDUAL_GUARD_SEED = 20_260_828


def _mono_float32(signal: ArrayLike) -> NDArray[np.float32]:
    samples = np.asarray(signal, dtype=np.float32)
    if samples.ndim != 1 or samples.size < 1 or not np.isfinite(samples).all():
        raise ValueError("mechanism renderer requires finite non-empty mono input")
    return samples


def _delay(signal: NDArray[np.float32], samples: int) -> NDArray[np.float32]:
    if samples < 0:
        raise ValueError("alignment delay cannot be negative")
    if samples == 0:
        return signal.copy()
    delayed = np.zeros_like(signal)
    if samples < signal.size:
        delayed[samples:] = signal[:-samples]
    return delayed


def _full_rate_island(
    fixture: str, signal: NDArray[np.float32], factor: int
) -> NDArray[np.float32]:
    """Offline float32 equivalent of the frozen causal ``FullRateIsland``."""
    if factor not in {2, 4}:
        raise ValueError("mechanism full-rate factor must be two or four")
    latency_samples = 16
    taps = factor * latency_samples + 1
    lowpass = (
        design_resampling_lowpass(factor, taps)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    high_rate = np.zeros(factor * signal.size, dtype=np.float32)
    high_rate[::factor] = signal
    denominator = np.ones(1, dtype=np.float32)
    interpolated = np.asarray(
        lfilter(float(factor) * lowpass, denominator, high_rate), dtype=np.float32
    )
    branch = apply_r2_fixture(fixture, interpolated, SAMPLE_RATE_HZ * factor).output
    nonlinear_residual = np.asarray(branch - interpolated, dtype=np.float32)
    filtered = np.asarray(
        lfilter(lowpass, denominator, nonlinear_residual), dtype=np.float32
    )
    residual = filtered[::factor]
    return np.asarray(_delay(signal, latency_samples) + residual, dtype=np.float32)


def render_mechanism_mode(
    fixture: str, signal: ArrayLike, mode: str
) -> NDArray[np.float32]:
    """Render one frozen fixture/mode pair without fitting or random state."""
    samples = _mono_float32(signal)
    if fixture not in R2_FIXTURES:
        raise ValueError(f"unknown R2 mechanism fixture: {fixture}")
    if mode == "off":
        return apply_r2_fixture(fixture, samples, SAMPLE_RATE_HZ).output
    if mode == "adaa1":
        return apply_r2_fixture(fixture, samples, SAMPLE_RATE_HZ, adaa=True).output
    if mode == "full_island_x2":
        return _full_rate_island(fixture, samples, 2)
    if mode == "teacher_x4":
        return _full_rate_island(fixture, samples, 4)
    raise ValueError(f"unknown R2 mechanism mode: {mode}")


def synthetic_192khz_reference(
    fixture: str, k0: int, amplitude: float
) -> NDArray[np.float32]:
    """Generate and downsample a direct 192 kHz analytic fixture reference."""
    if k0 not in K0_VALUES or amplitude not in AMPLITUDES:
        raise ValueError("reference probe is outside the frozen 3x3 grid")
    factor = REFERENCE_RATE_HZ // SAMPLE_RATE_HZ
    high_frame_samples = factor * DFT_SAMPLES
    index = np.arange(high_frame_samples, dtype=np.float64)
    frame = amplitude * np.sin(2.0 * np.pi * k0 * index / high_frame_samples)
    high_rate_input = np.tile(frame, FRAMES + REFERENCE_TAIL_FRAMES).astype(np.float32)
    high_rate_output = apply_r2_fixture(
        fixture, high_rate_input, REFERENCE_RATE_HZ
    ).output
    reference = derive_reference_rates(high_rate_output)[SAMPLE_RATE_HZ]
    expected = FRAMES * DFT_SAMPLES
    if reference.size < expected:
        raise RuntimeError("synthetic reference decimation returned too few samples")
    return np.asarray(reference[:expected], dtype=np.float32)


def _condition_result(
    *,
    fixture: str,
    mode: str,
    k0: int,
    amplitude: float,
    signal: NDArray[np.float32],
    aa_off: NDArray[np.float32],
    reference: NDArray[np.float32],
) -> dict[str, Any]:
    candidate = render_mechanism_mode(fixture, signal, mode)
    latency = MODE_LATENCY_SAMPLES[mode]
    aligned_off = _delay(aa_off, latency)
    aligned_reference = _delay(reference, latency)
    asr = sato_smith_asr(candidate, k0=k0)
    reference_asr = sato_smith_asr(aligned_reference, k0=k0)
    guard = anti_silence_guard(candidate, aligned_off, aligned_reference, k0=k0)
    candidate_frame = candidate[-DFT_SAMPLES:]
    reference_frame = aligned_reference[-DFT_SAMPLES:]
    fundamental_hz = k0 * SAMPLE_RATE_HZ / DFT_SAMPLES
    fundamental_error = complex_harmonic_error(
        candidate_frame,
        reference_frame,
        fundamental_hz=fundamental_hz,
        sample_rate=SAMPLE_RATE_HZ,
        maximum_harmonic=1,
    )
    parasite_db = known_reference_parasite_db(candidate_frame, reference_frame)
    values = (
        float(asr["asr_db"]),
        float(reference_asr["asr_db"]),
        float(fundamental_error),
        float(parasite_db),
    )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(
            f"non-finite mechanism metric for {fixture}/{mode}/{k0}/{amplitude}"
        )
    return {
        "k0": k0,
        "amplitude": amplitude,
        "frequency_hz": fundamental_hz,
        "asr_db": values[0],
        "reference_192khz_asr_db": values[3],
        "reference_output_asr_db": values[1],
        "periodicity_error_db": float(asr["periodicity_error_db"]),
        "reference_periodicity_error_db": float(reference_asr["periodicity_error_db"]),
        "fundamental_complex_error": values[2],
        "guard": guard,
    }


def _rf2047_residual_guard() -> float:
    generator = np.random.default_rng(RESIDUAL_GUARD_SEED)
    signal = generator.normal(0.0, 0.25, 32_768).astype(np.float32)
    return residual_energy_ratio(
        apply_r2_fixture("rf2047_residual", signal, SAMPLE_RATE_HZ)
    )


def qualify_synthetic_mechanism(
    *,
    fixtures: Sequence[str] = R2_FIXTURES,
    modes: Sequence[str] = MECHANISM_MODES,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Execute the frozen six-fixture/four-route mechanism matrix."""
    if tuple(fixtures) != R2_FIXTURES:
        raise ValueError("mechanism fixture matrix must equal the frozen order")
    if tuple(modes) != MECHANISM_MODES:
        raise ValueError("mechanism mode matrix must equal the frozen order")
    rows: list[dict[str, Any]] = []
    residual_ratio = _rf2047_residual_guard()
    for fixture in fixtures:
        if progress is not None:
            progress(f"fixture_started:{fixture}")
        condition_cache: dict[
            tuple[int, float],
            tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]],
        ] = {}
        for k0 in K0_VALUES:
            for amplitude in AMPLITUDES:
                signal = coherent_sine_probe(k0, amplitude)
                aa_off = render_mechanism_mode(fixture, signal, "off")
                reference = synthetic_192khz_reference(fixture, k0, amplitude)
                condition_cache[(k0, amplitude)] = (signal, aa_off, reference)
        for mode in modes:
            conditions = [
                _condition_result(
                    fixture=fixture,
                    mode=mode,
                    k0=k0,
                    amplitude=amplitude,
                    signal=condition_cache[(k0, amplitude)][0],
                    aa_off=condition_cache[(k0, amplitude)][1],
                    reference=condition_cache[(k0, amplitude)][2],
                )
                for k0 in K0_VALUES
                for amplitude in AMPLITUDES
            ]
            asr = aggregate_asr_grid(conditions)
            reference_asr = aggregate_asr_grid(
                [
                    {
                        "k0": condition["k0"],
                        "amplitude": condition["amplitude"],
                        "asr_db": condition["reference_192khz_asr_db"],
                    }
                    for condition in conditions
                ]
            )
            rows.append(
                {
                    "fixture": fixture,
                    "mode": mode,
                    "weights_id": f"r2-analytic-{fixture}-v1",
                    "asr_db": asr["median_asr_db"],
                    "reference_192khz_asr_db": reference_asr["median_asr_db"],
                    "reference_kind": "synthetic_192khz",
                    "physical_hardware_reference_used": False,
                    "guard_passed": all(
                        condition["guard"]["passed"] for condition in conditions
                    ),
                    "fundamental_complex_error": max(
                        condition["fundamental_complex_error"]
                        for condition in conditions
                    ),
                    "latency_samples": MODE_LATENCY_SAMPLES[mode],
                    "internal_sample_rate_hz": SAMPLE_RATE_HZ * MODE_FACTORS[mode],
                    "residual_energy_ratio": (
                        residual_ratio if fixture == "rf2047_residual" else 0.0
                    ),
                    "aggregation": {
                        "asr": "median_exact_3x3_grid",
                        "fundamental_complex_error": "maximum_exact_3x3_grid",
                        "guard": "all_exact_3x3_conditions",
                    },
                    "conditions": conditions,
                }
            )
        if progress is not None:
            progress(f"fixture_completed:{fixture}")
    return {
        "schema_version": 1,
        "campaign_version": "FSSR-R2-48K-v1",
        "evidence_tier": "SYNTHETIC",
        "reference_kind": "synthetic_192khz",
        "reference_generated_directly_at_hz": REFERENCE_RATE_HZ,
        "physical_hardware_reference_used": False,
        "physical_audio_samples_read": 0,
        "fixtures": list(fixtures),
        "modes": list(modes),
        "same_weights_across_modes": True,
        "rows": rows,
        "claim_boundary": (
            "Qualifies only the synthetic AA mechanism; it cannot establish "
            "hardware alias reduction or a global state-of-the-art claim."
        ),
    }
