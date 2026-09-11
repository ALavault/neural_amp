"""Prospective synthetic AA renderer with converged x8/x16 references."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import lfilter, resample_poly

from fssr_nam.data.r2_fixtures import (
    R2_FIXTURES,
    apply_r2_fixture,
    residual_energy_ratio,
)
from fssr_nam.models.oversampling import design_resampling_lowpass

from .quality_aliasing import (
    AMPLITUDES,
    DFT_SAMPLES,
    FRAMES,
    K0_VALUES,
    SAMPLE_RATE_HZ,
    aggregate_grid,
    coherent_sine_probe,
    dc_separated_asr,
    harmonic_fidelity_guard,
    known_reference_alias_residual,
)

MECHANISM_MODES = ("off", "full_island_x2", "teacher_x4")
MODE_FACTORS = {"off": 1, "full_island_x2": 2, "teacher_x4": 4}
MODE_LATENCY_SAMPLES = {"off": 0, "full_island_x2": 32, "teacher_x4": 32}
REFERENCE_FACTORS = (8, 16)
REFERENCE_GROUP_DELAY_BASE_SAMPLES = 64
REFERENCE_KAISER_BETA = 12.0
RESIDUAL_GUARD_SEED = 20_260_828


def _mono_float32(signal: ArrayLike) -> NDArray[np.float32]:
    samples = np.asarray(signal, dtype=np.float32)
    if samples.ndim != 1 or samples.size < 1 or not np.isfinite(samples).all():
        raise ValueError("renderer requires finite non-empty mono input")
    return samples


def _design_reference_lowpass(factor: int) -> NDArray[np.float64]:
    taps = 2 * factor * REFERENCE_GROUP_DELAY_BASE_SAMPLES + 1
    index = np.arange(taps, dtype=np.float64) - (taps - 1) / 2
    cutoff = 0.5 / factor
    impulse = 2.0 * cutoff * np.sinc(2.0 * cutoff * index)
    impulse *= np.kaiser(taps, REFERENCE_KAISER_BETA)
    return impulse / np.sum(impulse)


def delay_signal(signal: ArrayLike, samples: int) -> NDArray[np.float32]:
    source = _mono_float32(signal)
    if samples < 0:
        raise ValueError("alignment delay cannot be negative")
    if samples == 0:
        return source.copy()
    delayed = np.zeros_like(source)
    if samples < source.size:
        delayed[samples:] = source[:-samples]
    return delayed


def full_rate_island(
    fixture: str,
    signal: ArrayLike,
    factor: int,
    *,
    latency_samples: int = 32,
) -> NDArray[np.float32]:
    """Render the deployment FIR island with an exact base-rate delay."""
    samples = _mono_float32(signal)
    if factor not in {2, 4}:
        raise ValueError("full-rate factor must be two or four")
    taps = factor * latency_samples + 1
    lowpass = (
        design_resampling_lowpass(factor, taps)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    high_rate = np.zeros(factor * samples.size, dtype=np.float32)
    high_rate[::factor] = samples
    denominator = np.ones(1, dtype=np.float32)
    interpolated = np.asarray(
        lfilter(float(factor) * lowpass, denominator, high_rate), dtype=np.float32
    )
    branch = apply_r2_fixture(fixture, interpolated, SAMPLE_RATE_HZ * factor).output
    nonlinear_residual = np.asarray(branch - interpolated, dtype=np.float32)
    filtered = np.asarray(
        lfilter(lowpass, denominator, nonlinear_residual), dtype=np.float32
    )
    return np.asarray(
        delay_signal(samples, latency_samples) + filtered[::factor], dtype=np.float32
    )


def render_mechanism_mode(
    fixture: str, signal: ArrayLike, mode: str
) -> NDArray[np.float32]:
    samples = _mono_float32(signal)
    if fixture not in R2_FIXTURES:
        raise ValueError(f"unknown fixture: {fixture}")
    if mode == "off":
        return apply_r2_fixture(fixture, samples, SAMPLE_RATE_HZ).output
    if mode == "full_island_x2":
        return full_rate_island(fixture, samples, 2)
    if mode == "teacher_x4":
        return full_rate_island(fixture, samples, 4)
    raise ValueError(f"unknown mechanism mode: {mode}")


def synthetic_reference(
    fixture: str, k0: int, amplitude: float, factor: int
) -> NDArray[np.float32]:
    """Generate a direct high-rate fixture and zero-phase decimate offline."""
    if factor not in REFERENCE_FACTORS:
        raise ValueError(f"reference factor must be one of {REFERENCE_FACTORS}")
    if k0 not in K0_VALUES or amplitude not in AMPLITUDES:
        raise ValueError("reference condition is outside the frozen grid")
    high_frame_samples = factor * DFT_SAMPLES
    index = np.arange(high_frame_samples, dtype=np.float64)
    frame = amplitude * np.sin(2.0 * np.pi * k0 * index / high_frame_samples)
    high_input = np.tile(frame, FRAMES + 2).astype(np.float32)
    high_output = apply_r2_fixture(fixture, high_input, SAMPLE_RATE_HZ * factor).output
    lowpass = _design_reference_lowpass(factor)
    downsampled = resample_poly(
        np.asarray(high_output, dtype=np.float64),
        up=1,
        down=factor,
        window=lowpass,
        padtype="constant",
    )
    start = DFT_SAMPLES
    stop = start + FRAMES * DFT_SAMPLES
    if downsampled.size < stop:
        raise RuntimeError("reference decimation returned too few samples")
    return np.asarray(downsampled[start:stop], dtype=np.float32)


def reference_pair(
    fixture: str, k0: int, amplitude: float
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    return (
        synthetic_reference(fixture, k0, amplitude, 8),
        synthetic_reference(fixture, k0, amplitude, 16),
    )


def _rf2047_residual_guard() -> float:
    generator = np.random.default_rng(RESIDUAL_GUARD_SEED)
    signal = generator.normal(0.0, 0.25, 32_768).astype(np.float32)
    return residual_energy_ratio(
        apply_r2_fixture("rf2047_residual", signal, SAMPLE_RATE_HZ)
    )


def qualify_synthetic_mechanism(
    *,
    floor_db: float,
    fixtures: Sequence[str] = R2_FIXTURES,
    modes: Sequence[str] = MECHANISM_MODES,
    reference_provider: Callable[[str, int, float], NDArray[np.float32]] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Execute the preregistered matrix after the floor/reference lock."""
    if tuple(fixtures) != R2_FIXTURES:
        raise ValueError("fixture matrix differs from the frozen order")
    if tuple(modes) != MECHANISM_MODES:
        raise ValueError("mode matrix differs from the frozen order")
    rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []
    residual_ratio = _rf2047_residual_guard()
    for fixture in fixtures:
        if progress is not None:
            progress(f"fixture_started:{fixture}")
        cache: dict[tuple[int, float], tuple[NDArray, NDArray, NDArray, NDArray]] = {}
        for k0 in K0_VALUES:
            for amplitude in AMPLITUDES:
                signal = coherent_sine_probe(k0, amplitude)
                off = render_mechanism_mode(fixture, signal, "off")
                if reference_provider is None:
                    reference_x8, reference_x16 = reference_pair(fixture, k0, amplitude)
                    convergence_guard = harmonic_fidelity_guard(
                        reference_x8, reference_x16, reference_x16, k0=k0
                    )
                    convergence_rows.append(
                        {
                            "fixture": fixture,
                            "k0": k0,
                            "amplitude": amplitude,
                            "correlation": convergence_guard["values"]["correlation"],
                            "harmonic_complex_error": convergence_guard["values"][
                                "candidate_complex_harmonic_error"
                            ],
                            "dc_complex_error": convergence_guard["values"][
                                "dc_complex_error"
                            ],
                        }
                    )
                else:
                    reference_x16 = _mono_float32(
                        reference_provider(fixture, k0, amplitude)
                    )
                    reference_x8 = reference_x16
                cache[(k0, amplitude)] = (signal, off, reference_x8, reference_x16)
        for mode in modes:
            conditions = []
            latency = MODE_LATENCY_SAMPLES[mode]
            for k0 in K0_VALUES:
                for amplitude in AMPLITUDES:
                    signal, off, _, reference = cache[(k0, amplitude)]
                    candidate = render_mechanism_mode(fixture, signal, mode)
                    aligned_off = delay_signal(off, latency)
                    aligned_reference = delay_signal(reference, latency)
                    asr = dc_separated_asr(candidate, k0=k0, floor_db=floor_db)
                    off_asr = dc_separated_asr(aligned_off, k0=k0, floor_db=floor_db)
                    guard = harmonic_fidelity_guard(
                        candidate, aligned_off, aligned_reference, k0=k0
                    )
                    residual = known_reference_alias_residual(
                        candidate,
                        aligned_reference,
                        k0=k0,
                        floor_db=floor_db,
                    )
                    conditions.append(
                        {
                            "k0": k0,
                            "amplitude": amplitude,
                            "asr_db": asr["asr_db"],
                            "asr_linear": asr["asr_linear"],
                            "asr_floor_censored": asr["floor_censored"],
                            "off_asr_db": off_asr["asr_db"],
                            "off_asr_floor_censored": off_asr["floor_censored"],
                            "paired_asr_gain_db": off_asr["asr_db"] - asr["asr_db"],
                            "dc_to_harmonic_linear": asr["dc_to_harmonic_linear"],
                            "known_reference_alias_residual_db": residual["db"],
                            "known_reference_alias_residual_linear": residual["linear"],
                            "guard": guard,
                        }
                    )
            rows.append(
                {
                    "fixture": fixture,
                    "mode": mode,
                    "weights_id": f"quality-aa-analytic-{fixture}-v1",
                    "latency_samples": latency,
                    "internal_sample_rate_hz": SAMPLE_RATE_HZ * MODE_FACTORS[mode],
                    "guard_passed": all(item["guard"]["passed"] for item in conditions),
                    "asr": aggregate_grid(conditions, "asr_db"),
                    "paired_asr_gain": aggregate_grid(conditions, "paired_asr_gain_db"),
                    "known_reference_alias_residual": aggregate_grid(
                        conditions, "known_reference_alias_residual_db"
                    ),
                    "residual_energy_ratio": (
                        residual_ratio if fixture == "rf2047_residual" else 0.0
                    ),
                    "conditions": conditions,
                }
            )
        if progress is not None:
            progress(f"fixture_completed:{fixture}")
    return {
        "schema_version": 1,
        "campaign_version": "FSSR-QUALITY-AA-v2",
        "evidence_tier": "SYNTHETIC",
        "reference_kind": "direct_synthetic_x8_x16",
        "selected_reference_factor": 16,
        "physical_hardware_reference_used": False,
        "physical_audio_samples_read": 0,
        "same_weights_across_modes": True,
        "reference_provider": (
            "generated_in_run" if reference_provider is None else "locked_v2_preflight"
        ),
        "floor_db": floor_db,
        "fixtures": list(fixtures),
        "modes": list(modes),
        "reference_convergence": convergence_rows,
        "rows": rows,
        "claim_boundary": (
            "Synthetic AA backend qualification only; no amplifier architecture, "
            "hardware fidelity, or state-of-the-art claim."
        ),
    }
