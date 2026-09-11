"""Frozen Sato--Smith ASR probes and anti-silence guards for FSSR-R2."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .time import correlation, gain_error

SAMPLE_RATE_HZ = 48_000
DFT_SAMPLES = 65_536
K0_VALUES = (1705, 8191, 12287)
AMPLITUDES = (0.10, 0.25, 0.48)
FRAMES = 6
PERIODICITY_ERROR_DB_MAXIMUM = -60.0


class ASREvidenceError(ValueError):
    """Raised when a probe cannot support a valid aliasing measurement."""


def coherent_sine_probe(
    k0: int,
    amplitude: float,
    *,
    dft_samples: int = DFT_SAMPLES,
    frames: int = FRAMES,
) -> NDArray[np.float32]:
    """Generate exact-bin, frame-periodic input for the frozen R2 probe."""
    if k0 not in K0_VALUES:
        raise ValueError(f"k0 must be one of {K0_VALUES}")
    if amplitude not in AMPLITUDES:
        raise ValueError(f"amplitude must be one of {AMPLITUDES}")
    if frames < 2:
        raise ValueError("at least two coherent frames are required")
    frame_index = np.arange(dft_samples, dtype=np.float64)
    frame = amplitude * np.sin(2.0 * np.pi * k0 * frame_index / dft_samples)
    return np.tile(frame.astype(np.float32), frames)


def _signal(samples: ArrayLike, *, expected_samples: int) -> NDArray[np.float64]:
    signal = np.asarray(samples, dtype=np.float64)
    if signal.ndim != 1 or signal.size != expected_samples:
        raise ASREvidenceError(
            f"ASR output must be mono with exactly {expected_samples} samples"
        )
    if not np.isfinite(signal).all():
        raise ASREvidenceError("ASR output must be finite")
    if np.ptp(signal) <= np.finfo(np.float64).eps:
        raise ASREvidenceError("ASR output must be nonconstant")
    return signal


def _db_ratio(numerator: float, denominator: float) -> float:
    if denominator <= np.finfo(np.float64).tiny:
        raise ASREvidenceError("ASR denominator energy must be positive")
    if numerator <= 0.0:
        return -math.inf
    return float(10.0 * math.log10(numerator / denominator))


def sato_smith_asr(
    samples: ArrayLike,
    *,
    k0: int,
    dft_samples: int = DFT_SAMPLES,
    frames: int = FRAMES,
    periodicity_error_db_maximum: float = PERIODICITY_ERROR_DB_MAXIMUM,
) -> dict[str, Any]:
    """Measure one no-window/no-padding R2 ASR frame after a periodicity gate."""
    if k0 < 1 or k0 >= dft_samples // 2 or math.gcd(k0, dft_samples) != 1:
        raise ValueError("k0 must be coprime to N and below Nyquist")
    signal = _signal(samples, expected_samples=dft_samples * frames)
    previous = signal[-2 * dft_samples : -dft_samples]
    analyzed = signal[-dft_samples:]
    periodicity_error_db = _db_ratio(
        float(np.sum(np.square(analyzed - previous), dtype=np.float64)),
        float(np.sum(np.square(analyzed), dtype=np.float64)),
    )
    if periodicity_error_db > periodicity_error_db_maximum:
        raise ASREvidenceError(
            "ASR frame periodicity failed: "
            f"{periodicity_error_db:.3f} dB > {periodicity_error_db_maximum:.3f} dB"
        )
    spectrum = np.fft.rfft(analyzed)
    total_energy = float(np.sum(np.square(np.abs(spectrum)), dtype=np.float64))
    harmonic_bins = np.arange(k0, dft_samples // 2 + 1, k0, dtype=np.int64)
    harmonic_values = spectrum[harmonic_bins]
    harmonic_energy = float(
        np.sum(np.square(np.abs(harmonic_values)), dtype=np.float64)
    )
    if harmonic_energy <= np.finfo(np.float64).tiny:
        raise ASREvidenceError("ASR harmonic energy must be positive")
    alias_energy = max(0.0, total_energy - harmonic_energy)
    asr_linear = alias_energy / harmonic_energy
    return {
        "method": "sato-smith-r2-v1",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "dft_samples": dft_samples,
        "k0": k0,
        "window": "none",
        "zero_padding": False,
        "frames_received": frames,
        "analyzed_frame": frames,
        "periodicity_error_db": periodicity_error_db,
        "periodicity_passed": True,
        "total_energy": total_energy,
        "harmonic_energy": harmonic_energy,
        "alias_energy": alias_energy,
        "asr_linear": asr_linear,
        "asr_db": _db_ratio(alias_energy, harmonic_energy),
        "fundamental": {
            "real": float(spectrum[k0].real),
            "imag": float(spectrum[k0].imag),
        },
        "harmonic_bins": harmonic_bins.tolist(),
    }


def _last_frame(samples: ArrayLike, dft_samples: int) -> NDArray[np.float64]:
    signal = np.asarray(samples, dtype=np.float64)
    if signal.ndim != 1 or signal.size < dft_samples:
        raise ASREvidenceError("guard signals must contain at least one DFT frame")
    frame = signal[-dft_samples:]
    if not np.isfinite(frame).all() or np.ptp(frame) <= np.finfo(float).eps:
        raise ASREvidenceError("guard output must be finite and nonconstant")
    return frame


def anti_silence_guard(
    candidate: ArrayLike,
    aa_off: ArrayLike,
    harmonic_reference: ArrayLike,
    *,
    k0: int,
    dft_samples: int = DFT_SAMPLES,
) -> dict[str, Any]:
    """Reject ASR improvements obtained by attenuation or harmonic damage."""
    candidate_frame = _last_frame(candidate, dft_samples)
    off_frame = _last_frame(aa_off, dft_samples)
    reference_frame = _last_frame(harmonic_reference, dft_samples)
    if (
        candidate_frame.shape != off_frame.shape
        or off_frame.shape != reference_frame.shape
    ):
        raise ASREvidenceError("ASR guard signals must have identical shapes")
    candidate_spectrum = np.fft.rfft(candidate_frame)
    off_spectrum = np.fft.rfft(off_frame)
    reference_spectrum = np.fft.rfft(reference_frame)
    harmonic_bins = np.arange(k0, dft_samples // 2 + 1, k0, dtype=np.int64)
    candidate_harmonics = candidate_spectrum[harmonic_bins]
    off_harmonics = off_spectrum[harmonic_bins]
    reference_harmonics = reference_spectrum[harmonic_bins]
    candidate_harmonic_energy = float(np.sum(np.abs(candidate_harmonics) ** 2))
    off_harmonic_energy = float(np.sum(np.abs(off_harmonics) ** 2))
    fundamental_delta_db = _db_ratio(
        float(abs(candidate_spectrum[k0]) ** 2),
        float(abs(off_spectrum[k0]) ** 2),
    )
    harmonic_delta_db = _db_ratio(candidate_harmonic_energy, off_harmonic_energy)
    reference_energy = float(np.sum(np.abs(reference_harmonics) ** 2))
    if reference_energy <= np.finfo(float).tiny:
        raise ASREvidenceError("harmonic reference energy must be positive")
    off_complex_error = float(
        np.sum(np.abs(off_harmonics - reference_harmonics) ** 2) / reference_energy
    )
    candidate_complex_error = float(
        np.sum(np.abs(candidate_harmonics - reference_harmonics) ** 2)
        / reference_energy
    )
    complex_ratio = (
        candidate_complex_error / off_complex_error
        if off_complex_error > np.finfo(float).tiny
        else (1.0 if candidate_complex_error <= np.finfo(float).tiny else math.inf)
    )
    values = {
        "gain_error": gain_error(candidate_frame, off_frame),
        "correlation": correlation(candidate_frame, off_frame),
        "fundamental_delta_db": fundamental_delta_db,
        "harmonic_energy_delta_db": harmonic_delta_db,
        "complex_harmonic_error_ratio": complex_ratio,
    }
    checks = {
        "gain_error": values["gain_error"] > -0.2,
        "correlation": values["correlation"] > 0.9,
        "fundamental_level": abs(fundamental_delta_db) <= 0.5,
        "harmonic_energy": abs(harmonic_delta_db) <= 0.5,
        "complex_harmonic_error": complex_ratio <= 1.05,
    }
    return {"passed": all(checks.values()), "values": values, "checks": checks}


def aggregate_asr_grid(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Validate and summarize the exact 3x3 k0/amplitude probe grid."""
    expected = {(k0, amplitude) for k0 in K0_VALUES for amplitude in AMPLITUDES}
    observed: dict[tuple[int, float], float] = {}
    for row in rows:
        key = (int(row.get("k0", -1)), float(row.get("amplitude", math.nan)))
        if key in observed:
            raise ASREvidenceError(f"duplicate ASR grid condition: {key}")
        value = float(row.get("asr_db", math.nan))
        if not math.isfinite(value):
            raise ASREvidenceError("ASR grid values must be finite dB measurements")
        observed[key] = value
    if set(observed) != expected:
        raise ASREvidenceError(
            f"ASR grid mismatch; missing={sorted(expected - set(observed))}, "
            f"extra={sorted(set(observed) - expected)}"
        )
    values = np.asarray(list(observed.values()), dtype=np.float64)
    return {
        "conditions": len(observed),
        "median_asr_db": float(np.median(values)),
        "minimum_asr_db": float(np.min(values)),
        "maximum_asr_db": float(np.max(values)),
    }
