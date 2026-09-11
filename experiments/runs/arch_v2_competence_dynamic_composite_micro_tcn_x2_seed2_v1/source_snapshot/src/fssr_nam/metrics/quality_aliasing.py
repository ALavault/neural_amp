"""Prospective DC-separated aliasing metrics for QUALITY-AA-v1.

Unlike the frozen R2 metric, DC is not treated as alias energy.  All decision
statistics stay in the linear domain and only the reported dB value is
censored at a floor calibrated on identity probes before candidate rendering.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fssr_nam.metrics.time import correlation, gain_error

SAMPLE_RATE_HZ = 48_000
DFT_SAMPLES = 65_536
K0_VALUES = (1705, 8191, 12287)
AMPLITUDES = (0.10, 0.25, 0.48)
FRAMES = 6
PERIODICITY_ERROR_DB_MAXIMUM = -60.0


class QualityAliasingEvidenceError(ValueError):
    """Raised when evidence cannot satisfy the prospective metric contract."""


def coherent_sine_probe(
    k0: int,
    amplitude: float,
    *,
    dft_samples: int = DFT_SAMPLES,
    frames: int = FRAMES,
) -> NDArray[np.float32]:
    """Generate the exact coherent probe grid used by QUALITY-AA-v1."""
    if k0 not in K0_VALUES:
        raise ValueError(f"k0 must be one of {K0_VALUES}")
    if amplitude not in AMPLITUDES:
        raise ValueError(f"amplitude must be one of {AMPLITUDES}")
    if frames < 2:
        raise ValueError("at least two coherent frames are required")
    index = np.arange(dft_samples, dtype=np.float64)
    frame = amplitude * np.sin(2.0 * np.pi * k0 * index / dft_samples)
    return np.tile(frame.astype(np.float32), frames)


def _signal(samples: ArrayLike, *, expected_samples: int) -> NDArray[np.float64]:
    signal = np.asarray(samples, dtype=np.float64)
    if signal.ndim != 1 or signal.size != expected_samples:
        raise QualityAliasingEvidenceError(
            f"signal must be mono with exactly {expected_samples} samples"
        )
    if not np.isfinite(signal).all():
        raise QualityAliasingEvidenceError("signal must be finite")
    if np.ptp(signal) <= np.finfo(np.float64).eps:
        raise QualityAliasingEvidenceError("signal must be nonconstant")
    return signal


def _db_ratio(numerator: float, denominator: float) -> float:
    if denominator <= np.finfo(np.float64).tiny:
        raise QualityAliasingEvidenceError("ratio denominator must be positive")
    if numerator <= 0.0:
        return -math.inf
    return float(10.0 * math.log10(numerator / denominator))


def _spectral_partition(
    frame: NDArray[np.float64], k0: int
) -> tuple[NDArray[np.complex128], NDArray[np.int64], NDArray[np.bool_]]:
    spectrum = np.fft.rfft(frame)
    harmonic_bins = np.arange(k0, spectrum.size, k0, dtype=np.int64)
    alias_mask = np.ones(spectrum.size, dtype=bool)
    alias_mask[0] = False
    alias_mask[harmonic_bins] = False
    return spectrum, harmonic_bins, alias_mask


def dc_separated_asr(
    samples: ArrayLike,
    *,
    k0: int,
    floor_db: float,
    dft_samples: int = DFT_SAMPLES,
    frames: int = FRAMES,
    periodicity_error_db_maximum: float = PERIODICITY_ERROR_DB_MAXIMUM,
) -> dict[str, Any]:
    """Measure explicit excluded-bin energy while retaining DC separately."""
    if k0 < 1 or k0 >= dft_samples // 2 or math.gcd(k0, dft_samples) != 1:
        raise ValueError("k0 must be coprime to N and below Nyquist")
    if not math.isfinite(floor_db) or floor_db >= 0.0:
        raise ValueError("floor_db must be finite and negative")
    signal = _signal(samples, expected_samples=dft_samples * frames)
    previous = signal[-2 * dft_samples : -dft_samples]
    analyzed = signal[-dft_samples:]
    periodicity_error_db = _db_ratio(
        float(np.sum(np.square(analyzed - previous), dtype=np.float64)),
        float(np.sum(np.square(analyzed), dtype=np.float64)),
    )
    if periodicity_error_db > periodicity_error_db_maximum:
        raise QualityAliasingEvidenceError(
            "frame periodicity failed: "
            f"{periodicity_error_db:.3f} dB > "
            f"{periodicity_error_db_maximum:.3f} dB"
        )
    spectrum, harmonic_bins, alias_mask = _spectral_partition(analyzed, k0)
    harmonic_energy = float(
        np.sum(np.square(np.abs(spectrum[harmonic_bins])), dtype=np.float64)
    )
    if harmonic_energy <= np.finfo(np.float64).tiny:
        raise QualityAliasingEvidenceError("harmonic energy must be positive")
    alias_energy = float(
        np.sum(np.square(np.abs(spectrum[alias_mask])), dtype=np.float64)
    )
    dc_energy = float(abs(spectrum[0]) ** 2)
    asr_linear = alias_energy / harmonic_energy
    floor_linear = 10.0 ** (floor_db / 10.0)
    asr_db_raw = _db_ratio(alias_energy, harmonic_energy)
    return {
        "method": "dc-separated-explicit-bin-asr-v1",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "dft_samples": dft_samples,
        "k0": k0,
        "window": "none",
        "zero_padding": False,
        "periodicity_error_db": periodicity_error_db,
        "periodicity_passed": True,
        "harmonic_energy": harmonic_energy,
        "alias_energy": alias_energy,
        "dc_energy": dc_energy,
        "dc_to_harmonic_linear": dc_energy / harmonic_energy,
        "asr_linear": asr_linear,
        "asr_db_raw": asr_db_raw,
        "asr_db": max(asr_db_raw, floor_db),
        "floor_db": floor_db,
        "floor_censored": asr_linear <= floor_linear,
        "fundamental": {
            "real": float(spectrum[k0].real),
            "imag": float(spectrum[k0].imag),
        },
        "dc": {"real": float(spectrum[0].real), "imag": float(spectrum[0].imag)},
        "harmonic_bins": harmonic_bins.tolist(),
        "alias_bin_count": int(np.count_nonzero(alias_mask)),
    }


def calibrate_identity_floor(
    *, margin_db: float = 12.0, maximum_allowed_floor_db: float = -120.0
) -> dict[str, Any]:
    """Calibrate a reporting floor without rendering an AA candidate."""
    if margin_db <= 0.0:
        raise ValueError("identity floor margin must be positive")
    raw_values = []
    conditions = []
    for k0 in K0_VALUES:
        for amplitude in AMPLITUDES:
            probe = coherent_sine_probe(k0, amplitude)
            result = dc_separated_asr(probe, k0=k0, floor_db=-300.0)
            raw = float(result["asr_db_raw"])
            raw_values.append(raw)
            conditions.append({"k0": k0, "amplitude": amplitude, "asr_db_raw": raw})
    finite = [value for value in raw_values if math.isfinite(value)]
    worst_identity_db = max(finite) if finite else -300.0
    locked_floor_db = worst_identity_db + margin_db
    passed = locked_floor_db <= maximum_allowed_floor_db
    return {
        "method": "float32-identity-grid-plus-fixed-margin-v1",
        "margin_db": margin_db,
        "worst_identity_asr_db": worst_identity_db,
        "locked_floor_db": locked_floor_db,
        "maximum_allowed_floor_db": maximum_allowed_floor_db,
        "passed": passed,
        "candidate_outputs_observed": False,
        "conditions": conditions,
    }


def _last_frame(samples: ArrayLike, dft_samples: int) -> NDArray[np.float64]:
    signal = np.asarray(samples, dtype=np.float64)
    if signal.ndim != 1 or signal.size < dft_samples:
        raise QualityAliasingEvidenceError("guard signal lacks one complete frame")
    frame = signal[-dft_samples:]
    if not np.isfinite(frame).all() or np.ptp(frame) <= np.finfo(float).eps:
        raise QualityAliasingEvidenceError(
            "guard signal must be finite and nonconstant"
        )
    return frame


def harmonic_fidelity_guard(
    candidate: ArrayLike,
    aa_off: ArrayLike,
    reference: ArrayLike,
    *,
    k0: int,
    dft_samples: int = DFT_SAMPLES,
    absolute_complex_error_floor: float = 1.0e-5,
    maximum_complex_error_regression_ratio: float = 1.05,
) -> dict[str, Any]:
    """Guard useful harmonics and DC with nonsingular absolute+relative checks."""
    candidate_frame = _last_frame(candidate, dft_samples)
    off_frame = _last_frame(aa_off, dft_samples)
    reference_frame = _last_frame(reference, dft_samples)
    candidate_spectrum, harmonic_bins, _ = _spectral_partition(candidate_frame, k0)
    off_spectrum, _, _ = _spectral_partition(off_frame, k0)
    reference_spectrum, _, _ = _spectral_partition(reference_frame, k0)
    candidate_harmonics = candidate_spectrum[harmonic_bins]
    off_harmonics = off_spectrum[harmonic_bins]
    reference_harmonics = reference_spectrum[harmonic_bins]
    reference_energy = float(np.sum(np.abs(reference_harmonics) ** 2))
    if reference_energy <= np.finfo(float).tiny:
        raise QualityAliasingEvidenceError("reference harmonic energy must be positive")

    def normalized_error(values: NDArray, target: NDArray) -> float:
        return float(np.sum(np.abs(values - target) ** 2) / reference_energy)

    candidate_complex_error = normalized_error(candidate_harmonics, reference_harmonics)
    off_complex_error = normalized_error(off_harmonics, reference_harmonics)
    candidate_fundamental_error = float(
        abs(candidate_spectrum[k0] - reference_spectrum[k0]) ** 2
        / (abs(reference_spectrum[k0]) ** 2 + np.finfo(float).tiny)
    )
    candidate_dc_error = float(
        abs(candidate_spectrum[0] - reference_spectrum[0]) ** 2 / reference_energy
    )
    candidate_harmonic_energy = float(np.sum(np.abs(candidate_harmonics) ** 2))
    reference_fundamental_energy = float(abs(reference_spectrum[k0]) ** 2)
    fundamental_delta_db = _db_ratio(
        float(abs(candidate_spectrum[k0]) ** 2), reference_fundamental_energy
    )
    harmonic_delta_db = _db_ratio(candidate_harmonic_energy, reference_energy)
    relative_applicable = off_complex_error > absolute_complex_error_floor
    relative_limit = maximum_complex_error_regression_ratio * off_complex_error
    values = {
        "gain_error": gain_error(candidate_frame, reference_frame),
        "correlation": correlation(candidate_frame, reference_frame),
        "fundamental_delta_db": fundamental_delta_db,
        "harmonic_energy_delta_db": harmonic_delta_db,
        "candidate_complex_harmonic_error": candidate_complex_error,
        "off_complex_harmonic_error": off_complex_error,
        "complex_relative_limit": relative_limit,
        "relative_check_applicable": relative_applicable,
        "fundamental_complex_error": candidate_fundamental_error,
        "dc_complex_error": candidate_dc_error,
    }
    checks = {
        "gain_error": values["gain_error"] > -0.2,
        "correlation": values["correlation"] > 0.9,
        "fundamental_level": abs(fundamental_delta_db) <= 0.5,
        "harmonic_energy": abs(harmonic_delta_db) <= 0.5,
        "complex_absolute": candidate_complex_error <= absolute_complex_error_floor,
        "complex_relative": (
            not relative_applicable or candidate_complex_error <= relative_limit
        ),
        "fundamental_complex": candidate_fundamental_error <= 1.0e-5,
        "dc_complex": candidate_dc_error <= 1.0e-5,
    }
    return {"passed": all(checks.values()), "values": values, "checks": checks}


def known_reference_alias_residual(
    candidate: ArrayLike,
    reference: ArrayLike,
    *,
    k0: int,
    floor_db: float,
    dft_samples: int = DFT_SAMPLES,
) -> dict[str, Any]:
    """Measure residual energy only on bins classified as alias by the metric."""
    candidate_frame = _last_frame(candidate, dft_samples)
    reference_frame = _last_frame(reference, dft_samples)
    candidate_spectrum, harmonic_bins, alias_mask = _spectral_partition(
        candidate_frame, k0
    )
    reference_spectrum, _, _ = _spectral_partition(reference_frame, k0)
    reference_energy = float(np.sum(np.abs(reference_spectrum[harmonic_bins]) ** 2))
    residual = candidate_spectrum[alias_mask] - reference_spectrum[alias_mask]
    residual_energy = float(np.sum(np.abs(residual) ** 2))
    raw_db = _db_ratio(residual_energy, reference_energy)
    return {
        "linear": residual_energy / reference_energy,
        "db_raw": raw_db,
        "db": max(raw_db, floor_db),
        "floor_censored": raw_db <= floor_db,
    }


def aggregate_grid(rows: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    """Validate and aggregate the exact 3x3 grid for one numeric field."""
    expected = {(k0, amplitude) for k0 in K0_VALUES for amplitude in AMPLITUDES}
    observed: dict[tuple[int, float], float] = {}
    for row in rows:
        key = (int(row.get("k0", -1)), float(row.get("amplitude", math.nan)))
        if key in observed:
            raise QualityAliasingEvidenceError(f"duplicate grid condition: {key}")
        value = float(row.get(field, math.nan))
        if not math.isfinite(value):
            raise QualityAliasingEvidenceError(
                f"{field} must be finite after censoring"
            )
        observed[key] = value
    if set(observed) != expected:
        raise QualityAliasingEvidenceError("grid does not match the frozen 3x3 design")
    values = np.asarray(list(observed.values()), dtype=np.float64)
    return {
        "conditions": len(observed),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def fundamental_delay_samples(
    candidate: ArrayLike,
    reference: ArrayLike,
    *,
    k0: int,
    dft_samples: int = DFT_SAMPLES,
) -> float:
    """Infer a sub-sample delay from the coherent fundamental phase."""
    candidate_frame = _last_frame(candidate, dft_samples)
    reference_frame = _last_frame(reference, dft_samples)
    candidate_bin = np.fft.rfft(candidate_frame)[k0]
    reference_bin = np.fft.rfft(reference_frame)[k0]
    if (
        abs(candidate_bin) <= np.finfo(float).tiny
        or abs(reference_bin) <= np.finfo(float).tiny
    ):
        raise QualityAliasingEvidenceError("delay probe fundamental is absent")
    phase = float(np.angle(candidate_bin / reference_bin))
    angular_frequency = 2.0 * np.pi * k0 / dft_samples
    return -phase / angular_frequency
