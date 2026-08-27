"""Manifest-only construction of the deterministic M1 synthetic corpus."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from fssr_nam.data.excitations import EXCITATIONS, generate_excitation
from fssr_nam.data.systems import SYSTEMS, apply_system
from fssr_nam.dsp.multirate import derive_reference_rates


def _sha256_float32(signal: np.ndarray) -> str:
    canonical = np.asarray(signal, dtype="<f4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _summary(signal: np.ndarray) -> dict[str, Any]:
    samples = np.asarray(signal, dtype=np.float32)
    if samples.ndim != 1 or not np.all(np.isfinite(samples)):
        raise RuntimeError("synthetic corpus contains invalid samples")
    return {
        "sample_count": int(samples.size),
        "dtype": "float32",
        "peak": float(np.max(np.abs(samples), initial=0.0)),
        "rms": float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))),
        "sha256": _sha256_float32(samples),
    }


def build_corpus_manifest(
    config: Mapping[str, Any], *, external_di_48k: np.ndarray | None = None
) -> dict[str, Any]:
    """Generate every configured pair and retain reproducible numeric summaries."""
    master_rate = int(config["master_sample_rate"])
    if master_rate != 192_000:
        raise ValueError("M1 master_sample_rate must be 192000")
    duration = float(config["duration_seconds"])
    seed = int(config["seed"])
    excitation_names = list(config["excitations"])
    system_names = list(config["systems"])
    unknown_excitations = sorted(set(excitation_names) - set(EXCITATIONS))
    unknown_systems = sorted(set(system_names) - set(SYSTEMS))
    if unknown_excitations or unknown_systems:
        raise ValueError(
            f"unknown excitations={unknown_excitations}, systems={unknown_systems}"
        )

    excitation_entries: dict[str, Any] = {}
    target_entries: dict[str, Any] = {}
    expected_counts = {
        192_000: round(duration * 192_000),
        96_000: round(duration * 96_000),
        48_000: round(duration * 48_000),
    }
    for excitation_name in excitation_names:
        master_input = generate_excitation(
            excitation_name,
            sample_rate=master_rate,
            duration_seconds=duration,
            seed=seed,
        )
        inputs_by_rate = derive_reference_rates(master_input)
        for rate, samples in inputs_by_rate.items():
            if samples.size != expected_counts[rate]:
                raise RuntimeError(f"unexpected {rate} Hz input length")
        excitation_entries[excitation_name] = {
            str(rate): _summary(samples) for rate, samples in inputs_by_rate.items()
        }

        for system_name in system_names:
            master_target = apply_system(system_name, master_input, master_rate)
            targets_by_rate = derive_reference_rates(master_target)
            for rate, samples in targets_by_rate.items():
                if samples.size != expected_counts[rate]:
                    raise RuntimeError(f"unexpected {rate} Hz target length")
            target_entries[f"{excitation_name}/{system_name}"] = {
                str(rate): _summary(samples)
                for rate, samples in targets_by_rate.items()
            }

    manifest = {
        "schema_version": 1,
        "tier": "SYNTHETIC",
        "master_sample_rate": master_rate,
        "derived_sample_rates": [96_000, 48_000],
        "duration_seconds": duration,
        "seed": seed,
        "excitation_count": len(excitation_entries),
        "system_count": len(system_names),
        "pair_count": len(target_entries),
        "excitations": excitation_entries,
        "targets": target_entries,
        "limitations": [
            (
                "procedural_plucks is a rights-free synthetic guitar-like signal, "
                "not a recorded DI"
            ),
            (
                "synthetic systems are diagnostic references, not evidence of "
                "physical-device fidelity"
            ),
        ],
    }
    if external_di_48k is not None:
        external_samples = np.asarray(external_di_48k, dtype=np.float32)
        if external_samples.ndim != 1 or not np.all(np.isfinite(external_samples)):
            raise ValueError("external DI must be a finite mono signal")
        master_external = np.asarray(
            resample_poly(external_samples, 4, 1, window=("kaiser", 8.6)),
            dtype=np.float32,
        )
        external_rates = derive_reference_rates(master_external)
        external_targets = {}
        for system_name in system_names:
            master_target = apply_system(system_name, master_external, master_rate)
            external_targets[system_name] = {
                str(rate): _summary(samples)
                for rate, samples in derive_reference_rates(master_target).items()
            }
        manifest["external_di_diagnostic"] = {
            "source_rate": 48_000,
            "master_interpolation": "scipy.signal.resample_poly up=4, Kaiser beta=8.6",
            "inputs": {
                str(rate): _summary(samples) for rate, samples in external_rates.items()
            },
            "targets": external_targets,
        }
    return manifest
