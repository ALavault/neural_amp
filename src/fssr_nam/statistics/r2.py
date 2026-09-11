"""Preregistered paired hierarchical bootstraps for FSSR-R2-v1."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

CONFIRMATION_DEVICES = ("fulltone", "bigmuff", "blackstar", "ua1176")
CONFIRMATION_SEEDS = (0, 1, 2, 3, 4)


class R2StatisticsError(ValueError):
    """Raised when observations do not match the frozen resampling units."""


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R2StatisticsError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise R2StatisticsError(f"{label} must be finite")
    return result


def _confirmation_groups(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[int, list[tuple[float, float, float]]]]:
    groups: dict[str, dict[int, list[tuple[float, float, float]]]] = {
        device: defaultdict(list) for device in CONFIRMATION_DEVICES
    }
    observed: set[tuple[str, int, str]] = set()
    for row in rows:
        device = row.get("device")
        seed = row.get("seed")
        source = row.get("source")
        if device not in CONFIRMATION_DEVICES:
            raise R2StatisticsError(f"unknown confirmation device: {device!r}")
        if isinstance(seed, bool) or seed not in CONFIRMATION_SEEDS:
            raise R2StatisticsError(f"invalid confirmation seed: {seed!r}")
        if not isinstance(source, str) or not source:
            raise R2StatisticsError("confirmation source must be a non-empty string")
        key = (device, int(seed), source)
        if key in observed:
            raise R2StatisticsError(
                "duplicate device/seed/source observation; windows and probes are "
                "not independent bootstrap units"
            )
        observed.add(key)
        baseline = _finite(row.get("baseline_esr"), "baseline_esr")
        candidate = _finite(row.get("candidate_esr"), "candidate_esr")
        asr_reduction = _finite(row.get("asr_reduction_db"), "asr_reduction_db")
        if baseline <= 0.0 or candidate < 0.0:
            raise R2StatisticsError("ESR observations must have positive baselines")
        groups[device][int(seed)].append((baseline, candidate, asr_reduction))
    for device, seeds in groups.items():
        if set(seeds) != set(CONFIRMATION_SEEDS):
            raise R2StatisticsError(f"{device} confirmation seeds must be exactly 0--4")
        if any(not observations for observations in seeds.values()):
            raise R2StatisticsError(f"{device} has an empty seed/source stratum")
    return groups


def _device_improvement(observations: Sequence[tuple[float, float, float]]) -> float:
    baseline = float(np.mean([row[0] for row in observations]))
    candidate = float(np.mean([row[1] for row in observations]))
    return (baseline - candidate) / baseline


def hierarchical_confirmation_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int = 10_000,
    seed: int = 20_260_828,
) -> dict[str, Any]:
    """Resample paired seeds, then paired sources, with devices as fixed strata."""
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    groups = _confirmation_groups(rows)
    point_device: dict[str, float] = {}
    asr_by_device: dict[str, float] = {}
    for device, seed_groups in groups.items():
        observations = [row for seed_rows in seed_groups.values() for row in seed_rows]
        point_device[device] = _device_improvement(observations)
        asr_values = {row[2] for row in observations}
        if len(asr_values) != 1:
            raise R2StatisticsError(
                f"{device} ASR must be one probe aggregate, not source observations"
            )
        asr_by_device[device] = asr_values.pop()
    point = float(np.mean(list(point_device.values())))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    seed_values = np.asarray(CONFIRMATION_SEEDS, dtype=np.int64)
    for replicate in range(replicates):
        device_improvements = []
        for device in CONFIRMATION_DEVICES:
            sampled_seeds = rng.choice(seed_values, size=len(seed_values), replace=True)
            sampled_observations: list[tuple[float, float, float]] = []
            for sampled_seed in sampled_seeds:
                source_rows = groups[device][int(sampled_seed)]
                indices = rng.integers(0, len(source_rows), size=len(source_rows))
                sampled_observations.extend(
                    source_rows[int(index)] for index in indices
                )
            device_improvements.append(_device_improvement(sampled_observations))
        bootstrap[replicate] = np.mean(device_improvements)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    won = [device for device, improvement in point_device.items() if improvement > 0.0]
    return {
        "method": "paired-hierarchical-r2-confirmation-v1",
        "replicates": replicates,
        "seed": seed,
        "hierarchy": ["seed", "source"],
        "devices_fixed_strata": list(CONFIRMATION_DEVICES),
        "windows_resampled": False,
        "probes_resampled": False,
        "final_conditions": 40,
        "esr_relative_improvement": point,
        "esr_confidence_interval_95": {
            "lower": float(lower),
            "upper": float(upper),
        },
        "device_esr_relative_improvement": point_device,
        "devices_won": won,
        "device_asr_reduction_db": asr_by_device,
        "asr_reduction_db": float(np.median(list(asr_by_device.values()))),
    }


def _mushra_groups(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[set[str], dict[str, list[tuple[str, float]]]]:
    recruited: set[str] = set()
    retained_flags: dict[str, bool] = {}
    retained_rows: dict[str, list[tuple[str, float]]] = defaultdict(list)
    observed: set[tuple[str, str]] = set()
    for row in rows:
        participant = row.get("participant")
        excerpt = row.get("excerpt")
        retained = row.get("retained")
        if not isinstance(participant, str) or not participant:
            raise R2StatisticsError("MUSHRA participant must be a non-empty string")
        if not isinstance(excerpt, str) or not excerpt:
            raise R2StatisticsError("MUSHRA excerpt must be a non-empty string")
        if not isinstance(retained, bool):
            raise R2StatisticsError("MUSHRA retained must be boolean")
        if (
            participant in retained_flags
            and retained_flags[participant] is not retained
        ):
            raise R2StatisticsError("MUSHRA participant retention flag changed")
        retained_flags[participant] = retained
        recruited.add(participant)
        key = (participant, excerpt)
        if key in observed:
            raise R2StatisticsError("duplicate MUSHRA participant/excerpt observation")
        observed.add(key)
        if retained:
            candidate = _finite(row.get("candidate_score"), "candidate_score")
            baseline = _finite(row.get("a2_score"), "a2_score")
            if not 0.0 <= candidate <= 100.0 or not 0.0 <= baseline <= 100.0:
                raise R2StatisticsError("MUSHRA scores must lie in [0, 100]")
            retained_rows[participant].append((excerpt, candidate - baseline))
    if len(recruited) != 24:
        raise R2StatisticsError("MUSHRA requires exactly 24 recruited participants")
    retained_ids = {name for name, retained in retained_flags.items() if retained}
    if len(retained_ids) < 20:
        raise R2StatisticsError("MUSHRA requires at least 20 retained participants")
    if set(retained_rows) != retained_ids:
        raise R2StatisticsError(
            "retained MUSHRA participant observations are incomplete"
        )
    for participant, observations in retained_rows.items():
        if len(observations) != 8 or len({row[0] for row in observations}) != 8:
            raise R2StatisticsError(
                f"retained participant {participant} requires eight unique excerpts"
            )
    excerpt_sets = [{row[0] for row in rows_} for rows_ in retained_rows.values()]
    if len({frozenset(excerpts) for excerpts in excerpt_sets}) != 1:
        raise R2StatisticsError("all retained participants must rate the same excerpts")
    return recruited, retained_rows


def hierarchical_mushra_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int = 10_000,
    seed: int = 20_260_829,
) -> dict[str, Any]:
    """Bootstrap the candidate-minus-A2 advantage by participant then excerpt."""
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    recruited, groups = _mushra_groups(rows)
    participant_ids = tuple(sorted(groups))
    point = float(
        np.mean([difference for rows_ in groups.values() for _, difference in rows_])
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_participants = rng.choice(
            participant_ids, size=len(participant_ids), replace=True
        )
        differences = []
        for participant in sampled_participants:
            excerpt_rows = groups[str(participant)]
            indices = rng.integers(0, len(excerpt_rows), size=len(excerpt_rows))
            differences.extend(excerpt_rows[int(index)][1] for index in indices)
        bootstrap[replicate] = np.mean(differences)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "method": "paired-hierarchical-r2-mushra-v1",
        "recruited_participants": len(recruited),
        "retained_participants": len(groups),
        "excerpts_per_participant": 8,
        "replicates": replicates,
        "seed": seed,
        "hierarchy": ["participant", "excerpt"],
        "candidate_minus_a2_points": point,
        "confidence_interval_95": {
            "lower": float(lower),
            "upper": float(upper),
        },
    }
