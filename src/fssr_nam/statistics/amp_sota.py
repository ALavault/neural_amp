"""Paired hierarchical confirmation statistics for SOTA-PROTOTYPE-v1.1."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DEVICES = ("blackstar", "ua1176")
SEEDS = (0, 1, 2, 3, 4)


class AmpSotaStatisticsError(ValueError):
    """Confirmation rows do not match the frozen paired hierarchy."""


def _finite_positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AmpSotaStatisticsError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise AmpSotaStatisticsError(f"{label} must be finite and positive")
    return result


def _paired(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[int, dict[str, tuple[float, float]]]]:
    groups: dict[str, dict[int, dict[str, tuple[float, float]]]] = {
        device: defaultdict(dict) for device in DEVICES
    }
    for row in rows:
        device, seed, source = row.get("device"), row.get("seed"), row.get("source")
        if device not in DEVICES or seed not in SEEDS:
            raise AmpSotaStatisticsError("unknown device or seed")
        if not isinstance(source, str) or not source:
            raise AmpSotaStatisticsError("source must be non-empty")
        if source in groups[str(device)][int(seed)]:
            raise AmpSotaStatisticsError("duplicate device/seed/source observation")
        baseline = row.get("baseline")
        candidate = row.get("candidate")
        if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
            raise AmpSotaStatisticsError("baseline and candidate must be mappings")
        groups[str(device)][int(seed)][source] = (
            _finite_positive(baseline.get("esr"), "baseline ESR"),
            _finite_positive(candidate.get("esr"), "candidate ESR"),
        )
    for device, seed_groups in groups.items():
        if set(seed_groups) != set(SEEDS):
            raise AmpSotaStatisticsError(f"{device} requires seeds 0--4")
        source_sets = [set(seed_groups[seed]) for seed in SEEDS]
        if not source_sets[0] or any(value != source_sets[0] for value in source_sets):
            raise AmpSotaStatisticsError(
                f"{device} requires the same paired sources for every seed"
            )
    return groups


def _improvement(pairs: Sequence[tuple[float, float]]) -> float:
    values = [(baseline - candidate) / baseline for baseline, candidate in pairs]
    return float(np.median(values))


def hierarchical_confirmation_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int = 10_000,
    seed: int = 20_260_830,
) -> dict[str, Any]:
    """Resample paired seeds then sources; devices remain fixed strata."""
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    groups = _paired(rows)
    device_points = {
        device: _improvement(
            [pair for sources in groups[device].values() for pair in sources.values()]
        )
        for device in DEVICES
    }
    point = float(np.median(list(device_points.values())))
    generator = np.random.default_rng(seed)
    distribution = np.empty(replicates, dtype=np.float64)
    seed_ids = np.asarray(SEEDS, dtype=np.int64)
    for replicate in range(replicates):
        device_values = []
        for device in DEVICES:
            pairs: list[tuple[float, float]] = []
            sampled_seeds = generator.choice(seed_ids, size=len(seed_ids), replace=True)
            for sampled_seed in sampled_seeds:
                sources = groups[device][int(sampled_seed)]
                source_ids = np.asarray(sorted(sources), dtype=object)
                sampled_sources = generator.choice(
                    source_ids, size=len(source_ids), replace=True
                )
                pairs.extend(sources[str(source)] for source in sampled_sources)
            device_values.append(_improvement(pairs))
        distribution[replicate] = np.median(device_values)
    lower, upper = np.quantile(distribution, [0.025, 0.975], method="linear")
    return {
        "method": "paired-hierarchical-sota-confirmation-v1",
        "replicates": replicates,
        "seed": seed,
        "hierarchy": ["seed", "source"],
        "devices_fixed_strata": list(DEVICES),
        "windows_resampled": False,
        "esr_relative_improvement": point,
        "esr_relative_improvement_lower_95_bound": float(lower),
        "esr_relative_improvement_upper_95_bound": float(upper),
        "device_esr_relative_improvement": device_points,
    }
