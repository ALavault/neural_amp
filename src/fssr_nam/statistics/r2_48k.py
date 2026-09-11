"""Paired bootstraps with historical development excluded from primary inference."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from fssr_nam.campaign.r2_48k import DEVELOPMENT_DEVICES, PRIMARY_DEVICES

CONFIRMATION_SEEDS = (0, 1, 2, 3, 4)


class R248KStatisticsError(ValueError):
    """Raised when rows do not match the frozen paired resampling units."""


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R248KStatisticsError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise R248KStatisticsError(f"{label} must be finite")
    return result


def _confirmation_groups(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[int, list[tuple[str, float, float]]]]:
    devices = (*DEVELOPMENT_DEVICES, *PRIMARY_DEVICES)
    groups: dict[str, dict[int, list[tuple[str, float, float]]]] = {
        device: defaultdict(list) for device in devices
    }
    observed: set[tuple[str, int, str]] = set()
    for row in rows:
        device = row.get("device")
        seed = row.get("seed")
        source = row.get("source")
        if device not in devices:
            raise R248KStatisticsError(f"unknown evaluation device: {device!r}")
        if isinstance(seed, bool) or seed not in CONFIRMATION_SEEDS:
            raise R248KStatisticsError(f"invalid evaluation seed: {seed!r}")
        if not isinstance(source, str) or not source:
            raise R248KStatisticsError("evaluation source must be non-empty")
        expected_role = (
            "historical_development"
            if device in DEVELOPMENT_DEVICES
            else "prospective_primary"
        )
        if row.get("evidence_role") != expected_role:
            raise R248KStatisticsError(f"{device} evidence role is invalid")
        if row.get("physical_asr_used_for_decision") is not False:
            raise R248KStatisticsError("physical ASR cannot enter R2-48K inference")
        key = (str(device), int(seed), source)
        if key in observed:
            raise R248KStatisticsError(
                "duplicate device/seed/source; windows are not observations"
            )
        observed.add(key)
        baseline = _finite(row.get("baseline_esr"), "baseline_esr")
        candidate = _finite(row.get("candidate_esr"), "candidate_esr")
        if baseline <= 0.0 or candidate < 0.0:
            raise R248KStatisticsError("ESR requires a positive baseline")
        groups[str(device)][int(seed)].append((source, baseline, candidate))
    for device, seed_groups in groups.items():
        if set(seed_groups) != set(CONFIRMATION_SEEDS):
            raise R248KStatisticsError(f"{device} seeds must be exactly 0--4")
        source_sets = [
            {source for source, _, _ in seed_groups[seed]}
            for seed in CONFIRMATION_SEEDS
        ]
        if not source_sets[0] or any(
            sources != source_sets[0] for sources in source_sets
        ):
            raise R248KStatisticsError(
                f"{device} paired source set must be identical for every seed"
            )
    return groups


def _improvement(rows: Sequence[tuple[str, float, float]]) -> float:
    baseline = float(np.mean([row[1] for row in rows]))
    candidate = float(np.mean([row[2] for row in rows]))
    return (baseline - candidate) / baseline


def hierarchical_confirmation_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int = 10_000,
    seed: int = 20_260_828,
) -> dict[str, Any]:
    """Bootstrap seeds then sources on Blackstar/UA; dev stays descriptive."""
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    groups = _confirmation_groups(rows)
    devices = (*DEVELOPMENT_DEVICES, *PRIMARY_DEVICES)
    point_device = {
        device: _improvement(
            [row for seed_rows in groups[device].values() for row in seed_rows]
        )
        for device in devices
    }
    heldout_point = float(np.mean([point_device[device] for device in PRIMARY_DEVICES]))
    development_point = float(
        np.mean([point_device[device] for device in DEVELOPMENT_DEVICES])
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    seed_values = np.asarray(CONFIRMATION_SEEDS, dtype=np.int64)
    for replicate in range(replicates):
        device_improvements = []
        for device in PRIMARY_DEVICES:
            sampled_seeds = rng.choice(seed_values, size=len(seed_values), replace=True)
            sampled_rows: list[tuple[str, float, float]] = []
            for sampled_seed in sampled_seeds:
                source_rows = groups[device][int(sampled_seed)]
                indices = rng.integers(0, len(source_rows), size=len(source_rows))
                sampled_rows.extend(source_rows[int(index)] for index in indices)
            device_improvements.append(_improvement(sampled_rows))
        bootstrap[replicate] = np.mean(device_improvements)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    won = [device for device, improvement in point_device.items() if improvement > 0.0]
    source_counts = {
        device: len(groups[device][CONFIRMATION_SEEDS[0]]) for device in devices
    }
    return {
        "method": "paired-hierarchical-r2-48k-confirmation-v1",
        "replicates": replicates,
        "seed": seed,
        "hierarchy": ["seed", "source"],
        "primary_devices": list(PRIMARY_DEVICES),
        "development_devices": list(DEVELOPMENT_DEVICES),
        "development_excluded_from_primary_interval": True,
        "windows_resampled": False,
        "probes_resampled": False,
        "evaluation_conditions": 40,
        "prospective_conditions": 20,
        "heldout_esr_relative_improvement": heldout_point,
        "heldout_esr_confidence_interval_95": {
            "lower": float(lower),
            "upper": float(upper),
        },
        "development_esr_relative_improvement_descriptive": development_point,
        "device_esr_relative_improvement": point_device,
        "devices_won": won,
        "source_counts": source_counts,
        "physical_asr_in_primary_decision": False,
        "inference_limit": "available_training_seeds_and_archived_sources_only",
    }


def _mushra_groups(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[set[str], dict[str, list[tuple[str, str, float]]]]:
    recruited: set[str] = set()
    retained_flags: dict[str, bool] = {}
    retained_rows: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    observed: set[tuple[str, str]] = set()
    devices = {*DEVELOPMENT_DEVICES, *PRIMARY_DEVICES}
    for row in rows:
        participant = row.get("participant")
        excerpt = row.get("excerpt")
        device = row.get("device")
        retained = row.get("retained")
        if not isinstance(participant, str) or not participant:
            raise R248KStatisticsError("MUSHRA participant must be non-empty")
        if not isinstance(excerpt, str) or not excerpt:
            raise R248KStatisticsError("MUSHRA excerpt must be non-empty")
        if device not in devices:
            raise R248KStatisticsError("MUSHRA device is invalid")
        if not isinstance(retained, bool):
            raise R248KStatisticsError("MUSHRA retained must be boolean")
        if (
            participant in retained_flags
            and retained_flags[participant] is not retained
        ):
            raise R248KStatisticsError("MUSHRA retention flag changed")
        retained_flags[participant] = retained
        recruited.add(participant)
        key = (participant, excerpt)
        if key in observed:
            raise R248KStatisticsError("duplicate participant/excerpt observation")
        observed.add(key)
        expected_role = (
            "development_secondary"
            if device in DEVELOPMENT_DEVICES
            else "prospective_primary"
        )
        if row.get("evidence_role") != expected_role:
            raise R248KStatisticsError("MUSHRA evidence role is invalid")
        if retained:
            candidate = _finite(row.get("candidate_score"), "candidate_score")
            baseline = _finite(row.get("a2_score"), "a2_score")
            if not 0.0 <= candidate <= 100.0 or not 0.0 <= baseline <= 100.0:
                raise R248KStatisticsError("MUSHRA scores must lie in [0, 100]")
            retained_rows[participant].append(
                (excerpt, str(device), candidate - baseline)
            )
    if len(recruited) != 24:
        raise R248KStatisticsError("MUSHRA requires 24 recruited participants")
    retained_ids = {name for name, retained in retained_flags.items() if retained}
    if len(retained_ids) < 20:
        raise R248KStatisticsError("MUSHRA requires at least 20 retained participants")
    if set(retained_rows) != retained_ids:
        raise R248KStatisticsError("retained participant rows are incomplete")
    for participant, observations in retained_rows.items():
        if len(observations) != 8 or len({row[0] for row in observations}) != 8:
            raise R248KStatisticsError(
                f"retained participant {participant} requires eight excerpts"
            )
        counts = defaultdict(int)
        for _, device, _ in observations:
            counts[device] += 1
        if dict(counts) != {
            device: 2 for device in (*DEVELOPMENT_DEVICES, *PRIMARY_DEVICES)
        }:
            raise R248KStatisticsError("MUSHRA requires two excerpts per device")
    excerpt_sets = [
        frozenset(row[0] for row in items) for items in retained_rows.values()
    ]
    if len(set(excerpt_sets)) != 1:
        raise R248KStatisticsError("retained listeners must rate identical excerpts")
    return recruited, retained_rows


def hierarchical_mushra_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int = 10_000,
    seed: int = 20_260_829,
) -> dict[str, Any]:
    """Bootstrap listeners then the four prospective Blackstar/UA excerpts."""
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    recruited, groups = _mushra_groups(rows)
    participant_ids = tuple(sorted(groups))
    primary_values = [
        difference
        for participant_rows in groups.values()
        for _, device, difference in participant_rows
        if device in PRIMARY_DEVICES
    ]
    development_values = [
        difference
        for participant_rows in groups.values()
        for _, device, difference in participant_rows
        if device in DEVELOPMENT_DEVICES
    ]
    point = float(np.mean(primary_values))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_participants = rng.choice(
            participant_ids, size=len(participant_ids), replace=True
        )
        differences = []
        for participant in sampled_participants:
            primary_rows = [
                row for row in groups[str(participant)] if row[1] in PRIMARY_DEVICES
            ]
            indices = rng.integers(0, len(primary_rows), size=len(primary_rows))
            differences.extend(primary_rows[int(index)][2] for index in indices)
        bootstrap[replicate] = np.mean(differences)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "method": "paired-hierarchical-r2-48k-mushra-v1",
        "recruited_participants": len(recruited),
        "retained_participants": len(groups),
        "excerpts_per_participant": 8,
        "primary_excerpts_per_participant": 4,
        "replicates": replicates,
        "seed": seed,
        "hierarchy": ["participant", "primary_excerpt"],
        "primary_devices": list(PRIMARY_DEVICES),
        "development_excluded_from_primary_interval": True,
        "primary_candidate_minus_a2_points": point,
        "primary_confidence_interval_95": {
            "lower": float(lower),
            "upper": float(upper),
        },
        "development_candidate_minus_a2_points_descriptive": float(
            np.mean(development_values)
        ),
    }
