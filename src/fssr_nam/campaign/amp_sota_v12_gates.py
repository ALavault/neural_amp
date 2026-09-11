"""Fail-closed gates for AMP-SOTA-PROTOTYPE-v1.2."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .amp_sota_prototype_v1_2 import (
    CANDIDATE_FAMILY,
    CONTROL_FAMILY,
    EVALUATION_UPDATES,
    SEEDS,
    SNAPSHOT_UPDATES,
    SYSTEMS,
)


class SotaV12GateError(RuntimeError):
    """Raised when v1.2 evidence is incomplete, duplicated, or malformed."""


def _finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SotaV12GateError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise SotaV12GateError(f"{label} must be finite")
    return converted


def evaluate_preflight_gate(
    evidence: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate preflight before any selection-eligible v1.2 source generation."""
    checks = {
        "protocol_frozen": evidence.get("protocol_frozen") is True,
        "clean_execution_snapshot": evidence.get("clean_execution_snapshot") is True,
        "cuda_device": (
            evidence.get("cuda_device_name") == protocol["resource_budget"]["gpu_name"]
            and _finite(evidence.get("cuda_total_memory_bytes"), "CUDA memory")
            >= float(protocol["resource_budget"]["gpu_total_memory_bytes_minimum"])
        ),
        "parent_terminal": evidence.get("parent_verdict") == "NO-GO-MECHANISM",
        "aa_dependency_only": (
            evidence.get("aa_backend") == "full_island_x2"
            and evidence.get("aa_backend_requalified") is False
        ),
        "tests": evidence.get("tests_passed") is True,
        "lint": evidence.get("lint_passed") is True,
        "data_audit": evidence.get("data_audit_passed") is True,
        "no_physical_audio": evidence.get("physical_audio_samples_read") == 0,
        "no_selection_eligible_synthetic_outputs": (
            evidence.get("selection_eligible_synthetic_samples_generated") == 0
        ),
        "test_fixtures_ineligible": (
            evidence.get("test_only_synthetic_fixtures_eligible_for_selection") is False
        ),
        "confirmation_closed": (evidence.get("confirmation_was_opened") is False),
        "fm9_closed": evidence.get("fm9_outputs_accessed") is False,
        "budget": (
            _finite(evidence.get("goal_gpu_hours_used"), "goal GPU hours")
            < float(protocol["resource_budget"]["active_goal_gpu_hours_maximum"])
            and _finite(evidence.get("goal_disk_gib_used"), "goal disk GiB")
            < float(protocol["resource_budget"]["active_goal_disk_gib_maximum"])
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _aggregate_metrics(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise SotaV12GateError(f"{label} aggregate metrics are missing")
    metrics = {
        metric: _finite(value.get(metric), f"{label} {metric}")
        for metric in (
            "esr",
            "mae",
            "gain_error",
            "correlation",
            "prediction_rms",
            "target_rms",
        )
    }
    for metric in ("esr", "mae", "prediction_rms", "target_rms"):
        if metrics[metric] < 0.0:
            raise SotaV12GateError(f"{label} {metric} must be non-negative")
    if not -1.000001 <= metrics["correlation"] <= 1.000001:
        raise SotaV12GateError(f"{label} correlation is outside [-1,1]")
    return metrics


def _validate_training_contract(
    row: Mapping[str, Any], protocol: Mapping[str, Any], label: str
) -> None:
    optimization = protocol["optimization"]
    expected = {
        "updates": SNAPSHOT_UPDATES[-1],
        "snapshot_updates": list(SNAPSHOT_UPDATES),
        "evaluation_updates": list(EVALUATION_UPDATES),
        "chunk_samples": optimization["training_chunk_samples"],
        "common_preroll_samples_after_alignment": optimization[
            "common_preroll_samples_after_alignment"
        ],
        "scored_start": optimization["scored_start_samples"],
        "scored_samples_per_episode": optimization["scored_samples_per_episode"],
        "post_training_fit_applied": False,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise SotaV12GateError(f"{label} training contract changed: {key}")


def _checkpoint_index(row: Mapping[str, Any], label: str) -> dict[int, dict[str, Any]]:
    checkpoints = row.get("checkpoints")
    if not isinstance(checkpoints, Sequence) or isinstance(checkpoints, (str, bytes)):
        raise SotaV12GateError(f"{label} checkpoints are missing")
    indexed: dict[int, dict[str, Any]] = {}
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, Mapping):
            raise SotaV12GateError(f"{label} checkpoint is malformed")
        update = checkpoint.get("update")
        if update not in EVALUATION_UPDATES or update in indexed:
            raise SotaV12GateError(f"{label} checkpoint update is invalid")
        internal_dev = checkpoint.get("internal_dev")
        if not isinstance(internal_dev, Mapping):
            raise SotaV12GateError(f"{label} INTERNAL_DEV evidence is missing")
        aggregate = _aggregate_metrics(
            internal_dev.get("aggregate"), f"{label}/{update}"
        )
        indexed[int(update)] = {"aggregate": aggregate, "raw": internal_dev}
    if set(indexed) != set(EVALUATION_UPDATES):
        raise SotaV12GateError(f"{label} checkpoint matrix is incomplete")
    return indexed


# Codex: Sourcery low-code-quality (13%); refactor with behavior/protocol tests.
def evaluate_competence_system_gate(
    trajectories: Sequence[Mapping[str, Any]],
    *,
    system: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one complete three-seed system before authorizing the next."""
    if system not in SYSTEMS:
        raise SotaV12GateError("unknown v1.2 competence system")
    if len(trajectories) != len(SEEDS):
        raise SotaV12GateError("v1.2 competence seed triplet is incomplete")
    indexed: dict[int, Mapping[str, Any]] = {}
    for row in trajectories:
        if row.get("family") != CANDIDATE_FAMILY or row.get("system") != system:
            raise SotaV12GateError("v1.2 competence family or system changed")
        seed = row.get("seed")
        if isinstance(seed, bool) or seed not in SEEDS or seed in indexed:
            raise SotaV12GateError("v1.2 competence seed is invalid or duplicated")
        indexed[int(seed)] = row
    if set(indexed) != set(SEEDS):
        raise SotaV12GateError("v1.2 competence seeds are incomplete")

    stability_failures = {
        seed: str(row.get("failure_reason", "stability failure"))
        for seed, row in indexed.items()
        if row.get("status") == "stability_failed"
    }
    unexpected_statuses = {
        seed: row.get("status")
        for seed, row in indexed.items()
        if row.get("status") not in {"completed", "stability_failed"}
    }
    if unexpected_statuses:
        raise SotaV12GateError(
            f"v1.2 competence contains operational failures: {unexpected_statuses}"
        )
    if stability_failures:
        return {
            "passed": False,
            "verdict": protocol["sequential_competence"]["nonfinite_model_verdict"],
            "system": system,
            "trajectory_count": len(trajectories),
            "stability_failures": stability_failures,
            "guards": {},
            "median_esr": {},
        }

    checkpoints = {}
    for seed in SEEDS:
        label = f"{system}/seed{seed}"
        _validate_training_contract(indexed[seed], protocol, label)
        checkpoints[seed] = _checkpoint_index(indexed[seed], label)
        for update in EVALUATION_UPDATES:
            evidence = checkpoints[seed][update]["raw"]
            if (
                evidence.get("latency_samples")
                != protocol["optimization"]["declared_latency_samples"]
            ):
                raise SotaV12GateError(f"{label}/{update} latency changed")
            if (
                evidence.get("common_preroll_samples_after_alignment")
                != protocol["optimization"]["common_preroll_samples_after_alignment"]
            ):
                raise SotaV12GateError(f"{label}/{update} pre-roll changed")
            sources = evidence.get("sources")
            if not isinstance(sources, Sequence) or len(sources) != 4:
                raise SotaV12GateError(f"{label}/{update} sources are incomplete")
            expected_spectral = update == EVALUATION_UPDATES[-1]
            for source in sources:
                if (
                    source.get("scored_samples")
                    != protocol["optimization"]["scored_samples_per_episode"]
                ):
                    raise SotaV12GateError(f"{label}/{update} score window changed")
                if source.get("spectral_metrics_included") is not expected_spectral:
                    raise SotaV12GateError(
                        f"{label}/{update} spectral schedule changed"
                    )
                spectral_present = "log_mel" in source and "mrstft" in source
                if spectral_present is not expected_spectral:
                    raise SotaV12GateError(
                        f"{label}/{update} spectral evidence changed"
                    )
    config = protocol["sequential_competence"]
    guards: dict[str, dict[str, Any]] = {}
    for update in EVALUATION_UPDATES:
        per_seed = {}
        for seed in SEEDS:
            metrics = checkpoints[seed][update]["aggregate"]
            per_seed[str(seed)] = {
                "gain": metrics["gain_error"]
                > float(config["gain_error_strictly_greater_than"]),
                "correlation": metrics["correlation"]
                > float(config["correlation_strictly_greater_than"]),
            }
        guards[str(update)] = {
            "per_seed": per_seed,
            "passed": all(all(checks.values()) for checks in per_seed.values()),
        }
    median_esr = {
        str(update): float(
            np.median([checkpoints[seed][update]["aggregate"]["esr"] for seed in SEEDS])
        )
        for update in EVALUATION_UPDATES
    }
    if median_esr["10000"] <= 0.0:
        raise SotaV12GateError("v1.2 competence ESR denominator must be positive")
    stability_ratio = median_esr["15000"] / median_esr["10000"]
    stability_passed = stability_ratio <= float(
        config["median_esr_15000_over_10000_maximum"]
    )
    passed = all(row["passed"] for row in guards.values()) and stability_passed
    return {
        "passed": passed,
        "verdict": None if passed else config["guard_failure_verdict"],
        "system": system,
        "trajectory_count": len(trajectories),
        "stability_failures": {},
        "guards": guards,
        "median_esr": median_esr,
        "median_esr_15000_over_10000": stability_ratio,
        "median_esr_stability_passed": stability_passed,
    }


def _final_sources(
    row: Mapping[str, Any], label: str, protocol: Mapping[str, Any]
) -> list[dict[str, float | int]]:
    if row.get("post_training_fit_applied") is not False:
        raise SotaV12GateError(f"{label} post-training fit is forbidden")
    if (
        row.get("slow_value_evaluation_source_seed")
        != protocol["slow_value"]["evaluation_source_seed"]
    ):
        raise SotaV12GateError(f"{label} slow-value source seed changed")
    digest = row.get("slow_value_evaluation_data_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise SotaV12GateError(f"{label} slow-value data identity is missing")
    evaluation = row.get("slow_value_evaluation")
    if not isinstance(evaluation, Mapping):
        raise SotaV12GateError(f"{label} slow-value evaluation is missing")
    if (
        evaluation.get("latency_samples")
        != protocol["optimization"]["declared_latency_samples"]
    ):
        raise SotaV12GateError(f"{label} latency changed")
    if (
        evaluation.get("common_preroll_samples_after_alignment")
        != protocol["optimization"]["common_preroll_samples_after_alignment"]
    ):
        raise SotaV12GateError(f"{label} pre-roll changed")
    sources = evaluation.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise SotaV12GateError(f"{label} source metrics are missing")
    if len(sources) != 4:
        raise SotaV12GateError(f"{label} must contain four source episodes")
    rows: list[dict[str, float | int]] = []
    seen: set[int] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            raise SotaV12GateError(f"{label} source metric is malformed")
        episode = source.get("episode")
        if not isinstance(episode, int) or isinstance(episode, bool) or episode in seen:
            raise SotaV12GateError(f"{label} source episode is invalid")
        seen.add(episode)
        rows.append(
            {"episode": episode}
            | {
                metric: _finite(source.get(metric), f"{label}/{episode}/{metric}")
                for metric in ("esr", "mae", "log_mel", "mrstft")
            }
        )
        if (
            source.get("scored_samples")
            != protocol["optimization"]["scored_samples_per_episode"]
        ):
            raise SotaV12GateError(f"{label}/{episode} score window changed")
        if source.get("spectral_metrics_included") is not True:
            raise SotaV12GateError(f"{label}/{episode} spectral evidence is missing")
        if any(
            float(rows[-1][metric]) < 0.0
            for metric in ("esr", "mae", "log_mel", "mrstft")
        ):
            raise SotaV12GateError(f"{label}/{episode} metric must be non-negative")
    if seen != {0, 1, 2, 3}:
        raise SotaV12GateError(f"{label} source episodes are incomplete")
    return rows


def _index_sources(
    trajectories: Sequence[Mapping[str, Any]],
    family: str,
    protocol: Mapping[str, Any],
) -> dict[tuple[str, int, int], dict[str, float | int]]:
    if len(trajectories) != len(SYSTEMS) * len(SEEDS):
        raise SotaV12GateError(f"{family} trajectory matrix is incomplete")
    indexed: dict[tuple[str, int, int], dict[str, float | int]] = {}
    seen_trajectories: set[tuple[str, int]] = set()
    for row in trajectories:
        system = row.get("system")
        seed = row.get("seed")
        if (
            row.get("family") != family
            or system not in SYSTEMS
            or isinstance(seed, bool)
            or seed not in SEEDS
            or row.get("status") != "completed"
        ):
            raise SotaV12GateError(f"{family} trajectory identity/status is invalid")
        trajectory_key = (str(system), int(seed))
        if trajectory_key in seen_trajectories:
            raise SotaV12GateError(f"duplicate {family} trajectory")
        seen_trajectories.add(trajectory_key)
        for source in _final_sources(row, f"{family}/{system}/seed{seed}", protocol):
            key = (str(system), int(seed), int(source["episode"]))
            indexed[key] = source
    expected = {
        (system, seed, episode)
        for system in SYSTEMS
        for seed in SEEDS
        for episode in range(4)
    }
    if set(indexed) != expected:
        raise SotaV12GateError(f"{family} source matrix is incomplete")
    return indexed


def _hierarchical_estimand(values: Mapping[tuple[str, int, int], float]) -> float:
    system_values = []
    for system in SYSTEMS:
        seed_values = [
            float(np.median([values[(system, seed, episode)] for episode in range(4)]))
            for seed in SEEDS
        ]
        system_values.append(float(np.median(seed_values)))
    return float(np.median(system_values))


def _per_system_estimand(
    values: Mapping[tuple[str, int, int], float], system: str
) -> float:
    return float(
        np.median(
            [
                np.median([values[(system, seed, episode)] for episode in range(4)])
                for seed in SEEDS
            ]
        )
    )


def _bootstrap_lower_bound(
    values: Mapping[tuple[str, int, int], float], *, replicates: int, seed: int
) -> float:
    if replicates < 1:
        raise SotaV12GateError("bootstrap replicates must be positive")
    generator = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    seed_array = np.asarray(SEEDS)
    episode_array = np.arange(4)
    for replicate in range(replicates):
        selected_seeds = generator.choice(seed_array, size=len(SEEDS), replace=True)
        selected_episodes = [
            generator.choice(episode_array, size=4, replace=True)
            for _ in selected_seeds
        ]
        system_values = []
        for system in SYSTEMS:
            seed_values = []
            for selected_seed, episodes in zip(
                selected_seeds, selected_episodes, strict=True
            ):
                seed_values.append(
                    float(
                        np.median(
                            [
                                values[(system, int(selected_seed), int(episode))]
                                for episode in episodes
                            ]
                        )
                    )
                )
            system_values.append(float(np.median(seed_values)))
        samples[replicate] = float(np.median(system_values))
    return float(np.percentile(samples, 2.5, method="linear"))


def evaluate_slow_value_gate(
    candidate_trajectories: Sequence[Mapping[str, Any]],
    control_trajectories: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Test whether the learned slow path adds paired multi-metric value."""
    candidate = _index_sources(candidate_trajectories, CANDIDATE_FAMILY, protocol)
    control = _index_sources(control_trajectories, CONTROL_FAMILY, protocol)
    candidate_pairing = {
        (str(row["system"]), int(row["seed"])): (
            row.get("training_data_sha256"),
            row.get("slow_value_evaluation_data_sha256"),
        )
        for row in candidate_trajectories
    }
    control_pairing = {
        (str(row["system"]), int(row["seed"])): (
            row.get("training_data_sha256"),
            row.get("slow_value_evaluation_data_sha256"),
        )
        for row in control_trajectories
    }
    if candidate_pairing != control_pairing:
        raise SotaV12GateError("candidate/control source identities are not paired")
    improvements: dict[tuple[str, int, int], float] = {}
    regressions: dict[str, dict[tuple[str, int, int], float]] = {
        metric: {} for metric in ("mae", "log_mel", "mrstft")
    }
    for key in sorted(candidate):
        control_esr = float(control[key]["esr"])
        if control_esr <= 0.0:
            raise SotaV12GateError("slow-value control ESR must be positive")
        improvements[key] = (control_esr - float(candidate[key]["esr"])) / control_esr
        for metric in regressions:
            denominator = float(control[key][metric])
            if denominator <= 0.0:
                raise SotaV12GateError(f"slow-value control {metric} must be positive")
            regressions[metric][key] = (
                float(candidate[key][metric]) - denominator
            ) / denominator

    config = protocol["slow_value"]
    point = _hierarchical_estimand(improvements)
    per_system = {
        system: _per_system_estimand(improvements, system) for system in SYSTEMS
    }
    metric_regressions = {
        metric: _hierarchical_estimand(values) for metric, values in regressions.items()
    }
    bootstrap = config["bootstrap"]
    lower = _bootstrap_lower_bound(
        improvements,
        replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]),
    )
    checks = {
        "paired_esr": point >= float(config["paired_esr_improvement_median_minimum"]),
        "dynamic_positive": per_system["dynamic_primary"]
        > float(config["dynamic_median_improvement_strictly_greater_than"]),
        "two_clippers_positive": per_system["two_clippers_primary"]
        > float(config["two_clippers_median_improvement_strictly_greater_than"]),
        "static_regression": -per_system["static_primary"]
        <= float(config["static_median_regression_maximum"]),
        "bootstrap": lower > float(bootstrap["lower_95_bound_strictly_greater_than"]),
        "noninferiority": all(
            regression <= float(config["median_relative_regression_maximum"])
            for regression in metric_regressions.values()
        ),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "verdict": config["success_verdict"] if passed else config["failure_verdict"],
        "checks": checks,
        "paired_esr_improvement": point,
        "per_system_esr_improvement": per_system,
        "metric_relative_regressions": metric_regressions,
        "bootstrap_lower_95_bound": lower,
        "paired_observation_count": len(improvements),
    }
