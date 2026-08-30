from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from fssr_nam.campaign.amp_sota_prototype_v1_2 import (
    CANDIDATE_FAMILY,
    CONTROL_FAMILY,
    SEEDS,
    SYSTEMS,
)
from fssr_nam.campaign.amp_sota_v12_gates import (
    SotaV12GateError,
    _bootstrap_lower_bound,
    evaluate_competence_system_gate,
    evaluate_preflight_gate,
    evaluate_slow_value_gate,
)

ROOT = Path(__file__).resolve().parents[2]


def _protocol() -> dict[str, object]:
    return yaml.safe_load(
        (ROOT / "configs/amp_sota_prototype_v1_2/protocol.yaml").read_text(
            encoding="utf-8"
        )
    )


def _fast_protocol() -> dict[str, object]:
    protocol = _protocol()
    protocol["slow_value"]["bootstrap"]["replicates"] = 64
    return protocol


def _aggregate(
    *, esr: float, gain: float = -0.1, corr: float = 0.95
) -> dict[str, float]:
    return {
        "esr": esr,
        "mae": 0.1,
        "gain_error": gain,
        "correlation": corr,
        "prediction_rms": 0.3,
        "target_rms": 0.32,
    }


def _competence_rows(system: str = "dynamic_primary") -> list[dict[str, object]]:
    def evaluation(esr: float, *, spectral: bool) -> dict[str, object]:
        sources = []
        for episode in range(4):
            source = {
                "episode": episode,
                **_aggregate(esr=esr),
                "scored_samples": 57_568,
                "spectral_metrics_included": spectral,
            }
            if spectral:
                source |= {"log_mel": 0.2, "mrstft": 0.3}
            sources.append(source)
        return {
            "aggregate": _aggregate(esr=esr),
            "sources": sources,
            "latency_samples": 32,
            "common_preroll_samples_after_alignment": 14_400,
        }

    return [
        {
            "family": CANDIDATE_FAMILY,
            "system": system,
            "seed": seed,
            "status": "completed",
            "updates": 15_000,
            "snapshot_updates": [500, 1_000, 2_000, 5_000, 10_000, 15_000],
            "evaluation_updates": [10_000, 15_000],
            "chunk_samples": 8_192,
            "common_preroll_samples_after_alignment": 14_400,
            "scored_start": 14_432,
            "scored_samples_per_episode": 57_568,
            "post_training_fit_applied": False,
            "checkpoints": [
                {
                    "update": 10_000,
                    "internal_dev": evaluation(0.10 + seed * 0.01, spectral=False),
                },
                {
                    "update": 15_000,
                    "internal_dev": evaluation(0.09 + seed * 0.01, spectral=True),
                },
            ],
        }
        for seed in SEEDS
    ]


@given(order=st.permutations(tuple(range(3))))
@settings(max_examples=6, deadline=None)
def test_competence_gate_is_seed_order_invariant(order: tuple[int, ...]) -> None:
    rows = _competence_rows()
    gate = evaluate_competence_system_gate(
        [rows[index] for index in order],
        system="dynamic_primary",
        protocol=_protocol(),
    )
    assert gate["passed"] is True
    assert gate["median_esr_15000_over_10000"] < 1.0


def test_competence_gate_uses_strict_guards_and_classifies_stability() -> None:
    rows = _competence_rows()
    rows[0]["checkpoints"][-1]["internal_dev"]["aggregate"]["correlation"] = 0.9
    gate = evaluate_competence_system_gate(
        rows, system="dynamic_primary", protocol=_protocol()
    )
    assert gate["passed"] is False
    assert gate["verdict"] == "NO-GO-COMPETENCE-v1.2"

    stability = _competence_rows()
    stability[1] = {
        "family": CANDIDATE_FAMILY,
        "system": "dynamic_primary",
        "seed": 1,
        "status": "stability_failed",
        "failure_reason": "non-finite gradient",
    }
    result = evaluate_competence_system_gate(
        stability, system="dynamic_primary", protocol=_protocol()
    )
    assert result["verdict"] == "NO-GO-STABILITY-v1.2"
    assert result["stability_failures"] == {1: "non-finite gradient"}


def test_competence_gate_rejects_bool_seed_and_window_drift() -> None:
    rows = _competence_rows()
    rows[0]["seed"] = False
    with pytest.raises(SotaV12GateError, match="seed"):
        evaluate_competence_system_gate(
            rows, system="dynamic_primary", protocol=_protocol()
        )
    rows = _competence_rows()
    rows[0]["scored_samples_per_episode"] = 57_567
    with pytest.raises(SotaV12GateError, match="contract"):
        evaluate_competence_system_gate(
            rows, system="dynamic_primary", protocol=_protocol()
        )


def _source_rows(family: str, factor: float) -> list[dict[str, object]]:
    trajectories = []
    for system_index, system in enumerate(SYSTEMS):
        for seed in SEEDS:
            sources = []
            for episode in range(4):
                base = 1.0 + 0.01 * system_index + 0.001 * seed + 0.0001 * episode
                sources.append(
                    {
                        "episode": episode,
                        "esr": factor * base,
                        "mae": factor * 0.1 * base,
                        "log_mel": factor * 0.2 * base,
                        "mrstft": factor * 0.3 * base,
                    }
                )
            trajectories.append(
                {
                    "family": family,
                    "system": system,
                    "seed": seed,
                    "status": "completed",
                    "training_data_sha256": "a" * 64,
                    "slow_value_evaluation_data_sha256": "b" * 64,
                    "slow_value_evaluation_source_seed": 40_362_830,
                    "post_training_fit_applied": False,
                    "slow_value_evaluation": {
                        "latency_samples": 32,
                        "common_preroll_samples_after_alignment": 14_400,
                        "sources": [
                            source
                            | {
                                "scored_samples": 57_568,
                                "spectral_metrics_included": True,
                            }
                            for source in sources
                        ],
                    },
                }
            )
    return trajectories


@given(order=st.permutations(tuple(range(9))))
@settings(max_examples=10, deadline=None)
def test_slow_value_gate_is_trajectory_order_invariant(
    order: tuple[int, ...],
) -> None:
    candidate = _source_rows(CANDIDATE_FAMILY, 0.9)
    control = _source_rows(CONTROL_FAMILY, 1.0)
    gate = evaluate_slow_value_gate(
        [candidate[index] for index in order],
        [control[index] for index in reversed(order)],
        _fast_protocol(),
    )
    assert gate["passed"] is True
    assert gate["paired_esr_improvement"] == pytest.approx(0.1)
    assert gate["bootstrap_lower_95_bound"] == pytest.approx(0.1)
    assert gate["paired_observation_count"] == 36


def test_slow_value_gate_rejects_missing_and_strict_dynamic_boundary() -> None:
    candidate = _source_rows(CANDIDATE_FAMILY, 0.9)
    control = _source_rows(CONTROL_FAMILY, 1.0)
    with pytest.raises(SotaV12GateError, match="matrix"):
        evaluate_slow_value_gate(candidate[:-1], control, _fast_protocol())

    boundary = copy.deepcopy(candidate)
    for row in boundary:
        if row["system"] == "dynamic_primary":
            for source in row["slow_value_evaluation"]["sources"]:
                source["esr"] /= 0.9
    gate = evaluate_slow_value_gate(boundary, control, _fast_protocol())
    assert gate["checks"]["dynamic_positive"] is False
    assert gate["passed"] is False


def test_slow_value_gate_rejects_unpaired_or_out_of_domain_evidence() -> None:
    candidate = _source_rows(CANDIDATE_FAMILY, 0.9)
    control = _source_rows(CONTROL_FAMILY, 1.0)
    control[0]["slow_value_evaluation_data_sha256"] = "c" * 64
    with pytest.raises(SotaV12GateError, match="not paired"):
        evaluate_slow_value_gate(candidate, control, _fast_protocol())

    control = _source_rows(CONTROL_FAMILY, 1.0)
    candidate[0]["slow_value_evaluation"]["sources"][0]["mae"] = -0.1
    with pytest.raises(SotaV12GateError, match="non-negative"):
        evaluate_slow_value_gate(candidate, control, _fast_protocol())


def test_bootstrap_reuses_crossed_draws_across_fixed_systems() -> None:
    values = {
        (system, seed, episode): (0.1 * system_index + 0.01 * seed + 0.001 * episode)
        for system_index, system in enumerate(SYSTEMS)
        for seed in SEEDS
        for episode in range(4)
    }
    observed = _bootstrap_lower_bound(values, replicates=64, seed=73)
    generator = np.random.default_rng(73)
    samples = []
    for _ in range(64):
        selected_seeds = generator.choice(np.asarray(SEEDS), size=3, replace=True)
        selected_episodes = [
            generator.choice(np.arange(4), size=4, replace=True) for _ in selected_seeds
        ]
        per_system = []
        for system in SYSTEMS:
            per_seed = [
                np.median(
                    [values[(system, int(seed), int(episode))] for episode in episodes]
                )
                for seed, episodes in zip(
                    selected_seeds, selected_episodes, strict=True
                )
            ]
            per_system.append(np.median(per_seed))
        samples.append(np.median(per_system))
    expected = np.percentile(samples, 2.5, method="linear")
    assert observed == pytest.approx(expected)


def test_preflight_gate_requires_zero_observation_and_remaining_budget() -> None:
    evidence = {
        "protocol_frozen": True,
        "clean_execution_snapshot": True,
        "cuda_device_name": "NVIDIA RTX PRO 4000 Blackwell",
        "cuda_total_memory_bytes": 25_149_898_752,
        "parent_verdict": "NO-GO-MECHANISM",
        "aa_backend": "full_island_x2",
        "aa_backend_requalified": False,
        "tests_passed": True,
        "lint_passed": True,
        "data_audit_passed": True,
        "physical_audio_samples_read": 0,
        "selection_eligible_synthetic_samples_generated": 0,
        "test_only_synthetic_fixtures_eligible_for_selection": False,
        "confirmation_was_opened": False,
        "fm9_outputs_accessed": False,
        "goal_gpu_hours_used": 0.0,
        "goal_disk_gib_used": 1.0,
    }
    assert evaluate_preflight_gate(evidence, _protocol())["passed"] is True
    evidence["selection_eligible_synthetic_samples_generated"] = 1
    assert evaluate_preflight_gate(evidence, _protocol())["passed"] is False
