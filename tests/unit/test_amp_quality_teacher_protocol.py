from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from fssr_nam.campaign.amp_quality_teacher_v1 import (
    CAMPAIGN_VERSION,
    PARENT_CAMPAIGN,
    PROTOCOL_LOCK_PATH,
    STAGES,
    QualityTeacherAuthorizationError,
    QualityTeacherConfigError,
    build_trajectory_matrix,
    load_protocol,
    make_run_id,
    parse_run_id,
    validate_repository_state,
    validate_stage_authorization,
)
from fssr_nam.models.quality_teacher import QUALITY_TEACHER_FAMILIES
from fssr_nam.models.quality_teacher_comparators import (
    QUALITY_TEACHER_COMPARATORS,
)

ROOT = Path(__file__).resolve().parents[2]


def test_quality_teacher_pointer_lock_and_superseded_parent_are_canonical() -> None:
    pointer = yaml.safe_load(
        (ROOT / "configs/amp_quality_teacher_v1/protocol.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert pointer == {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "canonical_lock": str(PROTOCOL_LOCK_PATH),
    }

    lock = yaml.safe_load((ROOT / PROTOCOL_LOCK_PATH).read_text(encoding="utf-8"))
    assert load_protocol(ROOT) == lock
    assert validate_repository_state(ROOT) == lock

    parent_state = (
        ROOT / ".codex_campaign/amp_sota_prototype_v1_2/STATE.md"
    ).read_text(encoding="utf-8")
    assert f"Lignée : `{PARENT_CAMPAIGN}`" in parent_state
    assert "Statut : `superseded` par `AMP-QUALITY-TEACHER-v1`" in parent_state
    assert "préflight passé et préservé" in parent_state
    assert "Runs scientifiques v1.2 : 0" in parent_state
    assert "Gates v1.2 évalués : 1 (`preflight=passed`)" in parent_state
    assert "Aucun verdict `INVALID` ou scientifique" in parent_state

    teacher_state = (
        ROOT / ".codex_campaign/amp_quality_teacher_v1/STATE.md"
    ).read_text(encoding="utf-8")
    assert (
        "Statut : prospective, préflight" in teacher_state
        or "Statut : active, préflight passé" in teacher_state
        or "Statut : terminal `INVALID` au gate `data_audit`" in teacher_state
    )
    assert "Runs scientifiques : 0" in teacher_state
    assert "Waveforms de test lus : 0" in teacher_state
    maturity = json.loads(
        (ROOT / ".codex_campaign/amp_quality_teacher_v1/MATURITY.json").read_text()
    )
    assert maturity["scientific_runs_launched"] == 0
    assert maturity["test_waveform_samples_read"] == 0
    assert maturity["current_stage"] in {
        "preflight",
        "data_audit",
        "terminal_data_audit_invalid",
    }
    assert maturity["gates_evaluated"] in {0, 1, 2}


def test_quality_teacher_trajectory_matrix_is_exactly_25_and_300_gpu_hours() -> None:
    protocol = load_protocol(ROOT)
    budget = protocol["trajectory_matrix"]
    assert budget == {
        "ampeg_slow_value_trajectories": 2,
        "additional_development_candidate_trajectories": 2,
        "development_comparator_trajectories": 9,
        "confirmation_trajectories": 12,
        "maximum_trajectories": 25,
        "gpu_hours_per_trajectory_maximum": 12,
        "scientific_gpu_hours_maximum": 300,
        "preflight_and_evaluation_gpu_hours_reserved": 24,
        "residual_budget_variant_or_retry_allowed": False,
    }

    candidate = QUALITY_TEACHER_FAMILIES[0]
    comparator = QUALITY_TEACHER_COMPARATORS[0]
    runs = build_trajectory_matrix(candidate, comparator)
    assert len(runs) == len({run.run_id for run in runs}) == 25
    assert Counter(run.stage for run in runs) == {
        "ampeg_slow_value": 2,
        "development": 11,
        "confirmation_train": 12,
    }
    assert Counter(run.device for run in runs) == {
        "fulltone": 4,
        "bigmuff": 4,
        "ampeg": 5,
        "rodent": 6,
        "fuzzy_logic": 6,
    }
    assert 25 * budget["gpu_hours_per_trajectory_maximum"] == 300


@pytest.mark.parametrize(
    ("stage", "device", "family", "seed"),
    [
        ("preflight", "all", "protocol", 0),
        ("development", "bigmuff", "nam_a2_full", 0),
        (
            "confirmation_train",
            "fuzzy_logic",
            "s4_tfilm_wavenet_x2_teacher",
            2,
        ),
    ],
)
def test_quality_teacher_run_ids_are_exact_and_round_trip(
    stage: str, device: str, family: str, seed: int
) -> None:
    run_id = make_run_id(stage, device, family, seed)
    assert run_id == (f"quality_teacher_v1_{stage}_{device}_{family}_seed{seed}_v1")
    spec = parse_run_id(run_id)
    assert (spec.stage, spec.device, spec.family, spec.seed, spec.run_id) == (
        stage,
        device,
        family,
        seed,
        run_id,
    )


def test_quality_teacher_run_ids_reject_unregistered_components() -> None:
    with pytest.raises(QualityTeacherConfigError, match="outside registry"):
        make_run_id("development", "blackstar", "nam_a2_full", 0)
    with pytest.raises(QualityTeacherConfigError, match="non-negative integer"):
        make_run_id("development", "ampeg", "nam_a2_full", True)
    with pytest.raises(QualityTeacherConfigError, match="invalid quality-teacher"):
        parse_run_id("quality_teacher_v1_development_ampeg_nam_a2_full_seed00_v1")


def test_quality_teacher_stage_authorization_is_strictly_ordered() -> None:
    passed: dict[str, str] = {}
    for index, stage in enumerate(STAGES):
        validate_stage_authorization(stage, passed)
        if index:
            predecessor = STAGES[index - 1]
            missing_predecessor = dict(passed)
            missing_predecessor.pop(predecessor)
            with pytest.raises(
                QualityTeacherAuthorizationError,
                match=predecessor,
            ):
                validate_stage_authorization(stage, missing_predecessor)
        passed[stage] = "passed"

    with pytest.raises(QualityTeacherAuthorizationError, match="unknown stage"):
        validate_stage_authorization("retry", passed)
