from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from fssr_nam.campaign.r2_48k import (
    EVALUATION_CONDITIONS,
    INTERNAL_VALIDATION_TRAJECTORIES,
    PROSPECTIVE_CONDITIONS,
    R248KAuthorizationError,
    R248KConfigError,
    adaa_screen_specs,
    development_robustness_specs,
    evaluation_condition_keys,
    initial_screen_specs,
    internal_validation_specs,
    make_run_id,
    parse_run_id,
    prospective_condition_keys,
    teacher_specs,
    validate_protocol_config,
    validate_repository_configs,
    validate_sealed_test_boundary,
    validate_stage_authorization,
)

ROOT = Path(__file__).resolve().parents[2]


def test_r2_48k_is_preserved_ancestor_of_active_architecture_lineage() -> None:
    protocol = validate_repository_configs(ROOT)
    assert protocol["campaign_version"] == "FSSR-R2-48K-v1"
    assert protocol["claim_scope"]["physical_hardware_aliasing_claim_allowed"] is False
    assert protocol["decision"]["physical_asr_is_a_decision_metric"] is False
    lineage = json.loads(
        (ROOT / ".codex_campaign/LINEAGES.json").read_text(encoding="utf-8")
    )
    assert lineage["active"] == "amp_sota_prototype_v1_1"
    assert (
        lineage["lineages"]["amp_sota_prototype_v1_1"]["parent"]
        == "amp_sota_prototype_v1"
    )
    assert (
        lineage["lineages"]["amp_sota_prototype_v1"]["parent"] == "amp_quality_arch_v3"
    )
    assert (
        lineage["lineages"]["amp_quality_arch_v3"]["parent"] == "amp_competence_arch_v2"
    )
    assert lineage["lineages"]["amp_quality_arch_v1"]["parent"] == "quality_aa_v2"
    assert lineage["lineages"]["quality_aa_v2"]["parent"] == "quality_aa_v1"
    assert lineage["lineages"]["quality_aa_v1"]["parent"] == "r2_48k_v2"
    assert lineage["lineages"]["r2_48k_v2"]["parent"] == "r2_48k"
    assert lineage["lineages"]["r2_48k"]["historical_artifacts_immutable"] is True
    assert lineage["lineages"]["r2"]["historical_artifacts_immutable"] is True
    old_lock = yaml.safe_load(
        (ROOT / ".codex_campaign/r2/PROTOCOL_LOCK.yaml").read_text(encoding="utf-8")
    )
    assert old_lock["campaign_version"] == "FSSR-R2-v1"
    assert old_lock["status"] == "frozen_before_capture"
    assert old_lock["scientific_runs_launched"] == 0


def test_r2_48k_admits_xl_without_changing_old_exploratory_config() -> None:
    admitted = yaml.safe_load(
        (ROOT / "configs/models/r2_48k/aa_fssr_xl.yaml").read_text(encoding="utf-8")
    )
    old = yaml.safe_load(
        (ROOT / "configs/models/r2/aa_fssr_xl.yaml").read_text(encoding="utf-8")
    )
    assert admitted["selection_eligible"] is True
    assert admitted["admission_basis"].startswith("versioned_campaign")
    assert old["campaign_version"] == "FSSR-R2-v1"
    assert old["scientific_status"]["selection_eligible"] is False


def test_r2_48k_protocol_changes_fail_closed() -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs/r2_48k/protocol.yaml").read_text(encoding="utf-8")
    )
    protocol["claim_scope"]["physical_hardware_aliasing_claim_allowed"] = True
    with pytest.raises(R248KConfigError, match="physical_hardware_aliasing"):
        validate_protocol_config(protocol)


def test_r2_48k_run_ids_and_ambitious_matrix_are_exact() -> None:
    initial = initial_screen_specs()
    challengers = adaa_screen_specs("m4")
    assert len(initial) == 16
    assert len(challengers) == 6
    assert len({spec.run_id for spec in (*initial, *challengers)}) == 22
    assert any("aa-fssr-xl" in spec.run_id for spec in initial)
    assert len(teacher_specs("aa-fssr-xl")) == 2
    assert len(development_robustness_specs("aa-fssr-xl", "full_island_x2")) == 16
    assert (
        len(internal_validation_specs("aa-fssr-xl", "full_island_x2"))
        == INTERNAL_VALIDATION_TRAJECTORIES
    )
    assert len(evaluation_condition_keys()) == EVALUATION_CONDITIONS
    assert len(prospective_condition_keys()) == PROSPECTIVE_CONDITIONS
    run_id = make_run_id("screen", "bigmuff", "aa-fssr-xl-m4", "adaa1", 0)
    assert parse_run_id(run_id).run_id == run_id


@pytest.mark.parametrize(
    "run_id",
    [
        "r2_screen_bigmuff_aa-fssr_adaa1_seed0_v1",
        "r2_48k_screen_bigmuff_aa-fssr_adaa1_seed00_v1",
        "r2_48k_screen_bigmuff_aa-fssr_adaa1_seed0_v2",
        "r2_48k_screen_bigmuff_aa-fssr_adaa1_seed0_retry_v1",
    ],
)
def test_r2_48k_run_id_aliases_are_rejected(run_id: str) -> None:
    with pytest.raises(ValueError, match="invalid R2-48K run_id"):
        parse_run_id(run_id)


def test_r2_48k_gates_require_literal_prior_evidence() -> None:
    validate_stage_authorization("preflight", {})
    with pytest.raises(R248KAuthorizationError, match="preflight=passed"):
        validate_stage_authorization("data", {})
    validate_stage_authorization("mechanism", {"data": "passed"})
    with pytest.raises(R248KAuthorizationError, match="mechanism_x2=passed"):
        validate_stage_authorization("screen", {"mechanism_x2": "failed"})
    validate_stage_authorization("screen", {"mechanism_x2": "passed"})
    validate_stage_authorization(
        "distill",
        {"teacher": "passed", "deployable": "failed_fidelity_gate"},
    )


def test_r2_48k_seal_counts_only_prospective_tests_and_discloses_dev_history() -> None:
    decisions = {
        "confirm_validation": "passed",
        "python_cpp_parity": "passed",
        "benchmark": "passed",
        "mechanism_x2": "passed",
    }
    validate_sealed_test_boundary(
        decisions=decisions,
        internal_validation_tests_open_count=0,
        external_report_only_locked=True,
        development_tests_previously_observed=True,
    )
    with pytest.raises(R248KAuthorizationError, match="only once"):
        validate_sealed_test_boundary(
            decisions=decisions,
            internal_validation_tests_open_count=1,
            external_report_only_locked=True,
            development_tests_previously_observed=True,
        )
    with pytest.raises(R248KAuthorizationError, match="disclose"):
        validate_sealed_test_boundary(
            decisions=decisions,
            internal_validation_tests_open_count=0,
            external_report_only_locked=True,
            development_tests_previously_observed=False,
        )
