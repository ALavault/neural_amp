from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from fssr_nam.campaign.r2 import (
    FINAL_CONDITIONS,
    INTERNAL_VALIDATION_TRAJECTORIES,
    R2AuthorizationError,
    R2ConfigError,
    adaa_screen_specs,
    development_robustness_specs,
    final_condition_keys,
    initial_screen_specs,
    internal_validation_specs,
    make_run_id,
    parse_run_id,
    teacher_specs,
    validate_protocol_config,
    validate_repository_configs,
    validate_sealed_test_boundary,
    validate_stage_authorization,
)

ROOT = Path(__file__).resolve().parents[2]


def test_r2_canonical_protocol_and_administrative_lineage_validate() -> None:
    protocol = validate_repository_configs(ROOT)
    assert protocol["campaign_version"] == "FSSR-R2-v1"
    lineage = json.loads(
        (ROOT / ".codex_campaign/LINEAGES.json").read_text(encoding="utf-8")
    )
    assert lineage["active"] == "amp_sota_prototype_v1_2"
    assert (
        lineage["lineages"]["amp_sota_prototype_v1_2"]["parent"]
        == "amp_sota_prototype_v1_1"
    )
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
    assert lineage["lineages"]["r1"] == {
        "administrative_status": "superseded_by_R2",
        "historical_artifacts_immutable": True,
        "path": ".codex_campaign/r1",
    }
    assert lineage["lineages"]["r2"] == {
        "administrative_status": (
            "preserved_blocked_missing_physical_192khz_reference"
        ),
        "campaign_version": "FSSR-R2-v1",
        "historical_artifacts_immutable": True,
        "path": ".codex_campaign/r2",
    }


def test_ambitious_xl_is_declared_exploratory_and_not_a_frozen_screen_member() -> None:
    candidate = yaml.safe_load(
        (ROOT / "configs/models/r2/aa_fssr_xl.yaml").read_text(encoding="utf-8")
    )
    assert candidate["family"] == "aa-fssr-xl"
    assert candidate["role"] == "exploratory_out_of_gamut_not_in_frozen_screen"
    assert candidate["protocol_membership"] == "none_until_versioned_amendment"
    assert candidate["comparison_to_r1"]["deliberately_larger"] is True
    assert candidate["scientific_status"]["counted_run"] is False
    assert candidate["scientific_status"]["selection_eligible"] is False


def test_r2_protocol_threshold_changes_fail_closed() -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs/r2/protocol.yaml").read_text(encoding="utf-8")
    )
    protocol["decision"]["cpp_cpu_ratio_maximum"] = 1.251
    with pytest.raises(R2ConfigError, match="cpp_cpu_ratio_maximum"):
        validate_protocol_config(protocol)


def test_r2_run_ids_are_immutable_and_matrix_counts_are_exact() -> None:
    initial = initial_screen_specs()
    challengers = adaa_screen_specs("wright")
    assert len(initial) == 12
    assert len(challengers) == 4
    assert len({spec.run_id for spec in (*initial, *challengers)}) == 16
    assert len(teacher_specs("aa-fssr")) == 2
    assert len(development_robustness_specs("aa-fssr", "full_island_x2")) == 16
    internal = internal_validation_specs("aa-fssr", "full_island_x2")
    assert len(internal) == INTERNAL_VALIDATION_TRAJECTORIES
    assert len(final_condition_keys()) == FINAL_CONDITIONS
    run_id = make_run_id("screen", "bigmuff", "aa-fssr-m4", "adaa1", 0)
    assert parse_run_id(run_id).run_id == run_id


@pytest.mark.parametrize(
    "run_id",
    [
        "r2_screen_bigmuff_aa-fssr_adaa1_seed00_v1",
        "r2_screen_bigmuff_aa-fssr_adaa1_seed0_v2",
        "r2_screen_bigmuff_aa-fssr_adaa1_seed0_retry1_v1",
        "R2_screen_bigmuff_aa-fssr_adaa1_seed0_v1",
        "r2_unknown_bigmuff_aa-fssr_adaa1_seed0_v1",
    ],
)
def test_r2_run_id_aliases_and_retries_are_rejected(run_id: str) -> None:
    with pytest.raises(ValueError, match="invalid R2 run_id"):
        parse_run_id(run_id)


def test_r2_sequential_gates_require_literal_prior_evidence() -> None:
    validate_stage_authorization("preflight", {})
    with pytest.raises(R2AuthorizationError, match="preflight=passed"):
        validate_stage_authorization("capture", {})
    validate_stage_authorization("capture", {"preflight": "passed"})
    with pytest.raises(R2AuthorizationError, match="mechanism_x2=passed"):
        validate_stage_authorization("screen", {"mechanism_x2": "failed"})
    validate_stage_authorization("screen", {"mechanism_x2": "passed"})
    validate_stage_authorization(
        "distill", {"teacher": "passed", "deployable": "failed_double_gate"}
    )


def test_r2_sealed_boundary_requires_parity_benchmark_and_zero_prior_openings() -> None:
    decisions = {
        "confirm_validation": "passed",
        "python_cpp_parity": "passed",
        "benchmark": "passed",
    }
    validate_sealed_test_boundary(
        decisions=decisions,
        internal_tests_open_count=0,
        external_report_only_locked=True,
    )
    with pytest.raises(R2AuthorizationError, match="only once"):
        validate_sealed_test_boundary(
            decisions=decisions,
            internal_tests_open_count=1,
            external_report_only_locked=True,
        )
    with pytest.raises(R2AuthorizationError, match="EXTERNAL_REPORT_ONLY"):
        validate_sealed_test_boundary(
            decisions=decisions,
            internal_tests_open_count=0,
            external_report_only_locked=False,
        )
