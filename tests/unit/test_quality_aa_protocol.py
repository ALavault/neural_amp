from __future__ import annotations

import json
from pathlib import Path

import pytest

from fssr_nam.campaign.quality_aa_provenance import strict_json
from fssr_nam.campaign.quality_aa_v2 import (
    QualityAAV2AuthorizationError,
    make_run_id,
    resolve_protocol,
    validate_repository_configs,
    validate_stage_authorization,
)

ROOT = Path(__file__).resolve().parents[2]


def test_quality_aa_v2_repository_contract_is_preserved_and_frozen() -> None:
    protocol = validate_repository_configs(ROOT)
    assert protocol["campaign_version"] == "FSSR-QUALITY-AA-v2"
    assert protocol["scope"]["physical_audio_allowed"] is False
    assert protocol["scope"]["global_state_of_the_art_claim_allowed"] is False
    assert "off" in protocol["routes"]
    assert False not in protocol["routes"]


def test_quality_aa_v1_invalid_record_is_preserved() -> None:
    verdict = json.loads(
        (ROOT / ".codex_campaign/quality_aa_v1/VERDICT.json").read_text()
    )
    assert verdict["status"] == "INVALID"
    assert verdict["candidate_route_outputs_observed"] is False
    assert verdict["rerun_authorized"] is False


def test_v2_resolver_changes_only_transport_metadata_and_off_key() -> None:
    protocol = resolve_protocol(ROOT)
    assert protocol["version_amendment"]["scientific_thresholds_changed"] is False
    assert protocol["version_amendment"]["fixtures_routes_probes_changed"] is False
    assert protocol["version_amendment"]["parent_numeric_evidence_reused"] is False
    assert set(protocol["routes"]) == {
        "off",
        "full_island_x2",
        "teacher_x4",
        "adaa1",
    }
    strict_json(protocol)


def test_strict_json_rejects_non_text_keys_before_sorting() -> None:
    with pytest.raises(TypeError, match="must be text"):
        strict_json({"routes": {False: {}, "full_island_x2": {}}})


def test_quality_aa_ids_and_stage_gates_fail_closed() -> None:
    assert make_run_id("mechanism") == "quality_aa_v2_mechanism_synthetic_all_seed0_v1"
    with pytest.raises(ValueError, match="invalid component"):
        make_run_id("mechanism", "bad-route")
    with pytest.raises(QualityAAV2AuthorizationError, match="preflight=passed"):
        validate_stage_authorization("mechanism", {"preflight": "failed"})
    validate_stage_authorization("mechanism", {"preflight": "passed"})
