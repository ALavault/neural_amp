from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from fssr_nam.campaign.amp_quality_arch_v3 import (
    CAMPAIGN_VERSION,
    ArchV3AuthorizationError,
    ArchV3ConfigError,
    load_round_one_lock,
    make_run_id,
    parse_run_id,
    validate_protocol_config,
    validate_repository_state,
    validate_stage_authorization,
)

ROOT = Path(__file__).resolve().parents[2]


def _protocol() -> dict[str, object]:
    return yaml.safe_load(
        (ROOT / "configs/amp_quality_arch_v3/protocol.yaml").read_text(encoding="utf-8")
    )


def test_v3_repository_is_preserved_as_terminal_ancestor() -> None:
    protocol = validate_repository_state(ROOT, require_frozen=True)
    assert protocol["campaign_version"] == CAMPAIGN_VERSION
    protocol_lock = yaml.safe_load(
        (ROOT / ".codex_campaign/amp_quality_arch_v3/PROTOCOL_LOCK.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert protocol == protocol_lock
    maturity = json.loads(
        (ROOT / ".codex_campaign/amp_quality_arch_v3/MATURITY.json").read_text(
            encoding="utf-8"
        )
    )
    assert maturity["status"] == "terminal_invalid"
    assert maturity["verdict"] == "INVALID"
    run_ledger = [
        json.loads(line)
        for line in (ROOT / ".codex_campaign/RUN_LEDGER.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    v3_runs = [
        entry
        for entry in run_ledger
        if entry.get("phase") == "AMP-QUALITY-ARCH-v3-ROUND-1"
    ]
    assert len(v3_runs) == maturity["scientific_runs_launched"]
    assert sum(entry["status"] == "completed" for entry in v3_runs) == 15
    assert sum(entry["status"] == "failed" for entry in v3_runs) == 1
    parent_protocol = yaml.safe_load(
        (ROOT / "configs/amp_competence_arch_v2/protocol.yaml").read_text(
            encoding="utf-8"
        )
    )
    parent_lock = yaml.safe_load(
        (ROOT / ".codex_campaign/amp_competence_arch_v2/PROTOCOL_LOCK.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert parent_protocol == parent_lock


def test_v3_round_one_is_prospectively_locked_after_feasibility() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/amp_quality_arch_v3/round_1.yaml").read_text(encoding="utf-8")
    )
    lock = yaml.safe_load(
        (ROOT / ".codex_campaign/amp_quality_arch_v3/ROUND_1_LOCK.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config == lock
    assert lock["matrix"]["trajectories"] == 27
    assert lock["selection"]["evidence_split"] == "INTERNAL_DEV"
    feasibility = json.loads(
        (
            ROOT / ".codex_campaign/amp_quality_arch_v3/TRAINING_FEASIBILITY_V2.json"
        ).read_text(encoding="utf-8")
    )
    assert feasibility["status"] == "passed"
    assert feasibility["validation_source_accessed"] is False
    assert feasibility["physical_audio_samples_read"] == 0
    assert load_round_one_lock(ROOT, _protocol()) == lock


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    [
        ("boundaries", "fm9_capture_allowed", True),
        ("synthetic_data", "scored_samples", 4096),
        ("representability_gate", "finite_residual_scale_ceiling_allowed", True),
        ("stages", "maximum_exploration_rounds", 4),
        ("runtime_gate", "block_64_cpu_ratio_maximum", 1.5),
    ],
)
def test_v3_protocol_rejects_decision_bearing_drift(
    section: str, key: str, replacement: object
) -> None:
    protocol = copy.deepcopy(_protocol())
    protocol[section][key] = replacement
    with pytest.raises(ArchV3ConfigError):
        validate_protocol_config(protocol)


def test_v3_run_ids_are_unambiguous_and_round_trip() -> None:
    run_id = make_run_id("round_1", "dynamic_primary", "slow_state_micro_tcn_x2", 2)
    assert run_id == (
        "arch_v3_round_1_dynamic_primary__slow_state_micro_tcn_x2_seed2_v1"
    )
    spec = parse_run_id(run_id)
    assert (spec.stage, spec.system, spec.family, spec.seed) == (
        "round_1",
        "dynamic_primary",
        "slow_state_micro_tcn_x2",
        2,
    )


def test_v3_stage_authorization_is_fail_closed() -> None:
    validate_stage_authorization("preflight", {})
    with pytest.raises(ArchV3AuthorizationError):
        validate_stage_authorization("round_1", {})
    validate_stage_authorization("round_1", {"preflight": "passed"})
    with pytest.raises(ArchV3AuthorizationError):
        validate_stage_authorization(
            "competence", {"preflight": "passed", "round_1": "failed"}
        )
    validate_stage_authorization(
        "competence", {"preflight": "passed", "round_1": "passed"}
    )
