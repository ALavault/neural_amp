from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from fssr_nam.campaign.amp_sota_prototype_v1 import (
    BASE_CAMPAIGN_VERSION,
    CAMPAIGN_VERSION,
    SotaPrototypeAuthorizationError,
    SotaPrototypeConfigError,
    validate_amendment_config,
    validate_protocol_config,
    validate_repository_state,
    validate_stage_authorization,
)

ROOT = Path(__file__).resolve().parents[2]


def _protocol() -> dict[str, object]:
    return yaml.safe_load(
        (ROOT / "configs/amp_sota_prototype_v1/protocol.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_prototype_protocol_is_frozen_after_v3_terminal_closure() -> None:
    protocol = validate_repository_state(ROOT)
    assert protocol["campaign_version"] == BASE_CAMPAIGN_VERSION
    assert protocol == yaml.safe_load(
        (ROOT / ".codex_campaign/amp_sota_prototype_v1/PROTOCOL_LOCK.yaml").read_text(
            encoding="utf-8"
        )
    )
    amendment = yaml.safe_load(
        (ROOT / "configs/amp_sota_prototype_v1_1/protocol.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert amendment["campaign_version"] == CAMPAIGN_VERSION


def test_v1_1_terminal_lineage_matches_verdict_and_maturity() -> None:
    lineages = json.loads(
        (ROOT / ".codex_campaign/LINEAGES.json").read_text(encoding="utf-8")
    )
    maturity = json.loads(
        (ROOT / ".codex_campaign/amp_sota_prototype_v1_1/MATURITY.json").read_text(
            encoding="utf-8"
        )
    )
    verdict = json.loads(
        (ROOT / ".codex_campaign/amp_sota_prototype_v1_1/VERDICT.json").read_text(
            encoding="utf-8"
        )
    )
    entry = lineages["lineages"]["amp_sota_prototype_v1_1"]

    assert lineages["active"] == "amp_sota_prototype_v1_1"
    assert entry["administrative_status"] == "terminal_no_go_mechanism"
    assert entry["historical_artifacts_immutable"] is True
    assert maturity["status"] == "terminal_no_go"
    assert maturity["verdict"] == verdict["verdict"] == "NO-GO-MECHANISM"


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    [
        ("boundaries", "fm9_outputs_allowed", True),
        ("mechanism_screen", "maximum_promotions_per_axis", 2),
        ("architecture_search", "maximum_rounds", 4),
        ("confirmation", "median_esr_improvement_minimum", 0.09),
        ("prototype", "latency_samples_maximum", 65),
    ],
)
def test_prototype_protocol_rejects_decision_bearing_drift(
    section: str, key: str, replacement: object
) -> None:
    protocol = copy.deepcopy(_protocol())
    protocol[section][key] = replacement
    with pytest.raises(SotaPrototypeConfigError):
        validate_protocol_config(protocol)


def test_prototype_stages_are_fail_closed() -> None:
    validate_stage_authorization("preflight", {})
    validate_stage_authorization("fm9_protocol", {})
    with pytest.raises(SotaPrototypeAuthorizationError):
        validate_stage_authorization("screen", {})
    validate_stage_authorization("screen", {"preflight": "passed"})
    with pytest.raises(SotaPrototypeAuthorizationError):
        validate_stage_authorization("confirmation", {"candidate_lock": "passed"})
    validate_stage_authorization("native", {"candidate_lock": "passed"})
    validate_stage_authorization("confirmation", {"native": "passed"})


def test_v1_1_measurement_threshold_drift_is_rejected() -> None:
    amendment = yaml.safe_load(
        (ROOT / "configs/amp_sota_prototype_v1_1/protocol.yaml").read_text(
            encoding="utf-8"
        )
    )
    amendment["native_cost"]["block_size"] = 256
    with pytest.raises(SotaPrototypeConfigError, match="native block"):
        validate_amendment_config(amendment)
