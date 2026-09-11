from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from fssr_nam.campaign.amp_competence_arch_v2 import (
    ALL_FAMILIES,
    CANDIDATE_FAMILIES,
    CONTROL_FAMILY,
    SEEDS,
    SYSTEMS,
    ArchV2AuthorizationError,
    ArchV2ConfigError,
    make_run_id,
    parse_run_id,
    validate_protocol_config,
    validate_repository_state,
    validate_stage_authorization,
)

ROOT = Path(__file__).resolve().parents[2]


def _protocol() -> dict[str, object]:
    return yaml.safe_load(
        (ROOT / "configs/amp_competence_arch_v2/protocol.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_v2_protocol_is_competence_first_and_v1_is_terminal_immutable() -> None:
    protocol = validate_repository_state(ROOT)
    assert protocol["competence"]["control_family"] == CONTROL_FAMILY
    assert protocol["comparison"]["candidate_families"] == list(CANDIDATE_FAMILIES)
    assert protocol["boundaries"]["v1_runs_resumed"] is False
    assert protocol["boundaries"]["v1_runs_retuned"] is False
    assert protocol["boundaries"]["physical_audio_allowed"] is False


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("boundaries", "v1_runs_resumed", True, "v1_runs_resumed"),
        ("boundaries", "physical_audio_allowed", True, "physical_audio_allowed"),
        ("competence", "checkpoint_updates", [500, 1000], "checkpoints"),
        ("competence", "gain_error_strictly_greater_than", -0.3, "gain"),
        ("comparison", "bootstrap_replicates", 9999, "bootstrap"),
    ],
)
def test_decision_bearing_v2_protocol_drift_fails_closed(
    section: str, key: str, value: object, message: str
) -> None:
    protocol = deepcopy(_protocol())
    protocol[section][key] = value
    with pytest.raises(ArchV2ConfigError, match=message):
        validate_protocol_config(protocol)


@st.composite
def valid_run_components(draw: st.DrawFn) -> tuple[str, str, str, int]:
    stage = draw(st.sampled_from(["competence", "comparison"]))
    family = (
        CONTROL_FAMILY if stage == "competence" else draw(st.sampled_from(ALL_FAMILIES))
    )
    return (
        stage,
        draw(st.sampled_from(SYSTEMS)),
        family,
        draw(st.sampled_from(SEEDS)),
    )


@given(components=valid_run_components())
@settings(max_examples=40, deadline=None)
def test_v2_run_identifier_roundtrip_is_canonical(
    components: tuple[str, str, str, int],
) -> None:
    run_id = make_run_id(*components)
    parsed = parse_run_id(run_id)
    assert (parsed.stage, parsed.system, parsed.family, parsed.seed) == components
    assert parsed.run_id == run_id


def test_comparison_authorization_requires_passed_competence() -> None:
    with pytest.raises(ArchV2AuthorizationError, match="requires passed"):
        validate_stage_authorization("comparison", {"preflight": "passed"})
    with pytest.raises(ArchV2AuthorizationError, match="requires passed"):
        validate_stage_authorization(
            "comparison", {"preflight": "passed", "competence": "failed"}
        )
    validate_stage_authorization(
        "comparison", {"preflight": "passed", "competence": "passed"}
    )
