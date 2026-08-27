from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fssr_nam.training.r1_competence import (
    CompetenceProtocolError,
    evaluate_competence_gate,
    seed_zero_allows_additional_runs,
    validate_competence_protocol,
)

ROOT = Path(__file__).resolve().parents[2]


def _config() -> dict[str, object]:
    return yaml.safe_load(
        (ROOT / "configs/training/r1_competence.yaml").read_text(encoding="utf-8")
    )


def test_locked_competence_protocol_and_inclusive_seed_zero_boundary() -> None:
    config = _config()
    validate_competence_protocol(config)
    assert seed_zero_allows_additional_runs(config, 0.15)
    assert not seed_zero_allows_additional_runs(config, 0.1500001)


def test_competence_gate_stops_after_failed_seed_zero() -> None:
    decision = evaluate_competence_gate(_config(), {0: 0.16})
    assert decision.status == "failed"
    assert not decision.seed_zero_continuation
    assert decision.median_esr is None


def test_competence_gate_requires_and_evaluates_all_promoted_seeds() -> None:
    config = _config()
    with pytest.raises(ValueError, match="missing"):
        evaluate_competence_gate(config, {0: 0.12, 1: 0.13})
    passed = evaluate_competence_gate(config, {0: 0.12, 1: 0.14, 2: 0.19})
    assert passed.passed
    assert passed.median_esr == pytest.approx(0.14)
    failed = evaluate_competence_gate(config, {0: 0.12, 1: 0.14, 2: 0.21})
    assert not failed.passed
    assert failed.maximum_esr == pytest.approx(0.21)


def test_competence_protocol_rejects_recipe_drift() -> None:
    config = _config()
    config["maximum_epochs"] = 1999
    with pytest.raises(CompetenceProtocolError, match="maximum_epochs"):
        validate_competence_protocol(config)
