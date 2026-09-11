from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from fssr_nam.campaign.amp_arch_gates import (
    ArchGateError,
    evaluate_arch_mechanism_gate,
)

ROOT = Path(__file__).resolve().parents[2]


def _protocol() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/amp_quality_arch_v1/protocol.yaml").read_text(encoding="utf-8")
    )


def _row(family: str, system: str, weight: float, esr: float) -> dict:
    return {
        "family": family,
        "system": system,
        "auxiliary_weight": weight,
        "validation": {
            "esr": esr,
            "gain_error": -0.05,
            "correlation": 0.97,
        },
    }


def _passing_trajectories() -> list[dict]:
    return [
        _row("micro_tcn_x2", "static_composite", 0.0, 0.20),
        _row("micro_tcn_x2", "dynamic_composite", 0.0, 1.00),
        _row("phys_det_tcn_x2", "static_composite", 0.0, 0.205),
        _row("phys_det_tcn_x2", "dynamic_composite", 0.0, 0.85),
        _row("phys_s6_tcn_x2", "dynamic_composite", 0.01, 0.82),
        _row("phys_s6_tcn_x2", "dynamic_composite", 0.05, 0.78),
        _row("phys_s6_tcn_x2", "dynamic_composite", 0.10, 0.80),
        _row("phys_s6_tcn_x2", "static_composite", 0.05, 0.20),
        _row("rf2047_tfilm_x2", "two_clippers", 0.0, 0.40),
        _row("cascade_rf2047_tfilm_x2", "two_clippers", 0.0, 0.18),
    ]


def test_mechanism_gate_selects_auxiliary_and_preserves_all_passing_families() -> None:
    gate = evaluate_arch_mechanism_gate(_passing_trajectories(), _protocol())
    assert gate["valid"] is True
    assert gate["passed"] is True
    assert gate["selected_auxiliary_weight"] == 0.05
    assert gate["rejected_candidates"] == []
    assert gate["relative_improvements"]["phys_det_dynamic"] == pytest.approx(0.15)
    assert gate["relative_improvements"]["phys_s6_dynamic"] == pytest.approx(0.22)
    assert gate["relative_improvements"]["cascade_two_clippers"] == pytest.approx(0.55)


def test_mechanism_gate_prunes_only_failed_idea_without_replacement() -> None:
    trajectories = _passing_trajectories()
    for row in trajectories:
        if row["family"] == "phys_det_tcn_x2" and row["system"] == "dynamic_composite":
            row["validation"]["esr"] = 0.95
    gate = evaluate_arch_mechanism_gate(trajectories, _protocol())
    assert gate["passed"] is True
    assert gate["candidate_checks"]["phys_det_tcn_x2"] is False
    assert gate["rejected_candidates"] == ["phys_det_tcn_x2"]
    assert "phys_det_tcn_x2" not in gate["eligible_candidates"]


def test_mechanism_gate_fails_closed_on_missing_duplicate_or_nonfinite_evidence() -> (
    None
):
    with pytest.raises(ArchGateError, match="count"):
        evaluate_arch_mechanism_gate(_passing_trajectories()[:-1], _protocol())
    duplicate = _passing_trajectories()
    duplicate[-1] = deepcopy(duplicate[-2])
    with pytest.raises(ArchGateError, match="duplicate"):
        evaluate_arch_mechanism_gate(duplicate, _protocol())
    nonfinite = _passing_trajectories()
    nonfinite[0]["validation"]["esr"] = float("nan")
    with pytest.raises(ArchGateError, match="finite"):
        evaluate_arch_mechanism_gate(nonfinite, _protocol())


def test_mechanism_gate_fails_stage_if_fewer_than_two_candidates_survive() -> None:
    trajectories = _passing_trajectories()
    for row in trajectories:
        row["validation"]["correlation"] = 0.1
    # Keep only the two-clipper reference admissible; cascade itself fails its
    # anti-collapse guard, so fewer than two candidates survive.
    for row in trajectories:
        if row["family"] == "rf2047_tfilm_x2":
            row["validation"]["correlation"] = 0.97
    gate = evaluate_arch_mechanism_gate(trajectories, _protocol())
    assert gate["eligible_candidates"] == ["rf2047_tfilm_x2"]
    assert gate["passed"] is False
