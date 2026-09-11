from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from fssr_nam.campaign.amp_arch_v2_gates import (
    ArchV2GateError,
    evaluate_comparison_gate,
    evaluate_competence_gate,
)
from fssr_nam.campaign.amp_competence_arch_v2 import (
    ALL_FAMILIES,
    CANDIDATE_FAMILIES,
    CHECKPOINTS,
    CONTROL_FAMILY,
    SEEDS,
    SYSTEMS,
)

ROOT = Path(__file__).resolve().parents[2]


def _protocol() -> dict[str, object]:
    return yaml.safe_load(
        (ROOT / "configs/amp_competence_arch_v2/protocol.yaml").read_text(
            encoding="utf-8"
        )
    )


def _metrics(esr: float, gain: float = -0.1, corr: float = 0.95) -> dict[str, float]:
    return {
        "esr": esr,
        "gain_error": gain,
        "correlation": corr,
        "prediction_rms": 0.5,
        "target_rms": 0.55,
    }


def _passing_competence() -> list[dict[str, object]]:
    esr_by_checkpoint = {
        500: 1.20,
        1000: 1.00,
        2000: 0.96,
        5000: 0.94,
        10000: 0.93,
        15000: 0.92,
    }
    return [
        {
            "family": CONTROL_FAMILY,
            "system": system,
            "seed": seed,
            "checkpoints": [
                {
                    "update": update,
                    "validation": _metrics(
                        esr,
                        gain=-0.25 if update == 500 else -0.1,
                    ),
                }
                for update, esr in esr_by_checkpoint.items()
            ],
        }
        for system in SYSTEMS
        for seed in SEEDS
    ]


@given(order=st.permutations(tuple(range(9))))
@settings(max_examples=15, deadline=None)
def test_competence_gate_is_permutation_invariant(order: tuple[int, ...]) -> None:
    rows = _passing_competence()
    gate = evaluate_competence_gate([rows[index] for index in order], _protocol())
    assert gate["passed"] is True
    assert gate["selected_budget_updates"] == 1000
    assert gate["confirming_checkpoint_updates"] == 2000
    assert gate["selected_plateau_relative_esr_improvement"] == pytest.approx(0.04)


def test_competence_gate_cannot_select_unconfirmed_final_checkpoint() -> None:
    rows = _passing_competence()
    for row in rows:
        for checkpoint in row["checkpoints"]:
            checkpoint["validation"]["gain_error"] = (
                -0.1 if checkpoint["update"] == CHECKPOINTS[-1] else -0.3
            )
    gate = evaluate_competence_gate(rows, _protocol())
    assert gate["passed"] is False
    assert gate["selected_budget_updates"] is None


def test_competence_gate_rejects_duplicate_missing_and_nonfinite_evidence() -> None:
    with pytest.raises(ArchV2GateError, match="count"):
        evaluate_competence_gate(_passing_competence()[:-1], _protocol())
    duplicate = _passing_competence()
    duplicate[-1] = deepcopy(duplicate[-2])
    with pytest.raises(ArchV2GateError, match="duplicate"):
        evaluate_competence_gate(duplicate, _protocol())
    nonfinite = _passing_competence()
    nonfinite[0]["checkpoints"][0]["validation"]["esr"] = float("nan")
    with pytest.raises(ArchV2GateError, match="finite"):
        evaluate_competence_gate(nonfinite, _protocol())


def _passing_comparison() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family in ALL_FAMILIES:
        esr = 0.8 if family == CANDIDATE_FAMILIES[0] else 1.0
        for system in SYSTEMS:
            for seed in SEEDS:
                rows.append(
                    {
                        "family": family,
                        "system": system,
                        "seed": seed,
                        "validation": _metrics(esr),
                    }
                )
    return rows


@given(order=st.permutations(tuple(range(45))))
@settings(max_examples=5, deadline=None)
def test_comparison_gate_is_paired_and_permutation_invariant(
    order: tuple[int, ...],
) -> None:
    rows = _passing_comparison()
    protocol = _protocol()
    protocol["comparison"]["bootstrap_replicates"] = 50
    gate = evaluate_comparison_gate([rows[index] for index in order], protocol)
    assert gate["passed"] is True
    assert gate["promoted_candidate"] == CANDIDATE_FAMILIES[0]
    result = gate["candidate_results"][CANDIDATE_FAMILIES[0]]
    assert result["paired_median_relative_esr_improvement"] == pytest.approx(0.2)
    assert result["lower_95_confidence_bound"] == pytest.approx(0.2)


def test_comparison_gate_requires_fresh_control_guards() -> None:
    rows = _passing_comparison()
    for row in rows:
        if row["family"] == CONTROL_FAMILY:
            row["validation"]["correlation"] = 0.89
    gate = evaluate_comparison_gate(rows, _protocol())
    assert gate["passed"] is False
    assert gate["control_guards_pass"] is False
