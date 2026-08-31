from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from fssr_nam.campaign.amp_quality_teacher_gates import (
    CANDIDATE_FAMILY,
    COMPARATOR_FAMILIES,
    CONTROL_FAMILY,
    QualityTeacherGateError,
    evaluate_objective_verdict,
    evaluate_slow_value_gate,
    select_global_comparator,
)

ROOT = Path(__file__).resolve().parents[2]


def _protocol() -> dict:
    return yaml.safe_load(
        (ROOT / ".codex_campaign/amp_quality_teacher_v1/PROTOCOL_LOCK.yaml").read_text(
            encoding="utf-8"
        )
    )


def _slow_metrics(esr: float, loss: float, secondary: float) -> dict[str, float]:
    return {
        "esr": esr,
        "l1": loss / 2.0,
        "mrstft": loss / 2.0,
        "mae": secondary,
        "log_mel": secondary,
        "envelope_transient": secondary,
    }


def _slow_rows(
    family: str, *, esr: float, loss: float, secondary: float
) -> list[dict[str, object]]:
    return [
        {
            "device": "ampeg",
            "seed": 0,
            "file": file_id,
            "family": family,
            "metrics": _slow_metrics(esr, loss, secondary),
        }
        for file_id in ("a.wav", "b.wav")
    ]


def test_slow_value_gate_accepts_inclusive_esr_and_secondary_boundaries() -> None:
    candidate = _slow_rows(CANDIDATE_FAMILY, esr=0.95, loss=0.9, secondary=1.05)
    control = _slow_rows(CONTROL_FAMILY, esr=1.0, loss=1.0, secondary=1.0)

    result = evaluate_slow_value_gate(candidate, control, _protocol())

    assert result["passed"] is True
    assert result["selected_candidate"] == CANDIDATE_FAMILY
    assert result["per_device"]["ampeg"][
        "esr_median_relative_improvement"
    ] == pytest.approx(0.05)
    assert result["per_device"]["ampeg"][
        "l1_plus_mrstft_median_relative_improvement"
    ] == pytest.approx(0.10)
    assert result["per_device"]["ampeg"]["secondary_median_relative_regressions"] == {
        "mae": pytest.approx(0.05),
        "log_mel": pytest.approx(0.05),
        "envelope_transient": pytest.approx(0.05),
    }


@pytest.mark.parametrize(
    ("esr", "loss", "secondary", "failed_check"),
    [
        (0.951, 0.9, 1.0, "esr_median_relative_improvement"),
        (0.9, 1.0, 1.0, "l1_plus_mrstft_improvement"),
        (0.9, 0.9, 1.051, "mae_nonregression"),
    ],
)
def test_slow_value_failure_promotes_fast_control(
    esr: float, loss: float, secondary: float, failed_check: str
) -> None:
    result = evaluate_slow_value_gate(
        _slow_rows(CANDIDATE_FAMILY, esr=esr, loss=loss, secondary=secondary),
        _slow_rows(CONTROL_FAMILY, esr=1.0, loss=1.0, secondary=1.0),
        _protocol(),
    )
    assert result["passed"] is False
    assert result["selected_candidate"] == CONTROL_FAMILY
    assert result["checks"][failed_check] is False


def test_slow_value_gate_rejects_unpaired_files() -> None:
    candidate = _slow_rows(CANDIDATE_FAMILY, esr=0.9, loss=0.9, secondary=1.0)
    control = _slow_rows(CONTROL_FAMILY, esr=1.0, loss=1.0, secondary=1.0)
    control[0]["file"] = "other.wav"
    with pytest.raises(QualityTeacherGateError, match="not paired"):
        evaluate_slow_value_gate(candidate, control, _protocol())


def _comparator_rows() -> list[dict[str, object]]:
    scores = {
        COMPARATOR_FAMILIES[0]: (1.0, 0.8),
        COMPARATOR_FAMILIES[1]: (0.995, 1.0),
        COMPARATOR_FAMILIES[2]: (1.02, 0.5),
    }
    return [
        {
            "family": family,
            "device": device,
            "seed": 0,
            "metrics": {
                "esr": scores[family][0],
                "l1": scores[family][1] / 2.0,
                "mrstft": scores[family][1] / 2.0,
            },
        }
        for family in COMPARATOR_FAMILIES
        for device in ("fulltone", "bigmuff", "ampeg")
    ]


def test_global_comparator_uses_equal_device_score_then_strict_tie_break() -> None:
    result = select_global_comparator(_comparator_rows(), _protocol())

    assert result["selected_comparator"] == COMPARATOR_FAMILIES[0]
    assert result["tie_break_applied"] is True
    assert result["tie_candidates"] == list(COMPARATOR_FAMILIES[:2])
    assert result["families"][COMPARATOR_FAMILIES[0]][
        "equal_device_weighted_median_esr"
    ] == pytest.approx(1.0)
    assert result["families"][COMPARATOR_FAMILIES[0]]["per_device"]["ampeg"][
        "median_l1_plus_mrstft"
    ] == pytest.approx(0.8)
    assert result["per_device_comparator_selection"] is False


def test_exact_one_percent_primary_gap_is_not_a_tie() -> None:
    rows = _comparator_rows()
    scores = {
        COMPARATOR_FAMILIES[0]: (1.0, 1.0),
        COMPARATOR_FAMILIES[1]: (1.01, 0.1),
        COMPARATOR_FAMILIES[2]: (1.02, 0.05),
    }
    for row in rows:
        esr, loss = scores[row["family"]]
        row["metrics"] = {"esr": esr, "l1": loss / 2, "mrstft": loss / 2}
    result = select_global_comparator(rows, _protocol())
    assert result["selected_comparator"] == COMPARATOR_FAMILIES[0]
    assert result["tie_candidates"] == [COMPARATOR_FAMILIES[0]]


@given(order=st.permutations(tuple(range(9))))
@settings(max_examples=10, deadline=None)
def test_comparator_selection_is_row_order_invariant(order: tuple[int, ...]) -> None:
    rows = _comparator_rows()
    expected = select_global_comparator(rows, _protocol())
    observed = select_global_comparator([rows[index] for index in order], _protocol())
    assert observed == expected


def _confirmation_rows(esr: float = 0.85) -> list[dict[str, object]]:
    rows = []
    for device in ("rodent", "fuzzy_logic"):
        for file_id in ("a.wav", "b.wav"):
            for seed in (0, 1, 2):
                rows.append(
                    {
                        "device": device,
                        "file": file_id,
                        "seed": seed,
                        "comparator": {
                            "esr": 1.0,
                            "l1": 0.5,
                            "mrstft": 0.5,
                            "mae": 1.0,
                            "log_mel": 1.0,
                            "envelope_transient": 1.0,
                        },
                        "candidate": {
                            "esr": esr,
                            "l1": 0.4,
                            "mrstft": 0.4,
                            "mae": 1.04,
                            "log_mel": 1.04,
                            "envelope_transient": 1.04,
                            "correlation": 0.95,
                            "gain_error": -0.1,
                        },
                    }
                )
    return rows


def _evidence(rows: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "leakage_detected": False,
        "provenance_complete": True,
        "execution_complete": True,
        "rows": _confirmation_rows() if rows is None else rows,
    }


def test_objective_verdict_goes_only_with_locked_bootstrap_and_all_checks() -> None:
    result = evaluate_objective_verdict(_evidence(), _protocol())

    assert result["valid"] is True
    assert result["passed"] is True
    assert result["verdict"] == "GO-OBJECTIVE-SOTA"
    assert result["statistics"]["replicates"] == 10_000
    assert result["statistics"]["seed"] == 20_260_831
    assert set(result["per_device"]) == {"rodent", "fuzzy_logic"}
    assert all(result["checks"].values())


def _summary() -> dict[str, object]:
    per_device = {
        device: {
            "file_count": 2,
            "observation_count": 6,
            "esr_relative_improvement": 0.15,
            "l1_plus_mrstft_relative_improvement": 0.2,
            "secondary_relative_regressions": {
                "mae": 0.04,
                "log_mel": 0.04,
                "envelope_transient": 0.04,
            },
            "correlation": 0.95,
            "gain_error": -0.1,
        }
        for device in ("rodent", "fuzzy_logic")
    }
    return {
        "per_device": per_device,
        "aggregate_esr_relative_improvement": 0.15,
        "aggregate_l1_plus_mrstft_relative_improvement": 0.2,
        "maximum_per_device_secondary_relative_regressions": {
            "mae": 0.04,
            "log_mel": 0.04,
            "envelope_transient": 0.04,
        },
        "minimum_per_device_correlation": 0.95,
        "minimum_per_device_gain_error": -0.1,
        "confidence_intervals_95": {
            "esr_relative_improvement": {"lower_95": 0.01, "upper_95": 0.2},
            "l1_plus_mrstft_relative_improvement": {
                "lower_95": 0.01,
                "upper_95": 0.3,
            },
        },
    }


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("aggregate_esr", "aggregate_esr_relative_improvement"),
        ("esr_lower", "aggregate_esr_bootstrap_lower_95"),
        ("device_esr", "per_confirmation_device_esr"),
        ("aggregate_loss", "aggregate_l1_plus_mrstft_improvement"),
        ("loss_lower", "aggregate_l1_plus_mrstft_bootstrap_lower_95"),
        ("secondary", "mae_nonregression"),
        ("correlation", "correlation"),
        ("gain", "gain_error"),
    ],
)
def test_objective_threshold_failure_is_no_go_not_invalid(
    monkeypatch: pytest.MonkeyPatch, mutation: str, failed_check: str
) -> None:
    summary = _summary()
    if mutation == "aggregate_esr":
        summary["aggregate_esr_relative_improvement"] = 0.099
    elif mutation == "esr_lower":
        summary["confidence_intervals_95"]["esr_relative_improvement"]["lower_95"] = 0.0
    elif mutation == "device_esr":
        summary["per_device"]["rodent"]["esr_relative_improvement"] = 0.0
    elif mutation == "aggregate_loss":
        summary["aggregate_l1_plus_mrstft_relative_improvement"] = 0.0
    elif mutation == "loss_lower":
        summary["confidence_intervals_95"]["l1_plus_mrstft_relative_improvement"][
            "lower_95"
        ] = 0.0
    elif mutation == "secondary":
        summary["maximum_per_device_secondary_relative_regressions"]["mae"] = 0.051
    elif mutation == "correlation":
        summary["minimum_per_device_correlation"] = 0.9
    else:
        summary["minimum_per_device_gain_error"] = -0.2
    monkeypatch.setattr(
        "fssr_nam.campaign.amp_quality_teacher_gates.hierarchical_confirmation_bootstrap",
        lambda *args, **kwargs: summary,
    )

    result = evaluate_objective_verdict(_evidence(), _protocol())

    assert result["valid"] is True
    assert result["passed"] is False
    assert result["verdict"] == "NO-GO-OBJECTIVE-SOTA"
    assert result["checks"][failed_check] is False


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("leakage", "leakage"),
        ("provenance", "provenance"),
        ("execution", "execution"),
        ("missing_provenance", "missing integrity"),
        ("malformed_rows", "every frozen seed"),
    ],
)
def test_integrity_or_completeness_failure_is_invalid(
    mutation: str, reason: str
) -> None:
    evidence = _evidence()
    if mutation == "leakage":
        evidence["leakage_detected"] = True
    elif mutation == "provenance":
        evidence["provenance_complete"] = False
    elif mutation == "execution":
        evidence["execution_complete"] = False
    elif mutation == "missing_provenance":
        evidence.pop("provenance_complete")
    else:
        evidence["rows"] = evidence["rows"][:-1]

    result = evaluate_objective_verdict(evidence, _protocol())

    assert result["valid"] is False
    assert result["passed"] is False
    assert result["verdict"] == "INVALID"
    assert any(reason in item for item in result["invalid_reasons"])


def test_gate_rejects_any_frozen_threshold_drift() -> None:
    protocol = deepcopy(_protocol())
    protocol["objective_verdict"]["aggregate_esr_relative_improvement_minimum"] = 0.099
    with pytest.raises(QualityTeacherGateError, match=r"must equal 0\.1"):
        evaluate_objective_verdict(_evidence(), protocol)
