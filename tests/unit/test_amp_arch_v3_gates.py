from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from fssr_nam.campaign.amp_arch_v3_gates import (
    ArchV3GateError,
    analyze_train_ranges,
    evaluate_representability_gate,
    evaluate_round_one_gate,
)
from fssr_nam.campaign.amp_quality_arch_v3 import INITIAL_FAMILIES
from fssr_nam.data.arch_v3_fixtures import PRIMARY_SYSTEMS, STRESS_SYSTEMS

ROOT = Path(__file__).resolve().parents[2]


def _protocol() -> dict[str, object]:
    return yaml.safe_load(
        (ROOT / "configs/amp_quality_arch_v3/protocol.yaml").read_text(encoding="utf-8")
    )


def _round_lock() -> dict[str, object]:
    return yaml.safe_load(
        (ROOT / "configs/amp_quality_arch_v3/round_1.yaml").read_text(encoding="utf-8")
    )


def _episodes() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    samples = 72_000
    phase = np.linspace(0.0, 120.0 * np.pi, samples, dtype=np.float64)
    dry = (0.15 * np.sin(phase))[None].astype(np.float32)
    primary = (0.35 * np.sin(phase) + 0.08 * np.sin(0.37 * phase))[None].astype(
        np.float32
    )
    stress = (0.75 * np.sign(np.sin(phase)))[None].astype(np.float32)
    return {system: (dry.copy(), primary.copy()) for system in PRIMARY_SYSTEMS} | {
        STRESS_SYSTEMS[0]: (dry.copy(), stress)
    }


def _memory_rows() -> list[dict[str, object]]:
    return [
        {
            "family": family,
            "target_collision_rms": 0.02 if family == INITIAL_FAMILIES[0] else 0.001,
            "memory_capable": family != INITIAL_FAMILIES[0],
        }
        for family in INITIAL_FAMILIES
    ]


def test_representability_gate_passes_three_unbounded_families() -> None:
    protocol = _protocol()
    ranges = analyze_train_ranges(_episodes(), protocol)
    gate = evaluate_representability_gate(ranges, _memory_rows(), protocol)
    assert gate["passed"] is True
    assert gate["training_eligible_families"] == list(INITIAL_FAMILIES)
    assert gate["memory_capable_families"] == list(INITIAL_FAMILIES[1:])
    assert all(
        row["recommended_initial_residual_scale"] > 0.0
        for row in gate["range_rows"]
        if row["primary"]
    )


def test_representability_gate_rejects_primary_rail_collapse() -> None:
    protocol = _protocol()
    ranges = analyze_train_ranges(_episodes(), protocol)
    collapsed = copy.deepcopy(ranges)
    row = next(item for item in collapsed if item["system"] == "dynamic_primary")
    row["near_peak_fraction"] = 0.9
    gate = evaluate_representability_gate(collapsed, _memory_rows(), protocol)
    assert gate["passed"] is False
    assert gate["primary_checks"]["dynamic_primary"]["not_rail_collapsed"] is False


def test_representability_gate_rejects_non_train_evidence() -> None:
    protocol = _protocol()
    ranges = analyze_train_ranges(_episodes(), protocol)
    ranges[0]["tier"] = "validation"
    with pytest.raises(ArchV3GateError, match="crossed train tier"):
        evaluate_representability_gate(ranges, _memory_rows(), protocol)


def _round_metrics(
    esr: float, gain: float = -0.1, corr: float = 0.95
) -> dict[str, float]:
    return {"esr": esr, "gain_error": gain, "correlation": corr}


def _passing_round_one() -> list[dict[str, object]]:
    final_esr = {
        "gainhead_micro_tcn_x2": 0.50,
        "slow_state_micro_tcn_x2": 0.80,
        "long_rf_tcn_x2": 0.60,
    }
    return [
        {
            "family": family,
            "system": system,
            "seed": seed,
            "checkpoints": [
                {
                    "update": update,
                    "internal_dev": _round_metrics(
                        final_esr[family] + (15_000 - update) / 100_000.0
                    ),
                }
                for update in (500, 1000, 2000, 5000, 10000, 15000)
            ],
        }
        for family in INITIAL_FAMILIES
        for system in PRIMARY_SYSTEMS
        for seed in (0, 1, 2)
    ]


def _family_parameters() -> dict[str, int]:
    return {
        "gainhead_micro_tcn_x2": 6659,
        "slow_state_micro_tcn_x2": 8201,
        "long_rf_tcn_x2": 14531,
    }


@given(order=st.permutations(tuple(range(27))))
@settings(max_examples=10, deadline=None)
def test_round_one_selection_is_paired_and_permutation_invariant(
    order: tuple[int, ...],
) -> None:
    rows = _passing_round_one()
    gate = evaluate_round_one_gate(
        [rows[index] for index in order],
        _protocol(),
        _round_lock(),
        _family_parameters(),
    )
    assert gate["passed"] is True
    assert gate["selected_family"] == "long_rf_tcn_x2"
    assert "gainhead_micro_tcn_x2" not in gate["eligible_families"]
    assert gate["comparison_performed_after_guards"] is True


def test_round_one_compares_only_after_each_family_passes_guards() -> None:
    rows = _passing_round_one()
    for row in rows:
        if row["family"] == "long_rf_tcn_x2":
            row["checkpoints"][-1]["internal_dev"]["correlation"] = 0.9
    gate = evaluate_round_one_gate(
        rows, _protocol(), _round_lock(), _family_parameters()
    )
    assert gate["selected_family"] == "slow_state_micro_tcn_x2"
    assert gate["family_results"]["long_rf_tcn_x2"]["guards_pass"] is False

    for row in rows:
        if row["family"] == "slow_state_micro_tcn_x2":
            row["checkpoints"][-1]["internal_dev"]["gain_error"] = -0.2
    failed = evaluate_round_one_gate(
        rows, _protocol(), _round_lock(), _family_parameters()
    )
    assert failed["passed"] is False
    assert failed["comparison_performed_after_guards"] is False


def test_round_one_tie_uses_parameter_count_not_input_order() -> None:
    rows = _passing_round_one()
    for row in rows:
        if row["family"] in {
            "slow_state_micro_tcn_x2",
            "long_rf_tcn_x2",
        }:
            row["checkpoints"][-1]["internal_dev"]["esr"] = 0.7
    gate = evaluate_round_one_gate(
        rows, _protocol(), _round_lock(), _family_parameters()
    )
    assert gate["tie_candidates"] == [
        "slow_state_micro_tcn_x2",
        "long_rf_tcn_x2",
    ]
    assert gate["selected_family"] == "slow_state_micro_tcn_x2"


def test_round_one_rejects_missing_duplicate_and_nonfinite_evidence() -> None:
    rows = _passing_round_one()
    with pytest.raises(ArchV3GateError, match="count"):
        evaluate_round_one_gate(
            rows[:-1], _protocol(), _round_lock(), _family_parameters()
        )
    duplicate = copy.deepcopy(rows)
    duplicate[-1] = copy.deepcopy(duplicate[-2])
    with pytest.raises(ArchV3GateError, match="duplicate"):
        evaluate_round_one_gate(
            duplicate, _protocol(), _round_lock(), _family_parameters()
        )
    nonfinite = copy.deepcopy(rows)
    nonfinite[0]["checkpoints"][-1]["internal_dev"]["esr"] = float("nan")
    with pytest.raises(ArchV3GateError, match="finite"):
        evaluate_round_one_gate(
            nonfinite, _protocol(), _round_lock(), _family_parameters()
        )
