from __future__ import annotations

from copy import deepcopy

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fssr_nam.statistics.quality_teacher import (
    QualityTeacherStatisticsError,
    hierarchical_confirmation_bootstrap,
)


def _metrics(esr: float, *, correlation: float = 0.96) -> dict[str, float]:
    return {
        "esr": esr,
        "l1": 0.4,
        "mrstft": 0.4,
        "mae": 1.04,
        "log_mel": 1.03,
        "envelope_transient": 1.02,
        "correlation": correlation,
        "gain_error": -0.1,
    }


def _rows() -> list[dict[str, object]]:
    rows = []
    for device, candidate_esr in (("rodent", 0.8), ("fuzzy_logic", 0.9)):
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
                        "candidate": _metrics(candidate_esr),
                    }
                )
    return rows


def test_bootstrap_uses_paired_file_seed_hierarchy_and_equal_device_weight() -> None:
    summary = hierarchical_confirmation_bootstrap(_rows(), replicates=100, seed=73)

    assert summary["resampling_order_within_device"] == ["files", "seeds"]
    assert summary["devices_equal_weight"] is True
    assert summary["observation_unit"] == "device_file_seed"
    assert summary["observation_count"] == 12
    assert summary["aggregate_esr_relative_improvement"] == pytest.approx(0.15)
    assert summary["aggregate_l1_plus_mrstft_relative_improvement"] == pytest.approx(
        0.20
    )
    assert summary["per_device"]["rodent"]["esr_relative_improvement"] == pytest.approx(
        0.20
    )
    assert summary["per_device"]["fuzzy_logic"][
        "esr_relative_improvement"
    ] == pytest.approx(0.10)
    assert summary["per_device"]["rodent"]["file_count"] == 2
    assert summary["maximum_per_device_secondary_relative_regressions"] == {
        "mae": pytest.approx(0.04),
        "log_mel": pytest.approx(0.03),
        "envelope_transient": pytest.approx(0.02),
    }
    assert summary["minimum_per_device_correlation"] == pytest.approx(0.96)
    assert summary["minimum_per_device_gain_error"] == pytest.approx(-0.1)
    assert summary["confidence_intervals_95"]["esr_relative_improvement"] == {
        "lower_95": pytest.approx(0.15),
        "upper_95": pytest.approx(0.15),
    }


@given(order=st.permutations(tuple(range(12))))
@settings(max_examples=10, deadline=None)
def test_bootstrap_is_row_order_invariant(order: tuple[int, ...]) -> None:
    rows = _rows()
    reference = hierarchical_confirmation_bootstrap(rows, replicates=40, seed=91)
    reordered = hierarchical_confirmation_bootstrap(
        [rows[index] for index in order], replicates=40, seed=91
    )
    assert reordered == reference


def test_bootstrap_rejects_unpaired_duplicate_and_nonfinite_rows() -> None:
    missing = _rows()
    missing.pop(0)
    with pytest.raises(QualityTeacherStatisticsError, match="every frozen seed"):
        hierarchical_confirmation_bootstrap(missing, replicates=2)

    duplicate = _rows()
    duplicate.append(deepcopy(duplicate[0]))
    with pytest.raises(QualityTeacherStatisticsError, match="duplicate"):
        hierarchical_confirmation_bootstrap(duplicate, replicates=2)

    nonfinite = _rows()
    nonfinite[0]["candidate"]["esr"] = float("nan")
    with pytest.raises(QualityTeacherStatisticsError, match="finite"):
        hierarchical_confirmation_bootstrap(nonfinite, replicates=2)


def test_combined_loss_must_match_components_when_both_are_recorded() -> None:
    rows = _rows()
    rows[0]["candidate"]["l1_plus_mrstft"] = 0.9
    with pytest.raises(QualityTeacherStatisticsError, match="disagrees"):
        hierarchical_confirmation_bootstrap(rows, replicates=2)


@pytest.mark.parametrize(("replicates", "seed"), [(0, 1), (True, 1), (2, -1)])
def test_bootstrap_control_values_are_validated(
    replicates: object, seed: object
) -> None:
    with pytest.raises(QualityTeacherStatisticsError, match="bootstrap"):
        hierarchical_confirmation_bootstrap(
            _rows(),
            replicates=replicates,
            seed=seed,  # type: ignore[arg-type]
        )
