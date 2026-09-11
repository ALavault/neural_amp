from __future__ import annotations

import pytest

from fssr_nam.statistics.r2 import (
    R2StatisticsError,
    hierarchical_confirmation_bootstrap,
    hierarchical_mushra_bootstrap,
)


def _confirmation_rows() -> list[dict[str, object]]:
    return [
        {
            "device": device,
            "seed": seed,
            "source": f"source{source}",
            "baseline_esr": 1.0,
            "candidate_esr": 0.8,
            "asr_reduction_db": 12.0,
        }
        for device in ("fulltone", "bigmuff", "blackstar", "ua1176")
        for seed in range(5)
        for source in range(2)
    ]


def _mushra_rows(retained_count: int = 20) -> list[dict[str, object]]:
    rows = []
    for participant_index in range(24):
        retained = participant_index < retained_count
        for excerpt_index in range(8):
            row: dict[str, object] = {
                "participant": f"p{participant_index:02d}",
                "excerpt": f"e{excerpt_index}",
                "retained": retained,
            }
            if retained:
                row.update({"candidate_score": 72.0, "a2_score": 60.0})
            rows.append(row)
    return rows


def test_confirmation_bootstrap_is_paired_seed_then_source_and_deterministic() -> None:
    rows = _confirmation_rows()
    first = hierarchical_confirmation_bootstrap(rows, replicates=200)
    second = hierarchical_confirmation_bootstrap(rows, replicates=200)
    assert first == second
    assert first["hierarchy"] == ["seed", "source"]
    assert first["windows_resampled"] is False
    assert first["probes_resampled"] is False
    assert first["esr_relative_improvement"] == pytest.approx(0.2)
    assert first["esr_confidence_interval_95"]["lower"] == pytest.approx(0.2)
    assert set(first["devices_won"]) == {
        "fulltone",
        "bigmuff",
        "blackstar",
        "ua1176",
    }
    assert first["asr_reduction_db"] == 12.0


def test_confirmation_bootstrap_rejects_window_level_pseudoreplication() -> None:
    rows = _confirmation_rows()
    rows.append(dict(rows[0]))
    with pytest.raises(R2StatisticsError, match="windows and probes"):
        hierarchical_confirmation_bootstrap(rows, replicates=10)


def test_confirmation_rejects_asr_repeated_as_source_observations() -> None:
    rows = _confirmation_rows()
    rows[0]["asr_reduction_db"] = 13.0
    with pytest.raises(R2StatisticsError, match="one probe aggregate"):
        hierarchical_confirmation_bootstrap(rows, replicates=10)


def test_mushra_bootstrap_uses_participants_then_excerpts() -> None:
    result = hierarchical_mushra_bootstrap(_mushra_rows(), replicates=200)
    assert result["recruited_participants"] == 24
    assert result["retained_participants"] == 20
    assert result["hierarchy"] == ["participant", "excerpt"]
    assert result["candidate_minus_a2_points"] == 12.0
    assert result["confidence_interval_95"]["lower"] == 12.0


def test_mushra_bootstrap_rejects_fewer_than_twenty_retained() -> None:
    with pytest.raises(R2StatisticsError, match="at least 20"):
        hierarchical_mushra_bootstrap(_mushra_rows(retained_count=19), replicates=10)


def test_mushra_bootstrap_rejects_different_excerpt_sets() -> None:
    rows = _mushra_rows()
    rows[0]["excerpt"] = "different"
    with pytest.raises(R2StatisticsError, match="same excerpts"):
        hierarchical_mushra_bootstrap(rows, replicates=10)
