from __future__ import annotations

from copy import deepcopy

import pytest

from fssr_nam.statistics.r2_48k import (
    R248KStatisticsError,
    hierarchical_confirmation_bootstrap,
    hierarchical_mushra_bootstrap,
)


def _confirmation_rows() -> list[dict]:
    sources = {
        "fulltone": ["fulltone-test"],
        "bigmuff": ["bigmuff-test"],
        "blackstar": ["blackstar-test"],
        "ua1176": ["ua-bass2", "ua-gtr7"],
    }
    rows = []
    for device, device_sources in sources.items():
        role = (
            "prospective_primary"
            if device in {"blackstar", "ua1176"}
            else "historical_development"
        )
        candidate = 0.16 if role == "prospective_primary" else 0.18
        for seed in range(5):
            for source in device_sources:
                rows.append(
                    {
                        "device": device,
                        "seed": seed,
                        "source": source,
                        "evidence_role": role,
                        "baseline_esr": 0.20,
                        "candidate_esr": candidate,
                        "physical_asr_used_for_decision": False,
                    }
                )
    return rows


def _mushra_rows() -> list[dict]:
    devices = ("fulltone", "bigmuff", "blackstar", "ua1176")
    rows = []
    for participant_index in range(24):
        retained = participant_index < 20
        for device in devices:
            primary = device in {"blackstar", "ua1176"}
            for excerpt_index in range(2):
                row = {
                    "participant": f"p{participant_index:02d}",
                    "excerpt": f"{device}-{excerpt_index}",
                    "device": device,
                    "retained": retained,
                    "evidence_role": (
                        "prospective_primary" if primary else "development_secondary"
                    ),
                }
                if retained:
                    row["a2_score"] = 50.0
                    row["candidate_score"] = 62.0 if primary else 30.0
                rows.append(row)
    return rows


def test_primary_confirmation_interval_excludes_historical_development() -> None:
    rows = _confirmation_rows()
    summary = hierarchical_confirmation_bootstrap(rows, replicates=300)
    assert summary["heldout_esr_relative_improvement"] == pytest.approx(0.20)
    assert summary["development_esr_relative_improvement_descriptive"] == pytest.approx(
        0.10
    )
    assert summary["primary_devices"] == ["blackstar", "ua1176"]
    assert summary["source_counts"]["blackstar"] == 1
    assert summary["source_counts"]["ua1176"] == 2
    assert summary["physical_asr_in_primary_decision"] is False

    changed_dev = deepcopy(rows)
    for row in changed_dev:
        if row["evidence_role"] == "historical_development":
            row["candidate_esr"] = 0.40
    changed = hierarchical_confirmation_bootstrap(changed_dev, replicates=300)
    assert (
        changed["heldout_esr_relative_improvement"]
        == summary["heldout_esr_relative_improvement"]
    )
    assert (
        changed["heldout_esr_confidence_interval_95"]
        == summary["heldout_esr_confidence_interval_95"]
    )


def test_confirmation_rejects_windows_or_mismatched_source_sets() -> None:
    rows = _confirmation_rows()
    rows.append(dict(rows[0]))
    with pytest.raises(R248KStatisticsError, match="windows are not observations"):
        hierarchical_confirmation_bootstrap(rows, replicates=2)
    rows = _confirmation_rows()
    rows.pop(
        next(
            index
            for index, row in enumerate(rows)
            if row["device"] == "ua1176"
            and row["seed"] == 4
            and row["source"] == "ua-gtr7"
        )
    )
    with pytest.raises(R248KStatisticsError, match="source set"):
        hierarchical_confirmation_bootstrap(rows, replicates=2)


def test_primary_mushra_interval_uses_only_blackstar_and_ua() -> None:
    rows = _mushra_rows()
    summary = hierarchical_mushra_bootstrap(rows, replicates=300)
    assert summary["primary_candidate_minus_a2_points"] == pytest.approx(12.0)
    assert summary[
        "development_candidate_minus_a2_points_descriptive"
    ] == pytest.approx(-20.0)
    assert summary["primary_excerpts_per_participant"] == 4
    assert summary["development_excluded_from_primary_interval"] is True

    changed_dev = deepcopy(rows)
    for row in changed_dev:
        if row["evidence_role"] == "development_secondary" and row["retained"]:
            row["candidate_score"] = 100.0
    changed = hierarchical_mushra_bootstrap(changed_dev, replicates=300)
    assert (
        changed["primary_candidate_minus_a2_points"]
        == summary["primary_candidate_minus_a2_points"]
    )
    assert (
        changed["primary_confidence_interval_95"]
        == summary["primary_confidence_interval_95"]
    )
