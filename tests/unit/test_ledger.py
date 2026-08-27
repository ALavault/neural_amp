from __future__ import annotations

from pathlib import Path

import pytest

from fssr_nam.reporting.ledger import append_run, read_runs


def _entry(run_id: str = "m0_test") -> dict[str, object]:
    return {
        "date": "2026-08-27T12:00:00+02:00",
        "run_id": run_id,
        "phase": "M0",
        "model": "identity",
        "device": "synthetic_identity",
        "seed": 0,
        "commit": "uncommitted",
        "config_sha256": "not_applicable",
        "data_sha256": "not_applicable",
        "status": "passed",
        "failure_reason": None,
        "results_path": "experiments/summaries/m0_identity",
    }


def test_append_run_round_trip(tmp_path: Path) -> None:
    ledger = tmp_path / "runs.jsonl"
    append_run(ledger, _entry())
    assert read_runs(ledger) == [_entry()]


def test_append_run_rejects_duplicate_identifier(tmp_path: Path) -> None:
    ledger = tmp_path / "runs.jsonl"
    append_run(ledger, _entry())
    with pytest.raises(ValueError, match="already exists"):
        append_run(ledger, _entry())


def test_append_run_requires_complete_provenance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        append_run(tmp_path / "runs.jsonl", {"run_id": "incomplete"})
