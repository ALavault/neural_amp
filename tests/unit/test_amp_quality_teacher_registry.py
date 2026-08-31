from __future__ import annotations

import json
from pathlib import Path

import pytest

from fssr_nam.campaign.amp_quality_teacher_registry import (
    REQUIRED_RUN_ARTIFACTS,
    QualityTeacherRegistryError,
    append_gate_event,
    append_run_event,
    gate_decisions,
    read_gate_events,
    read_run_events,
    validate_campaign_run_registrations,
    validate_run_artifacts,
)

RUN_ID = "quality_teacher_v1_development_fulltone_s4_tfilm_wavenet_x2_teacher_seed0_v1"


def _protocol() -> dict[str, object]:
    return {"run_artifacts": {"required": list(REQUIRED_RUN_ARTIFACTS)}}


def _write_run_artifacts(
    root: Path,
    run_id: str = RUN_ID,
    *,
    seed: int = 0,
    status: str = "completed",
    failure_reason: str = "",
) -> Path:
    run_dir = root / "experiments" / "runs" / run_id
    records: dict[str, object] = {
        name: {} for name in REQUIRED_RUN_ARTIFACTS if name.endswith(".json")
    }
    records["seed.json"] = {"seed": seed}
    records["status.json"] = {
        "status": status,
        "failure_reason": failure_reason,
    }
    for name, record in records.items():
        path = run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
    (run_dir / "resolved_config.yaml").write_text("{}\n", encoding="utf-8")
    return run_dir


def _run_event(
    run_id: str = RUN_ID,
    *,
    status: str = "completed",
    failure_reason: str = "",
) -> dict[str, object]:
    return {
        "date": "2026-08-31T12:00:00+02:00",
        "run_id": run_id,
        "phase": "development",
        "model": "s4_tfilm_wavenet_x2_teacher",
        "device": "fulltone",
        "seed": 0,
        "commit": "uncommitted-test-fixture",
        "config_sha256": "not_computed_test_fixture",
        "data_sha256": "not_computed_test_fixture",
        "status": status,
        "failure_reason": failure_reason,
        "results_path": f"experiments/runs/{run_id}",
    }


def _gate_event(status: str = "passed") -> dict[str, str]:
    return {
        "campaign_version": "AMP-QUALITY-TEACHER-v1",
        "stage": "preflight",
        "status": status,
        "evidence_path": "reports/preflight.json",
    }


def test_gate_events_are_append_only_and_idempotent(tmp_path: Path) -> None:
    ledger = tmp_path / "GATE_LEDGER.jsonl"
    assert append_gate_event(ledger, _gate_event()) == _gate_event()
    assert append_gate_event(ledger, _gate_event()) == _gate_event()
    assert read_gate_events(ledger) == [_gate_event()]
    assert gate_decisions(ledger) == {"preflight": "passed"}

    with pytest.raises(QualityTeacherRegistryError, match="append-only"):
        append_gate_event(ledger, _gate_event("failed"))
    assert read_gate_events(ledger) == [_gate_event()]


def test_run_artifact_layout_is_exact_and_run_id_is_parsed(tmp_path: Path) -> None:
    run_dir = _write_run_artifacts(tmp_path)
    artifacts = validate_run_artifacts(tmp_path, RUN_ID, _protocol())
    assert tuple(artifacts) == REQUIRED_RUN_ARTIFACTS
    assert all(path.is_relative_to(run_dir) for path in artifacts.values())

    (run_dir / "predictions/manifest.json").unlink()
    with pytest.raises(QualityTeacherRegistryError, match="predictions/manifest"):
        validate_run_artifacts(tmp_path, RUN_ID, _protocol())
    with pytest.raises(QualityTeacherRegistryError, match="invalid campaign run_id"):
        validate_run_artifacts(tmp_path, "../outside", _protocol())

    drifted = _protocol()
    drifted["run_artifacts"]["required"].remove("history.json")  # type: ignore[index, union-attr]
    with pytest.raises(QualityTeacherRegistryError, match="PROTOCOL_LOCK"):
        validate_run_artifacts(tmp_path, RUN_ID, drifted)


@pytest.mark.parametrize("status", ["completed", "failed", "invalid"])
def test_all_terminal_run_outcomes_are_registered(tmp_path: Path, status: str) -> None:
    reason = "terminal fixture" if status != "completed" else ""
    _write_run_artifacts(
        tmp_path,
        status=status,
        failure_reason=reason,
    )
    ledger = tmp_path / ".codex_campaign" / "RUN_LEDGER.jsonl"
    event = _run_event(status=status, failure_reason=reason)

    assert append_run_event(ledger, event, tmp_path, _protocol()) == event
    assert append_run_event(ledger, event, tmp_path, _protocol()) == event
    assert read_run_events(ledger) == [event]
    assert validate_campaign_run_registrations(tmp_path, ledger, _protocol()) == (
        RUN_ID,
    )


def test_run_events_cannot_rewrite_or_escape_run_directory(tmp_path: Path) -> None:
    _write_run_artifacts(tmp_path)
    ledger = tmp_path / ".codex_campaign" / "RUN_LEDGER.jsonl"
    event = _run_event()
    append_run_event(ledger, event, tmp_path, _protocol())

    changed = dict(event, date="2026-08-31T12:01:00+02:00")
    with pytest.raises(QualityTeacherRegistryError, match="append-only"):
        append_run_event(ledger, changed, tmp_path, _protocol())

    escaped = dict(event, results_path=f"../{RUN_ID}")
    with pytest.raises(QualityTeacherRegistryError, match="results_path"):
        append_run_event(
            tmp_path / "other-ledger.jsonl", escaped, tmp_path, _protocol()
        )


def test_unregistered_failed_run_is_rejected(tmp_path: Path) -> None:
    _write_run_artifacts(
        tmp_path,
        status="failed",
        failure_reason="record this failure",
    )
    ledger = tmp_path / ".codex_campaign" / "RUN_LEDGER.jsonl"
    with pytest.raises(QualityTeacherRegistryError, match="missing ledger events"):
        validate_campaign_run_registrations(tmp_path, ledger, _protocol())
