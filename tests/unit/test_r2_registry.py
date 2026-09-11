from __future__ import annotations

from pathlib import Path

import pytest

from fssr_nam.campaign.r2_registry import (
    R2RegistryError,
    append_gate_event,
    gate_decisions,
    read_gate_events,
)


def _event(status: str = "passed") -> dict[str, str]:
    return {
        "campaign_version": "FSSR-R2-v1",
        "stage": "preflight",
        "status": status,
        "evidence_path": "stdout:preflight",
    }


def test_r2_gate_registry_is_append_only_and_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "GATE_LEDGER.jsonl"
    assert append_gate_event(path, _event()) == _event()
    assert append_gate_event(path, _event()) == _event()
    assert read_gate_events(path) == [_event()]
    assert gate_decisions(path) == {"preflight": "passed"}
    with pytest.raises(R2RegistryError, match="append-only"):
        append_gate_event(path, _event("failed"))
    assert read_gate_events(path) == [_event()]


def test_r2_gate_registry_rejects_malformed_or_duplicate_lines(tmp_path: Path) -> None:
    path = tmp_path / "GATE_LEDGER.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(R2RegistryError, match="malformed"):
        read_gate_events(path)
    duplicate = tmp_path / "duplicate.jsonl"
    line = (
        '{"campaign_version":"FSSR-R2-v1","evidence_path":"x",'
        '"stage":"capture","status":"passed"}\n'
    )
    duplicate.write_text(line + line, encoding="utf-8")
    with pytest.raises(R2RegistryError, match="duplicate"):
        gate_decisions(duplicate)
