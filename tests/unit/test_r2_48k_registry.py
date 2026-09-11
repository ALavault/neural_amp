from __future__ import annotations

from pathlib import Path

import pytest

from fssr_nam.campaign.r2_48k_registry import (
    R248KRegistryError,
    append_gate_event,
    gate_decisions,
    read_gate_events,
)


def _event(status: str = "passed") -> dict[str, str]:
    return {
        "campaign_version": "FSSR-R2-48K-v1",
        "stage": "preflight",
        "status": status,
        "evidence_path": "stdout:preflight",
    }


def test_r2_48k_registry_is_append_only_and_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "GATE_LEDGER.jsonl"
    assert append_gate_event(path, _event()) == _event()
    assert append_gate_event(path, _event()) == _event()
    assert read_gate_events(path) == [_event()]
    assert gate_decisions(path) == {"preflight": "passed"}
    with pytest.raises(R248KRegistryError, match="already recorded"):
        append_gate_event(path, _event("failed"))


def test_r2_48k_registry_rejects_wrong_campaign(tmp_path: Path) -> None:
    event = _event()
    event["campaign_version"] = "FSSR-R2-v1"
    with pytest.raises(R248KRegistryError, match="campaign_version"):
        append_gate_event(tmp_path / "GATE_LEDGER.jsonl", event)
