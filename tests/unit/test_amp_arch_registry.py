from __future__ import annotations

import json

import pytest

from fssr_nam.campaign.amp_arch_registry import (
    ArchRegistryError,
    append_gate_event,
    gate_decisions,
)


def _event(stage: str, status: str = "passed") -> dict[str, str]:
    return {
        "campaign_version": "AMP-QUALITY-ARCH-v1",
        "stage": stage,
        "status": status,
        "evidence_path": f"evidence/{stage}.json",
    }


def test_arch_registry_is_append_only_and_idempotent(tmp_path) -> None:
    path = tmp_path / "gates.jsonl"
    event = _event("preflight")
    assert append_gate_event(path, event) == event
    assert append_gate_event(path, event) == event
    assert gate_decisions(path) == {"preflight": "passed"}
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    with pytest.raises(ArchRegistryError, match="already recorded"):
        append_gate_event(path, _event("preflight", "failed"))


def test_arch_registry_rejects_malformed_or_duplicate_lines(tmp_path) -> None:
    path = tmp_path / "gates.jsonl"
    path.write_text(
        json.dumps(_event("preflight")) + "\n" + json.dumps(_event("preflight")),
        encoding="utf-8",
    )
    with pytest.raises(ArchRegistryError, match="duplicate"):
        gate_decisions(path)
