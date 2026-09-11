"""Append-only gate registry for AMP-COMPETENCE-ARCH-v2."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CAMPAIGN_VERSION = "AMP-COMPETENCE-ARCH-v2"


class ArchV2RegistryError(RuntimeError):
    """Raised when v2 gate evidence is malformed or duplicated."""


def read_gate_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ArchV2RegistryError(
                f"malformed architecture v2 gate line {line_number}: {error}"
            ) from error
        if not isinstance(event, dict):
            raise ArchV2RegistryError("architecture v2 gate events must be objects")
        events.append(event)
    return events


def gate_decisions(path: Path) -> dict[str, str]:
    decisions: dict[str, str] = {}
    for event in read_gate_events(path):
        stage = event.get("stage")
        status = event.get("status")
        if not isinstance(stage, str) or not isinstance(status, str):
            raise ArchV2RegistryError("v2 gate stage/status must be strings")
        if stage in decisions:
            raise ArchV2RegistryError(f"duplicate architecture v2 gate: {stage}")
        decisions[stage] = status
    return decisions


def append_gate_event(path: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    required = {"campaign_version", "stage", "status", "evidence_path"}
    if required - set(event):
        raise ArchV2RegistryError("v2 gate event is missing required fields")
    if event.get("campaign_version") != CAMPAIGN_VERSION:
        raise ArchV2RegistryError("architecture v2 campaign version changed")
    normalized = dict(event)
    for prior in read_gate_events(path):
        if prior.get("stage") == normalized["stage"]:
            if prior == normalized:
                return prior
            raise ArchV2RegistryError(
                f"architecture v2 gate {normalized['stage']} is already recorded"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        normalized, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized + "\n")
    return normalized
