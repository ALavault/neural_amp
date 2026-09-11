"""Append-only R2 gate registry; scientific evidence files remain immutable."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class R2RegistryError(RuntimeError):
    """Raised when a gate event would rewrite or contradict prior state."""


def read_gate_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise R2RegistryError(
                f"malformed R2 gate registry line {line_number}: {error}"
            ) from error
        if not isinstance(event, dict):
            raise R2RegistryError(
                f"R2 gate registry line {line_number} is not an object"
            )
        events.append(event)
    return events


def gate_decisions(path: Path) -> dict[str, str]:
    decisions: dict[str, str] = {}
    for event in read_gate_events(path):
        stage = event.get("stage")
        status = event.get("status")
        if not isinstance(stage, str) or not isinstance(status, str):
            raise R2RegistryError("R2 gate event stage/status must be strings")
        if stage in decisions:
            raise R2RegistryError(f"duplicate R2 gate event: {stage}")
        decisions[stage] = status
    return decisions


def append_gate_event(path: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    """Append one terminal gate event, accepting an identical idempotent replay."""
    required = {"campaign_version", "stage", "status", "evidence_path"}
    missing = required - set(event)
    if missing:
        raise R2RegistryError(f"R2 gate event missing fields: {sorted(missing)}")
    if event.get("campaign_version") != "FSSR-R2-v1":
        raise R2RegistryError("R2 gate event campaign_version is invalid")
    if not isinstance(event.get("stage"), str) or not event["stage"]:
        raise R2RegistryError("R2 gate event stage must be non-empty")
    if not isinstance(event.get("status"), str) or not event["status"]:
        raise R2RegistryError("R2 gate event status must be non-empty")
    normalized = dict(event)
    existing = read_gate_events(path)
    for prior in existing:
        if prior.get("stage") == normalized["stage"]:
            if prior == normalized:
                return prior
            raise R2RegistryError(
                f"R2 gate {normalized['stage']} is append-only and already recorded"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        normalized, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized + "\n")
    return normalized
