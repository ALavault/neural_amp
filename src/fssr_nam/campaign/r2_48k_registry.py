"""Append-only gate registry for FSSR-R2-48K-v1."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fssr_nam.campaign.r2_48k import CAMPAIGN_VERSION


class R248KRegistryError(RuntimeError):
    """Raised when a gate event would contradict immutable campaign state."""


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
            raise R248KRegistryError(
                f"malformed R2-48K gate registry line {line_number}: {error}"
            ) from error
        if not isinstance(event, dict):
            raise R248KRegistryError(
                f"R2-48K gate registry line {line_number} is not an object"
            )
        events.append(event)
    return events


def gate_decisions(path: Path) -> dict[str, str]:
    decisions: dict[str, str] = {}
    for event in read_gate_events(path):
        stage = event.get("stage")
        status = event.get("status")
        if not isinstance(stage, str) or not isinstance(status, str):
            raise R248KRegistryError("R2-48K gate stage/status must be strings")
        if stage in decisions:
            raise R248KRegistryError(f"duplicate R2-48K gate event: {stage}")
        decisions[stage] = status
    return decisions


def append_gate_event(path: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    """Append one terminal event, accepting only an identical replay."""
    required = {"campaign_version", "stage", "status", "evidence_path"}
    missing = required - set(event)
    if missing:
        raise R248KRegistryError(f"R2-48K gate event missing fields: {sorted(missing)}")
    if event.get("campaign_version") != CAMPAIGN_VERSION:
        raise R248KRegistryError("R2-48K gate event campaign_version is invalid")
    if not isinstance(event.get("stage"), str) or not event["stage"]:
        raise R248KRegistryError("R2-48K gate stage must be non-empty")
    if not isinstance(event.get("status"), str) or not event["status"]:
        raise R248KRegistryError("R2-48K gate status must be non-empty")
    normalized = dict(event)
    for prior in read_gate_events(path):
        if prior.get("stage") == normalized["stage"]:
            if prior == normalized:
                return prior
            raise R248KRegistryError(
                f"R2-48K gate {normalized['stage']} is already recorded"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        normalized, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized + "\n")
    return normalized
