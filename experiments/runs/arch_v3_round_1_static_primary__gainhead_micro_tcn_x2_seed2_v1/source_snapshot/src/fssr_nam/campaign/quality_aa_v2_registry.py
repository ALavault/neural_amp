"""Append-only gate registry for QUALITY-AA-v2."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .quality_aa_v2 import CAMPAIGN_VERSION


class QualityAAV2RegistryError(RuntimeError):
    """Raised when a v2 gate event is malformed or duplicated."""


def read_gate_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise QualityAAV2RegistryError(
                f"malformed v2 gate registry line {line_number}: {error}"
            ) from error
        if not isinstance(event, dict):
            raise QualityAAV2RegistryError("v2 gate events must be objects")
        events.append(event)
    return events


def gate_decisions(path: Path) -> dict[str, str]:
    decisions: dict[str, str] = {}
    for event in read_gate_events(path):
        stage = event.get("stage")
        status = event.get("status")
        if not isinstance(stage, str) or not isinstance(status, str):
            raise QualityAAV2RegistryError("v2 gate stage/status must be strings")
        if stage in decisions:
            raise QualityAAV2RegistryError(f"duplicate v2 gate event: {stage}")
        decisions[stage] = status
    return decisions


def append_gate_event(path: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    required = {"campaign_version", "stage", "status", "evidence_path"}
    if required - set(event):
        raise QualityAAV2RegistryError("v2 gate event is missing required fields")
    if event.get("campaign_version") != CAMPAIGN_VERSION:
        raise QualityAAV2RegistryError("v2 gate campaign version changed")
    normalized = dict(event)
    for prior in read_gate_events(path):
        if prior.get("stage") == normalized["stage"]:
            if prior == normalized:
                return prior
            raise QualityAAV2RegistryError(
                f"v2 gate {normalized['stage']} is already recorded"
            )
    serialized = json.dumps(
        normalized, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized + "\n")
    return normalized
