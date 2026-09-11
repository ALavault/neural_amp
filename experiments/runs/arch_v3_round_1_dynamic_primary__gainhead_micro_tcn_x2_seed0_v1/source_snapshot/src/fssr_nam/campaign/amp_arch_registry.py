"""Append-only gate registry for AMP-QUALITY-ARCH-v1."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .amp_quality_arch_v1 import CAMPAIGN_VERSION


class ArchRegistryError(RuntimeError):
    """Raised when architecture gate evidence is malformed or duplicated."""


def read_gate_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ArchRegistryError(
                f"malformed architecture gate line {line_number}: {error}"
            ) from error
        if not isinstance(event, dict):
            raise ArchRegistryError("architecture gate events must be objects")
        events.append(event)
    return events


def gate_decisions(path: Path) -> dict[str, str]:
    decisions: dict[str, str] = {}
    for event in read_gate_events(path):
        stage = event.get("stage")
        status = event.get("status")
        if not isinstance(stage, str) or not isinstance(status, str):
            raise ArchRegistryError("architecture gate stage/status must be strings")
        if stage in decisions:
            raise ArchRegistryError(f"duplicate architecture gate event: {stage}")
        decisions[stage] = status
    return decisions


def append_gate_event(path: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    required = {"campaign_version", "stage", "status", "evidence_path"}
    if required - set(event):
        raise ArchRegistryError("architecture gate event is missing required fields")
    if event.get("campaign_version") != CAMPAIGN_VERSION:
        raise ArchRegistryError("architecture gate campaign version changed")
    normalized = dict(event)
    for prior in read_gate_events(path):
        if prior.get("stage") == normalized["stage"]:
            if prior == normalized:
                return prior
            raise ArchRegistryError(
                f"architecture gate {normalized['stage']} is already recorded"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        normalized, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized + "\n")
    return normalized
