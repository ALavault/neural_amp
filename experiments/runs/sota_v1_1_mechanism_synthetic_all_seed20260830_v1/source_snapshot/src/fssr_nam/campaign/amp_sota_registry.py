"""Append-only gate registry for AMP-SOTA-PROTOTYPE-v1."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CAMPAIGN_VERSION = "AMP-SOTA-PROTOTYPE-v1.1"


class SotaPrototypeRegistryError(RuntimeError):
    """Raised when prototype gate evidence is malformed or duplicated."""


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
            raise SotaPrototypeRegistryError(
                f"malformed prototype gate line {line_number}: {error}"
            ) from error
        if not isinstance(event, dict):
            raise SotaPrototypeRegistryError("prototype gate events must be objects")
        events.append(event)
    return events


def gate_decisions(path: Path) -> dict[str, str]:
    decisions: dict[str, str] = {}
    for event in read_gate_events(path):
        stage = event.get("stage")
        status = event.get("status")
        if not isinstance(stage, str) or not isinstance(status, str):
            raise SotaPrototypeRegistryError(
                "prototype gate stage/status must be strings"
            )
        if stage in decisions:
            raise SotaPrototypeRegistryError(f"duplicate prototype gate: {stage}")
        decisions[stage] = status
    return decisions


def append_gate_event(path: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    required = {"campaign_version", "stage", "status", "evidence_path"}
    if required - set(event):
        raise SotaPrototypeRegistryError(
            "prototype gate event is missing required fields"
        )
    if event.get("campaign_version") != CAMPAIGN_VERSION:
        raise SotaPrototypeRegistryError("prototype campaign version changed")
    normalized = dict(event)
    for prior in read_gate_events(path):
        if prior.get("stage") == normalized["stage"]:
            if prior == normalized:
                return prior
            raise SotaPrototypeRegistryError(
                f"prototype gate {normalized['stage']} is already recorded"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        normalized, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized + "\n")
    return normalized
