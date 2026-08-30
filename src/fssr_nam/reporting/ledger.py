"""Append-only JSON Lines ledger for training and benchmark runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = frozenset(
    {
        "date",
        "run_id",
        "phase",
        "model",
        "device",
        "seed",
        "commit",
        "config_sha256",
        "data_sha256",
        "status",
        "failure_reason",
        "results_path",
    }
)


def read_runs(path: Path) -> list[dict[str, Any]]:
    """Read ledger entries, ignoring blank lines but not malformed JSON."""
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_run(path: Path, entry: Mapping[str, Any]) -> None:
    """Append a validated entry while rejecting a reused run identifier."""
    missing = sorted(REQUIRED_FIELDS - set(entry))
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")
    run_id = entry["run_id"]
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if any(existing["run_id"] == run_id for existing in read_runs(path)):
        raise ValueError(f"run_id already exists: {run_id}")

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dict(entry), sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as ledger:
        ledger.write(serialized + "\n")


def append_run_once_or_equal(path: Path, entry: Mapping[str, Any]) -> None:
    """Append a run, accepting only an identical existing recovery entry."""
    normalized = dict(entry)
    matches = [
        row for row in read_runs(path) if row.get("run_id") == entry.get("run_id")
    ]
    if matches:
        if len(matches) == 1 and matches[0] == normalized:
            return
        raise ValueError(
            f"run_id already exists with different evidence: {entry.get('run_id')}"
        )
    append_run(path, normalized)
