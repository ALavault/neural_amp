#!/usr/bin/env python3
"""Freeze the R1 diagnostic protocol after a clean implementation commit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from fssr_nam.reporting.r1_preflight import (
    protocol_digest,
    validate_external_freeze,
    validate_historical_bytes,
    validate_prepared_manifests,
    validate_protocol_counts,
    validate_stage_configs,
    validate_submodule_pins,
)

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / ".codex_campaign/r1/DIAGNOSTIC_LOCK.yaml"
LOCK_DIGEST_PATH = ROOT / ".codex_campaign/r1/DIAGNOSTIC_LOCK.sha256"
FREEZE_PATH = ROOT / ".codex_campaign/r1/EXTERNAL_FREEZE.json"


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    validate_historical_bytes(ROOT)
    validate_stage_configs(ROOT)
    validate_protocol_counts(ROOT)
    validate_external_freeze(ROOT)
    validate_submodule_pins(ROOT)
    validate_prepared_manifests(ROOT)
    existing = yaml.safe_load(LOCK_PATH.read_text(encoding="utf-8"))
    if existing.get("status") == "frozen":
        raise RuntimeError("diagnostic lock already exists and is immutable")
    status = git_output("status", "--porcelain")
    if status:
        raise RuntimeError("commit the complete R1 implementation before freezing")
    commit = git_output("rev-parse", "HEAD")
    existing["status"] = "frozen"
    existing["implementation_commit"] = commit
    existing["protocol_sha256"] = protocol_digest(ROOT)
    serialized = yaml.safe_dump(existing, sort_keys=False)
    LOCK_PATH.write_text(serialized, encoding="utf-8")
    lock_digest = hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest()
    LOCK_DIGEST_PATH.write_text(lock_digest + "\n", encoding="utf-8")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("diagnostic_protocol_frozen") is not False:
        raise RuntimeError("unexpected diagnostic freeze state")
    freeze["diagnostic_protocol_frozen"] = True
    FREEZE_PATH.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(f"diagnostic_protocol_sha256={existing['protocol_sha256']}")


if __name__ == "__main__":
    main()
