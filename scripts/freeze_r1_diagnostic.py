#!/usr/bin/env python3
"""Freeze the R1 diagnostic protocol after a clean implementation commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml

from fssr_nam.reporting.r1_preflight import (
    CAMPAIGN_VERSION,
    DIAGNOSTIC_ACTIVE_PATH,
    DIAGNOSTIC_AMENDMENT_PATH,
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
QUARANTINED_RUN_ID = "r1_competence_bigmuff_wright_lstm64_wright_seed0_v1"
QUARANTINE_PATH = (
    ROOT / "experiments/quarantine" / QUARANTINED_RUN_ID / "QUARANTINE.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--amend-after-quarantine",
        action="store_true",
        help=(
            "create the sole versioned implementation amendment after the "
            "recorded unauthorized competence launch"
        ),
    )
    return parser.parse_args()


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_implementation_commit() -> str:
    status = git_output("status", "--porcelain")
    if status:
        raise RuntimeError("commit the complete R1 implementation before freezing")
    return git_output("rev-parse", "HEAD")


def _write_amendment(existing: dict[str, object]) -> None:
    if existing.get("status") != "frozen":
        raise RuntimeError("an amendment requires the immutable initial lock")
    base_digest = hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest()
    recorded_base = LOCK_DIGEST_PATH.read_text(encoding="utf-8").strip()
    if recorded_base != base_digest:
        raise RuntimeError("initial diagnostic lock digest mismatch")
    active_path = ROOT / DIAGNOSTIC_ACTIVE_PATH
    amendment_path = ROOT / DIAGNOSTIC_AMENDMENT_PATH
    amendment_digest_path = amendment_path.with_suffix(".sha256")
    if (
        active_path.exists()
        or amendment_path.exists()
        or amendment_digest_path.exists()
    ):
        raise RuntimeError("diagnostic lock amendment already exists and is immutable")
    if not QUARANTINE_PATH.is_file():
        raise RuntimeError("the post-lock invalid attempt is not quarantined")
    quarantine = json.loads(QUARANTINE_PATH.read_text(encoding="utf-8"))
    if (
        quarantine.get("disposition") != "quarantined_non_counted_protocol_attempt"
        or quarantine.get("canonical_seed0_run_id")
        != "r1_competence_bigmuff_lstm64_wright_seed0_v1"
    ):
        raise RuntimeError("quarantine evidence is inconsistent")
    ledger = (ROOT / ".codex_campaign/RUN_LEDGER.jsonl").read_text(encoding="utf-8")
    if not any(
        json.loads(line).get("run_id") == QUARANTINED_RUN_ID
        for line in ledger.splitlines()
        if line.strip()
    ):
        raise RuntimeError("immutable failed ledger evidence is absent")
    commit = _require_clean_implementation_commit()
    digest = protocol_digest(ROOT)
    if existing.get("protocol_sha256") != digest:
        raise RuntimeError("scientific protocol changed since the initial lock")
    amendment = {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "status": "frozen",
        "amendment_number": 1,
        "base_lock_path": str(LOCK_PATH.relative_to(ROOT)),
        "base_lock_sha256": base_digest,
        "reason": (
            "replace the nonconforming competence orchestration after an "
            "unauthorized post-lock launch; no valid counted trajectory existed"
        ),
        "quarantine_evidence": str(QUARANTINE_PATH.relative_to(ROOT)),
        "unauthorized_run_id": QUARANTINED_RUN_ID,
        "capacity_exclusion_basis": (
            "the attempt was never reserved or authorized by R1Executor and its "
            "identifier is outside the frozen 3/8/4/4 matrix; it remains recorded "
            "as an engineering protocol failure, not a diagnostic trajectory"
        ),
        "scientific_protocol_changed": False,
        "protocol_sha256": digest,
        "effective_implementation_commit": commit,
        "external_report_only_locked": True,
    }
    serialized = yaml.safe_dump(amendment, sort_keys=False)
    amendment_path.write_text(serialized, encoding="utf-8")
    amendment_digest = hashlib.sha256(amendment_path.read_bytes()).hexdigest()
    if not re.fullmatch(r"[0-9a-f]{64}", amendment_digest):  # pragma: no cover
        raise RuntimeError("invalid amendment digest")
    amendment_digest_path.write_text(amendment_digest + "\n", encoding="utf-8")
    active_path.write_text(DIAGNOSTIC_AMENDMENT_PATH + "\n", encoding="utf-8")
    print(f"diagnostic_protocol_sha256={digest}")
    print(f"diagnostic_amendment_sha256={amendment_digest}")


def main() -> None:
    args = parse_args()
    validate_historical_bytes(ROOT)
    validate_stage_configs(ROOT)
    validate_protocol_counts(ROOT)
    validate_external_freeze(ROOT)
    validate_submodule_pins(ROOT)
    validate_prepared_manifests(ROOT)
    existing = yaml.safe_load(LOCK_PATH.read_text(encoding="utf-8"))
    if existing.get("status") == "frozen":
        if not args.amend_after_quarantine:
            raise RuntimeError("diagnostic lock already exists and is immutable")
        _write_amendment(existing)
        return
    if args.amend_after_quarantine:
        raise RuntimeError("cannot amend a diagnostic protocol that is not frozen")
    commit = _require_clean_implementation_commit()
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
