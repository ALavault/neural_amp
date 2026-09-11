#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import yaml

from fssr_nam.campaign.r2 import validate_stage_authorization
from fssr_nam.campaign.r2_registry import (
    append_gate_event,
    gate_decisions,
)
from fssr_nam.data.r2_capture import (
    CaptureAuditError,
    load_and_audit_capture_manifest,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ledger = root / ".codex_campaign/r2/GATE_LEDGER.jsonl"
    validate_stage_authorization("capture", gate_decisions(ledger))
    config = yaml.safe_load(
        (root / "configs/data/r2_capture.yaml").read_text(encoding="utf-8")
    )
    manifest_path = root / config["manifest_path"]
    try:
        report = load_and_audit_capture_manifest(manifest_path, root=root)
    except CaptureAuditError as error:
        missing = not manifest_path.exists()
        report = {
            "format": "fssr-r2-capture-audit-v1",
            "status": "PENDING_EXTERNAL" if missing else "INVALID",
            "valid": False,
            "reason": str(error),
            "scientific_run_launched": False,
        }
        print(json.dumps(report, allow_nan=False, sort_keys=True))
        return 2 if missing else 1
    append_gate_event(
        ledger,
        {
            "campaign_version": "FSSR-R2-v1",
            "stage": "capture",
            "status": "passed",
            "evidence_path": config["manifest_path"],
        },
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
