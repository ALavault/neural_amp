#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from fssr_nam.campaign.r2_48k import (
    CAMPAIGN_VERSION,
    validate_stage_authorization,
)
from fssr_nam.campaign.r2_48k_registry import append_gate_event, gate_decisions
from fssr_nam.data.r2_48k import audit_r2_48k_archive


def _write_immutable_report(path: Path, report: dict) -> None:
    serialized = json.dumps(report, allow_nan=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError("R2-48K data-audit evidence is immutable and changed")
        return
    path.write_text(serialized, encoding="utf-8")


def _mark_data_passed(root: Path) -> None:
    path = root / ".codex_campaign/r2_48k/MATURITY.json"
    maturity = json.loads(path.read_text(encoding="utf-8"))
    gates = maturity.get("gates", {})
    if gates.get("data") not in {"pending", "passed"}:
        raise RuntimeError("R2-48K data maturity cannot be advanced")
    gates["data"] = "passed"
    if gates.get("mechanism") == "locked":
        gates["mechanism"] = "pending_measurement_evidence"
    maturity["current_stage"] = "mechanism"
    maturity["status"] = "ready_for_synthetic_mechanism_no_scientific_run"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(maturity, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ledger = root / ".codex_campaign/r2_48k/GATE_LEDGER.jsonl"
    validate_stage_authorization("data", gate_decisions(ledger))
    report = audit_r2_48k_archive(root)
    report_path = root / ".codex_campaign/r2_48k/DATA_AUDIT.json"
    _write_immutable_report(report_path, report)
    append_gate_event(
        ledger,
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "data",
            "status": "passed",
            "evidence_path": ".codex_campaign/r2_48k/DATA_AUDIT.json",
        },
    )
    _mark_data_passed(root)
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
