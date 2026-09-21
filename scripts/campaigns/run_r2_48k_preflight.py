#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from fssr_nam.campaign.r2_48k import CAMPAIGN_VERSION
from fssr_nam.campaign.r2_48k_registry import append_gate_event
from fssr_nam.reporting.r2_48k_preflight import run_r2_48k_preflight


def _write_immutable_report(path: Path, report: dict) -> None:
    serialized = json.dumps(report, allow_nan=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError("R2-48K preflight evidence is immutable and changed")
        return
    path.write_text(serialized, encoding="utf-8")


def _mark_preflight_passed(root: Path) -> None:
    path = root / ".codex_campaign/r2_48k/MATURITY.json"
    maturity = json.loads(path.read_text(encoding="utf-8"))
    gates = maturity.get("gates", {})
    if gates.get("preflight") not in {"pending", "passed"}:
        raise RuntimeError("R2-48K preflight maturity cannot be advanced")
    gates["preflight"] = "passed"
    if gates.get("data") != "passed":
        gates["data"] = "pending"
        maturity["current_stage"] = "data"
        maturity["status"] = "preflight_passed_data_audit_pending_no_scientific_run"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(maturity, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    report = run_r2_48k_preflight(root)
    report_path = root / ".codex_campaign/r2_48k/PREFLIGHT.json"
    _write_immutable_report(report_path, report)
    append_gate_event(
        root / ".codex_campaign/r2_48k/GATE_LEDGER.jsonl",
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "preflight",
            "status": "passed",
            "evidence_path": ".codex_campaign/r2_48k/PREFLIGHT.json",
        },
    )
    _mark_preflight_passed(root)
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
