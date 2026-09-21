#!/usr/bin/env python3
"""Freeze AMP-COMPETENCE-ARCH-v2 before any scientific run."""

from __future__ import annotations

import json
from pathlib import Path

from fssr_nam.campaign.amp_arch_v2_registry import append_gate_event, gate_decisions
from fssr_nam.campaign.amp_competence_arch_v2 import (
    CAMPAIGN_PATH,
    CAMPAIGN_VERSION,
    validate_repository_state,
    validate_stage_authorization,
)
from fssr_nam.campaign.quality_aa_provenance import write_new_json, write_new_text
from fssr_nam.reporting.ledger import read_runs

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / CAMPAIGN_PATH
GATE_LEDGER = CAMPAIGN_DIR / "GATE_LEDGER.jsonl"
GLOBAL_LEDGER = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
SUMMARY = ROOT / "experiments/summaries/amp_competence_arch_v2/preflight.json"


def main() -> int:
    decisions = gate_decisions(GATE_LEDGER)
    validate_stage_authorization("preflight", decisions)
    protocol = validate_repository_state(ROOT)
    lock_path = CAMPAIGN_DIR / "PROTOCOL_LOCK.yaml"
    if lock_path.exists() or SUMMARY.exists():
        raise RuntimeError("v2 preflight evidence is already frozen")
    if any(
        entry["run_id"].startswith("arch_v2_") for entry in read_runs(GLOBAL_LEDGER)
    ):
        raise RuntimeError("v2 preflight found a prior scientific ledger entry")
    run_root = ROOT / "experiments/runs"
    if any(run_root.glob("arch_v2_*")):
        raise RuntimeError("v2 preflight found an unregistered run directory")
    source_text = (ROOT / "configs/amp_competence_arch_v2/protocol.yaml").read_text(
        encoding="utf-8"
    )
    write_new_text(lock_path, source_text)
    summary = {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "stage": "preflight",
        "status": "passed",
        "v1_terminal_verdict": "NO-GO-ARCH",
        "v1_runs_resumed": 0,
        "v1_runs_retuned": 0,
        "v2_scientific_runs_present": 0,
        "candidate_runs_authorized": False,
        "physical_audio_samples_read": 0,
        "sealed_outputs_accessed": {"blackstar": False, "ua1176": False},
        "external_report_only_locked": True,
        "competence_trajectory_count": protocol["competence"]["trajectory_count"],
        "candidate_family_count": len(protocol["comparison"]["candidate_families"]),
    }
    write_new_json(SUMMARY, summary)
    append_gate_event(
        GATE_LEDGER,
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "preflight",
            "status": "passed",
            "evidence_path": str(SUMMARY.relative_to(ROOT)),
        },
    )
    maturity_path = CAMPAIGN_DIR / "MATURITY.json"
    maturity = json.loads(maturity_path.read_text(encoding="utf-8"))
    maturity["current_stage"] = "preflight_passed"
    maturity_path.write_text(
        json.dumps(maturity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
