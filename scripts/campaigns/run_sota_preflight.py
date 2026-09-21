#!/usr/bin/env python3
"""Execute the metadata-only AMP-SOTA-PROTOTYPE-v1.1 preflight once."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from fssr_nam.campaign.amp_sota_gates import evaluate_preflight_gate
from fssr_nam.campaign.amp_sota_prototype_v1 import (
    CAMPAIGN_PATH,
    CAMPAIGN_VERSION,
    validate_repository_state,
    validate_stage_authorization,
)
from fssr_nam.campaign.amp_sota_registry import append_gate_event, gate_decisions
from fssr_nam.campaign.quality_aa_provenance import replace_json, write_new_json
from fssr_nam.data.fm9_capture import (
    FM9AuthorizationError,
    assert_fm9_operation_authorized,
    load_fm9_protocol,
)
from fssr_nam.data.sota_audit import audit_sota_data_metadata

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / CAMPAIGN_PATH
GATE_LEDGER = CAMPAIGN_DIR / "GATE_LEDGER.jsonl"
SUMMARY = ROOT / "experiments/summaries/amp_sota_prototype_v1_1/preflight.json"


def _run_validation(target: str) -> None:
    subprocess.run(["make", target], cwd=ROOT, check=True)


def main() -> int:
    decisions = gate_decisions(GATE_LEDGER)
    validate_stage_authorization("preflight", decisions)
    if "preflight" in decisions or SUMMARY.exists():
        raise RuntimeError("SOTA v1.1 preflight is already frozen")
    protocol = validate_repository_state(ROOT)
    parent = json.loads(
        (ROOT / ".codex_campaign/amp_quality_arch_v3/VERDICT.json").read_text(
            encoding="utf-8"
        )
    )
    predecessor = json.loads(
        (ROOT / ".codex_campaign/amp_sota_prototype_v1/VERDICT.json").read_text(
            encoding="utf-8"
        )
    )
    if parent.get("verdict") != "INVALID":
        raise RuntimeError("v3 INVALID verdict is required")
    if predecessor.get("verdict") != "SUPERSEDED-BEFORE-FIRST-SCIENTIFIC-RUN":
        raise RuntimeError("v1 supersession verdict is required")

    data_audit = audit_sota_data_metadata(ROOT)
    fm9 = load_fm9_protocol(ROOT)
    for operation in ("capture", "render_import"):
        try:
            assert_fm9_operation_authorized(fm9, operation)
        except FM9AuthorizationError:
            pass
        else:
            raise RuntimeError(f"FM9 {operation} unexpectedly authorized")

    _run_validation("data-audit")
    _run_validation("test")
    _run_validation("lint")
    evidence = {
        "campaign_version": CAMPAIGN_VERSION,
        "parent_campaign": "AMP-QUALITY-ARCH-v3",
        "parent_verdict": "INVALID",
        "predecessor_campaign": "AMP-SOTA-PROTOTYPE-v1",
        "predecessor_verdict": predecessor["verdict"],
        "protocol_frozen": True,
        "data_audit_passed": True,
        "tests_passed": True,
        "lint_passed": True,
        "public_license_audit_passed": data_audit["passed"],
        "source_file_disjoint_splits_verified": data_audit["passed"],
        "physical_audio_samples_read": 0,
        "confirmation_audio_samples_read": 0,
        "fm9_audio_samples_read": 0,
        "failed_or_invalid_run_resumed": False,
        "commercial_claim_supported": False,
        "data_audit": data_audit,
    }
    gate = evaluate_preflight_gate(evidence, protocol)
    evidence["gate"] = gate
    status = "passed" if gate["passed"] else "failed"
    evidence["status"] = status
    write_new_json(SUMMARY, evidence)
    append_gate_event(
        GATE_LEDGER,
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "preflight",
            "status": status,
            "evidence_path": str(SUMMARY.relative_to(ROOT)),
        },
    )
    maturity_path = CAMPAIGN_DIR / "MATURITY.json"
    maturity = json.loads(maturity_path.read_text(encoding="utf-8"))
    maturity["current_stage"] = (
        "preflight_passed" if gate["passed"] else "preflight_failed"
    )
    if not gate["passed"]:
        maturity["status"] = "terminal_no_go"
        maturity["verdict"] = "NO-GO-PREFLIGHT"
    replace_json(maturity_path, maturity)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
