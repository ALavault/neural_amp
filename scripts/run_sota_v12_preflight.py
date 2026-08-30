#!/usr/bin/env python3
"""Execute the metadata-only AMP-SOTA-PROTOTYPE-v1.2 preflight once."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch

from fssr_nam.campaign.amp_sota_prototype_v1_2 import (
    CAMPAIGN_PATH,
    CAMPAIGN_VERSION,
    confirmation_was_opened,
    validate_clean_worktree,
    validate_cuda_device_properties,
    validate_repository_state,
    validate_stage_authorization,
)
from fssr_nam.campaign.amp_sota_v12_gates import evaluate_preflight_gate
from fssr_nam.campaign.amp_sota_v12_registry import (
    append_gate_event,
    gate_decisions,
)
from fssr_nam.campaign.quality_aa_provenance import replace_json, write_new_json
from fssr_nam.data.sota_audit import audit_sota_data_metadata

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / CAMPAIGN_PATH
GATE_LEDGER = CAMPAIGN_DIR / "GATE_LEDGER.jsonl"
SUMMARY = ROOT / "experiments/summaries/amp_sota_prototype_v1_2/preflight.json"


def _run_validation(target: str) -> None:
    subprocess.run(["make", target], cwd=ROOT, check=True)


def _v12_run_disk_gib() -> float:
    total = 0
    run_root = ROOT / "experiments/runs"
    for path in run_root.glob("sota_v1_2_*"):
        if path.is_file():
            total += path.stat().st_size
        elif path.is_dir():
            total += sum(
                child.stat().st_size for child in path.rglob("*") if child.is_file()
            )
    return total / (1024.0**3)


def main() -> int:
    validate_clean_worktree(ROOT)
    decisions = gate_decisions(GATE_LEDGER)
    validate_stage_authorization("preflight", decisions)
    if decisions or SUMMARY.exists():
        raise RuntimeError("v1.2 preflight is already frozen")
    protocol = validate_repository_state(ROOT)
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen CUDA device is unavailable")
    device_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    validate_cuda_device_properties(
        protocol,
        name=device_properties.name,
        total_memory_bytes=device_properties.total_memory,
    )
    parent = json.loads(
        (ROOT / ".codex_campaign/amp_sota_prototype_v1_1/VERDICT.json").read_text(
            encoding="utf-8"
        )
    )
    aa = json.loads(
        (ROOT / ".codex_campaign/quality_aa_v2/VERDICT.json").read_text(
            encoding="utf-8"
        )
    )
    data_audit = audit_sota_data_metadata(ROOT)

    _run_validation("data-audit")
    _run_validation("test")
    _run_validation("lint")
    evidence = {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "stage": "preflight",
        "protocol_frozen": True,
        "clean_execution_snapshot": True,
        "cuda_device_name": device_properties.name,
        "cuda_total_memory_bytes": device_properties.total_memory,
        "parent_campaign": "AMP-SOTA-PROTOTYPE-v1.1",
        "parent_verdict": parent.get("verdict"),
        "aa_backend": aa.get("selected_backend"),
        "aa_backend_requalified": False,
        "data_audit_passed": data_audit.get("passed") is True,
        "tests_passed": True,
        "lint_passed": True,
        "public_license_audit_passed": data_audit.get("passed") is True,
        "source_file_disjoint_splits_verified": data_audit.get("passed") is True,
        "physical_audio_samples_read": 0,
        "selection_eligible_synthetic_samples_generated": 0,
        "test_only_synthetic_fixtures_may_have_been_generated": True,
        "test_only_synthetic_fixtures_eligible_for_selection": False,
        "confirmation_was_opened": confirmation_was_opened(ROOT),
        "fm9_outputs_accessed": False,
        "goal_gpu_hours_used": 0.0,
        "goal_gpu_hours_method": "no_v1_2_scientific_run_exists",
        "goal_disk_gib_used": _v12_run_disk_gib(),
        "goal_disk_method": "sum_of_existing_sota_v1_2_run_files",
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
        "competence_dynamic_authorized" if gate["passed"] else "preflight_failed"
    )
    if not gate["passed"]:
        maturity["status"] = "terminal_no_go"
        maturity["verdict"] = "NO-GO-PREFLIGHT-v1.2"
    replace_json(maturity_path, maturity)
    print(json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
