#!/usr/bin/env python3
"""Evaluate evidence-backed R2-48K-v2 gates without altering v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fssr_nam.campaign.r2_48k_v2 import (
    CAMPAIGN_VERSION,
    validate_repository_configs,
    validate_stage_authorization,
)
from fssr_nam.campaign.r2_48k_v2_gates import (
    R248KV2EvidenceError,
    evaluate_mechanism_gate,
)
from fssr_nam.campaign.r2_48k_v2_registry import append_gate_event, gate_decisions
from fssr_nam.reporting.json_evidence import loads_strict_json

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / ".codex_campaign/r2_48k_v2"
SUMMARY_DIR = ROOT / "experiments/summaries/r2_48k_v2"


def _write_new_json(path: Path, payload: Any) -> None:
    serialized = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == serialized:
            return
        raise R248KV2EvidenceError(f"refusing to overwrite v2 gate evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)


def _replace_json(path: Path, payload: Any) -> None:
    serialized = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def _update_maturity(result: dict[str, Any]) -> None:
    path = CAMPAIGN_DIR / "MATURITY.json"
    maturity = json.loads(path.read_text(encoding="utf-8"))
    if result["campaign_continue"]:
        maturity["current_stage"] = "screen"
        maturity["gates"]["mechanism"] = "passed"
        maturity["gates"]["screen"] = "pending_evidence"
        maturity["status"] = "mechanism_passed_ready_for_screen"
    else:
        maturity["current_stage"] = "terminal"
        maturity["gates"]["mechanism"] = "failed_valid"
        maturity["status"] = "terminal_no_go"
        maturity["verdict"] = "NO-GO-R2-48K-v2"
    _replace_json(path, maturity)


def _terminalize_invalid(reason: str, evidence_path: Path) -> dict[str, Any]:
    failure_path = SUMMARY_DIR / "mechanism_invalid.json"
    failure = {
        "campaign_version": CAMPAIGN_VERSION,
        "format": "fssr-r2-48k-v2-mechanism-invalid-v1",
        "physical_audio_samples_read": 0,
        "reason": reason,
        "source_evidence_path": str(evidence_path.relative_to(ROOT)),
        "stage": "mechanism",
        "status": "INVALID",
        "valid_scientific_gate_result": False,
        "x2_gate_evaluated": False,
    }
    _write_new_json(failure_path, failure)
    append_gate_event(
        CAMPAIGN_DIR / "GATE_LEDGER.jsonl",
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "mechanism_x2",
            "status": "invalid",
            "evidence_path": str(failure_path.relative_to(ROOT)),
        },
    )
    maturity_path = CAMPAIGN_DIR / "MATURITY.json"
    maturity = json.loads(maturity_path.read_text(encoding="utf-8"))
    maturity["current_stage"] = "terminal"
    maturity["gates"]["mechanism"] = "invalid"
    maturity["status"] = "terminal_invalid"
    maturity["verdict"] = "INVALID"
    _replace_json(maturity_path, maturity)
    verdict = {
        "campaign_version": CAMPAIGN_VERSION,
        "evidence_path": str(failure_path.relative_to(ROOT)),
        "physical_audio_samples_read": 0,
        "reason": reason,
        "terminal_stage": "mechanism",
        "valid_scientific_gate_result": False,
        "verdict": "INVALID",
        "x2_gate_evaluated": False,
    }
    _write_new_json(CAMPAIGN_DIR / "VERDICT.json", verdict)
    return {**failure, "verdict": "INVALID"}


def run_mechanism() -> dict[str, Any]:
    validate_repository_configs(ROOT)
    ledger = CAMPAIGN_DIR / "GATE_LEDGER.jsonl"
    decisions = gate_decisions(ledger)
    validate_stage_authorization("mechanism", decisions)
    if decisions.get("mechanism_x2") == "invalid":
        failure = json.loads(
            (SUMMARY_DIR / "mechanism_invalid.json").read_text(encoding="utf-8")
        )
        return {**failure, "verdict": "INVALID"}
    if "mechanism_x2" in decisions:
        return json.loads((SUMMARY_DIR / "mechanism_gate.json").read_text())
    evidence_path = SUMMARY_DIR / "mechanism.json"
    if not evidence_path.is_file():
        raise R248KV2EvidenceError(f"missing v2 mechanism evidence: {evidence_path}")
    try:
        evidence = loads_strict_json(evidence_path.read_text(encoding="utf-8"))
        result = evaluate_mechanism_gate(evidence)
    except (R248KV2EvidenceError, ValueError) as error:
        return _terminalize_invalid(str(error), evidence_path)
    gate_path = SUMMARY_DIR / "mechanism_gate.json"
    _write_new_json(gate_path, result)
    append_gate_event(
        ledger,
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "mechanism_x2",
            "status": "passed" if result["campaign_continue"] else "failed",
            "evidence_path": str(evidence_path.relative_to(ROOT)),
        },
    )
    _update_maturity(result)
    if not result["campaign_continue"]:
        verdict = {
            "campaign_version": CAMPAIGN_VERSION,
            "verdict": "NO-GO-R2-48K-v2",
            "valid_scientific_gate_result": True,
            "terminal_stage": "mechanism",
            "x2_gate_evaluated": True,
            "physical_audio_samples_read": 0,
            "evidence_path": str(gate_path.relative_to(ROOT)),
        }
        _write_new_json(CAMPAIGN_DIR / "VERDICT.json", verdict)
        result["verdict"] = verdict["verdict"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("mechanism",))
    parser.parse_args()
    try:
        result = run_mechanism()
    except (R248KV2EvidenceError, RuntimeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "campaign_version": CAMPAIGN_VERSION,
                    "stage": "mechanism",
                    "status": "INVALID",
                    "reason": str(error),
                },
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 1 if result.get("verdict") in {"NO-GO-R2-48K-v2", "INVALID"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
