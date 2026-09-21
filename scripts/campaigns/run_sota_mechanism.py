#!/usr/bin/env python3
"""Execute the frozen AMP-SOTA-PROTOTYPE-v1.1 mechanism screen once."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from fssr_nam.campaign.amp_sota_gates import evaluate_mechanism_promotion_gate
from fssr_nam.campaign.amp_sota_prototype_v1 import (
    AMENDMENT_PATH,
    CAMPAIGN_PATH,
    CAMPAIGN_VERSION,
    load_protocol,
    validate_repository_state,
    validate_stage_authorization,
)
from fssr_nam.campaign.amp_sota_registry import append_gate_event, gate_decisions
from fssr_nam.campaign.quality_aa_provenance import (
    capture_provenance,
    digest_text,
    replace_json,
    strict_json,
    write_new_json,
)
from fssr_nam.metrics.sota_screen import run_synthetic_mechanism_screen
from fssr_nam.reporting.ledger import append_run, read_runs

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / CAMPAIGN_PATH
GATE_LEDGER = CAMPAIGN_DIR / "GATE_LEDGER.jsonl"
GLOBAL_LEDGER = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
RUN_ID = "sota_v1_1_mechanism_synthetic_all_seed20260830_v1"
RUN_DIR = ROOT / "experiments/runs" / RUN_ID
SUMMARY = ROOT / "experiments/summaries/amp_sota_prototype_v1_1/mechanism.json"
SOURCE_FILES = (
    "configs/amp_sota_prototype_v1/protocol.yaml",
    "configs/amp_sota_prototype_v1_1/protocol.yaml",
    ".codex_campaign/amp_sota_prototype_v1/PROTOCOL_LOCK.yaml",
    ".codex_campaign/amp_sota_prototype_v1_1/PROTOCOL_LOCK.yaml",
    "src/fssr_nam/campaign/amp_sota_prototype_v1.py",
    "src/fssr_nam/campaign/amp_sota_gates.py",
    "src/fssr_nam/campaign/amp_sota_registry.py",
    "src/fssr_nam/data/arch_fixtures.py",
    "src/fssr_nam/metrics/quality_aliasing.py",
    "src/fssr_nam/metrics/sota_screen.py",
    "src/fssr_nam/models/approximants.py",
    "src/fssr_nam/models/equiripple.py",
    "scripts/campaigns/run_sota_mechanism.py",
)


def _ledger_entry(
    *,
    timestamp: str,
    commit: str,
    protocol: dict[str, Any],
    amendment: dict[str, Any],
    status: str,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "date": timestamp,
        "run_id": RUN_ID,
        "phase": "AMP-SOTA-PROTOTYPE-v1.1-MECHANISM",
        "model": "isolated_approximant_slow_control_resampler_matrix",
        "device": "synthetic_cpu_float64_one_thread",
        "seed": 20_260_830,
        "commit": commit,
        "config_sha256": digest_text(strict_json(amendment)),
        "data_sha256": digest_text(
            strict_json(
                {
                    "tier": "SYNTHETIC",
                    "physical_audio_samples_read": 0,
                    "mechanism_screen": amendment["mechanism_screen"],
                }
            )
        ),
        "status": status,
        "failure_reason": failure_reason,
        "results_path": str(RUN_DIR.relative_to(ROOT)),
        "base_protocol_campaign": protocol["campaign_version"],
    }


def main() -> int:
    decisions = gate_decisions(GATE_LEDGER)
    validate_stage_authorization("screen", decisions)
    if "screen" in decisions:
        raise RuntimeError("SOTA v1.1 mechanism screen is already frozen")
    if (
        RUN_DIR.exists()
        or SUMMARY.exists()
        or any(run["run_id"] == RUN_ID for run in read_runs(GLOBAL_LEDGER))
    ):
        raise RuntimeError("SOTA v1.1 mechanism run ID is already reserved")
    protocol = validate_repository_state(ROOT)
    amendment = yaml.safe_load((ROOT / AMENDMENT_PATH).read_text(encoding="utf-8"))
    if not isinstance(amendment, dict):
        raise RuntimeError("SOTA v1.1 amendment must be a mapping")

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    provenance = capture_provenance(
        ROOT,
        RUN_DIR,
        ["uv", "run", "python", "scripts/campaigns/run_sota_mechanism.py"],
        source_files=SOURCE_FILES,
    )
    write_new_json(
        RUN_DIR / "status.json",
        {"status": "running", "started_at": started_at, "finished_at": None},
    )
    try:
        evidence = run_synthetic_mechanism_screen(amendment)
        evidence.update(
            {
                "run_id": RUN_ID,
                "base_protocol_campaign": protocol["campaign_version"],
                "confirmation_outputs_accessed": False,
                "fm9_outputs_accessed": False,
                "provenance": provenance,
            }
        )
        gate = evaluate_mechanism_promotion_gate(evidence, load_protocol(ROOT))
        evidence["gate"] = gate
        write_new_json(RUN_DIR / "mechanism.json", evidence)
        write_new_json(SUMMARY, evidence)
    except Exception as error:
        finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
        reason = f"{type(error).__name__}: {error}"
        append_run(
            GLOBAL_LEDGER,
            _ledger_entry(
                timestamp=finished_at,
                commit=provenance["git_head"],
                protocol=protocol,
                amendment=amendment,
                status="failed",
                failure_reason=reason,
            ),
        )
        replace_json(
            RUN_DIR / "status.json",
            {
                "status": "failed",
                "started_at": started_at,
                "finished_at": finished_at,
                "failure_reason": reason,
            },
        )
        maturity = json.loads((CAMPAIGN_DIR / "MATURITY.json").read_text())
        maturity.update(
            {
                "current_stage": "mechanism_invalid",
                "status": "terminal_invalid",
                "verdict": "INVALID-MECHANISM",
                "scientific_runs_launched": 1,
            }
        )
        replace_json(CAMPAIGN_DIR / "MATURITY.json", maturity)
        raise

    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    append_run(
        GLOBAL_LEDGER,
        _ledger_entry(
            timestamp=finished_at,
            commit=provenance["git_head"],
            protocol=protocol,
            amendment=amendment,
            status="completed",
            failure_reason="",
        ),
    )
    replace_json(
        RUN_DIR / "status.json",
        {
            "status": "completed",
            "started_at": started_at,
            "finished_at": finished_at,
            "failure_reason": "",
        },
    )
    status = "passed" if gate["passed"] else "no-go"
    append_gate_event(
        GATE_LEDGER,
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "screen",
            "status": status,
            "evidence_path": str(SUMMARY.relative_to(ROOT)),
        },
    )
    maturity = json.loads((CAMPAIGN_DIR / "MATURITY.json").read_text())
    maturity.update(
        {
            "current_stage": (
                "mechanism_screen_passed" if gate["passed"] else "mechanism_no_go"
            ),
            "status": "active" if gate["passed"] else "terminal_no_go",
            "verdict": None if gate["passed"] else "NO-GO-MECHANISM",
            "scientific_runs_launched": 1,
        }
    )
    replace_json(CAMPAIGN_DIR / "MATURITY.json", maturity)
    print(json.dumps(gate, allow_nan=False, indent=2, sort_keys=True))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
