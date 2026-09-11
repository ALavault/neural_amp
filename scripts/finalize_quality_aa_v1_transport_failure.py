"""Close the one observed QUALITY-AA-v1 preflight crash without rerunning it."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fssr_nam.campaign.quality_aa_provenance import (
    digest_text,
    replace_json,
    write_new_json,
)
from fssr_nam.campaign.quality_aa_registry import append_gate_event, gate_decisions
from fssr_nam.campaign.quality_aa_v1 import CAMPAIGN_VERSION, make_run_id
from fssr_nam.reporting.ledger import append_run, read_runs

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / ".codex_campaign/quality_aa_v1"
RUN_ID = make_run_id("preflight")
RUN_DIR = ROOT / "experiments/runs" / RUN_ID
LEDGER = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
REASON = (
    "TypeError: PyYAML decoded the unquoted protocol key 'off' as boolean false; "
    "strict JSON sorting rejected mixed boolean/string mapping keys"
)


def main() -> None:
    status_path = RUN_DIR / "status.json"
    if not status_path.is_file():
        raise RuntimeError("the observed v1 preflight run is missing")
    import json

    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "running":
        raise RuntimeError("v1 preflight is not in the observed running state")
    if any(run["run_id"] == RUN_ID for run in read_runs(LEDGER)):
        raise RuntimeError("v1 preflight is already registered")
    if "preflight" in gate_decisions(CAMPAIGN_DIR / "GATE_LEDGER.jsonl"):
        raise RuntimeError("v1 preflight gate is already registered")
    manifest = json.loads(
        (RUN_DIR / "source_snapshot_manifest.json").read_text(encoding="utf-8")
    )
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    failure = {
        "campaign_version": CAMPAIGN_VERSION,
        "run_id": RUN_ID,
        "status": "INVALID",
        "failure_stage": "post_measurement_evidence_registration",
        "failure_reason": REASON,
        "candidate_route_outputs_observed": False,
        "reference_outputs_observed": True,
        "physical_audio_samples_read": 0,
        "rerun_authorized": False,
    }
    write_new_json(RUN_DIR / "failure.json", failure)
    write_new_json(CAMPAIGN_DIR / "VERDICT.json", failure)
    append_run(
        LEDGER,
        {
            "date": finished_at,
            "run_id": RUN_ID,
            "phase": "QUALITY-AA-PREFLIGHT",
            "model": "instrumentation_without_candidate_routes",
            "device": "synthetic",
            "seed": 0,
            "commit": manifest["git_head"],
            "config_sha256": digest_text(
                (ROOT / "configs/quality_aa_v1/protocol.yaml").read_text()
            ),
            "data_sha256": digest_text("direct_synthetic_x8_x16_reference_grid_v1"),
            "status": "failed",
            "failure_reason": REASON,
            "results_path": str(RUN_DIR.relative_to(ROOT)),
        },
    )
    replace_json(
        status_path,
        {
            "status": "failed",
            "started_at": status["started_at"],
            "finished_at": finished_at,
            "failure_reason": REASON,
        },
    )
    append_gate_event(
        CAMPAIGN_DIR / "GATE_LEDGER.jsonl",
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "preflight",
            "status": "invalid",
            "evidence_path": str((RUN_DIR / "failure.json").relative_to(ROOT)),
        },
    )
    replace_json(
        CAMPAIGN_DIR / "MATURITY.json",
        {
            "campaign_version": CAMPAIGN_VERSION,
            "current_stage": "terminal_invalid",
            "status": "terminal_invalid",
            "scientific_runs_launched": 1,
            "invalid": True,
        },
    )


if __name__ == "__main__":
    main()
