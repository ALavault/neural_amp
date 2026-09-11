from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from fssr_nam.campaign.quality_aa_provenance import (
    capture_provenance,
    digest_text,
    replace_json,
    strict_json,
    write_new_json,
)
from fssr_nam.campaign.quality_aa_v2 import (
    CAMPAIGN_VERSION,
    make_run_id,
    validate_repository_configs,
    validate_stage_authorization,
)
from fssr_nam.campaign.quality_aa_v2_gates import evaluate_mechanism_gate
from fssr_nam.campaign.quality_aa_v2_registry import (
    append_gate_event,
    gate_decisions,
)
from fssr_nam.data.r2_fixtures import R2_FIXTURES
from fssr_nam.metrics.quality_aa_mechanism import qualify_synthetic_mechanism
from fssr_nam.reporting.ledger import append_run, read_runs

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / ".codex_campaign/quality_aa_v2"
GLOBAL_LEDGER = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
RUN_ID = make_run_id("mechanism")
RUN_DIR = ROOT / "experiments/runs" / RUN_ID
SUMMARY_PATH = ROOT / "experiments/summaries/quality_aa_v2/mechanism.json"
PREFLIGHT_PATH = ROOT / "experiments/summaries/quality_aa_v2/preflight.json"


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return value


def _reference_index(preflight: dict[str, Any]) -> dict[tuple[str, int, float], Path]:
    convergence = preflight.get("reference_convergence", {})
    if convergence.get("all_conditions_passed") is not True:
        raise RuntimeError("v2 reference convergence did not pass")
    rows = convergence.get("rows")
    if not isinstance(rows, list) or len(rows) != 54:
        raise RuntimeError("v2 reference index is incomplete")
    index = {}
    for row in rows:
        key = (str(row["fixture"]), int(row["k0"]), float(row["amplitude"]))
        path = ROOT / str(row["selected_x16_path"])
        if key in index or not path.is_file():
            raise RuntimeError(f"invalid v2 reference entry: {key}")
        index[key] = path
    return index


def main() -> int:
    decisions = gate_decisions(CAMPAIGN_DIR / "GATE_LEDGER.jsonl")
    validate_stage_authorization("mechanism", decisions)
    if "mechanism" in decisions:
        raise RuntimeError("QUALITY-AA-v2 mechanism is already recorded")
    if (
        RUN_DIR.exists()
        or SUMMARY_PATH.exists()
        or any(run["run_id"] == RUN_ID for run in read_runs(GLOBAL_LEDGER))
    ):
        raise RuntimeError("QUALITY-AA-v2 mechanism run ID is already reserved")
    protocol = validate_repository_configs(ROOT)
    measurement_lock = _load_json(
        CAMPAIGN_DIR / "MEASUREMENT_LOCK.json", "measurement lock"
    )
    if measurement_lock.get("status") != "locked_before_candidate_mechanism_render":
        raise RuntimeError("candidate mechanism lacks a valid pre-measurement lock")
    preflight = _load_json(PREFLIGHT_PATH, "preflight evidence")
    if preflight.get("candidate_route_outputs_observed") is not False:
        raise RuntimeError("preflight candidate boundary was violated")
    references = _reference_index(preflight)
    floor_db = float(measurement_lock["locked_floor_db"])
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    provenance = capture_provenance(
        ROOT,
        RUN_DIR,
        ["uv", "run", "python", "scripts/run_quality_aa_mechanism.py"],
    )
    write_new_json(
        RUN_DIR / "status.json",
        {"status": "running", "started_at": started_at, "finished_at": None},
    )

    def provide(fixture: str, k0: int, amplitude: float) -> np.ndarray:
        return np.load(references[(fixture, k0, amplitude)], allow_pickle=False)

    evidence = qualify_synthetic_mechanism(
        floor_db=floor_db,
        reference_provider=provide,
        progress=lambda message: print(message, flush=True),
    )
    evidence.update(
        {
            "run_id": RUN_ID,
            "parent_numeric_evidence_reused": False,
            "same_v2_locked_reference_reused": True,
            "measurement_lock_path": str(
                (CAMPAIGN_DIR / "MEASUREMENT_LOCK.json").relative_to(ROOT)
            ),
            "preflight_evidence_path": str(PREFLIGHT_PATH.relative_to(ROOT)),
            "provenance": provenance,
        }
    )
    gate = evaluate_mechanism_gate(evidence)
    evidence["gate"] = gate
    write_new_json(RUN_DIR / "mechanism.json", evidence)
    write_new_json(SUMMARY_PATH, evidence)
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    append_run(
        GLOBAL_LEDGER,
        {
            "date": finished_at,
            "run_id": RUN_ID,
            "phase": "QUALITY-AA-MECHANISM",
            "model": "analytic_fixture_matrix",
            "device": "synthetic",
            "seed": 0,
            "commit": provenance["git_head"],
            "config_sha256": digest_text(strict_json(protocol)),
            "data_sha256": digest_text(
                strict_json(
                    {
                        "fixtures": list(R2_FIXTURES),
                        "floor_db": floor_db,
                        "reference": "locked_v2_preflight_x16",
                    }
                )
            ),
            "status": "completed",
            "failure_reason": "",
            "results_path": str(RUN_DIR.relative_to(ROOT)),
        },
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
    append_gate_event(
        CAMPAIGN_DIR / "GATE_LEDGER.jsonl",
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "mechanism",
            "status": "passed" if gate["passed"] else "no-go",
            "evidence_path": str(SUMMARY_PATH.relative_to(ROOT)),
        },
    )
    if gate["passed"]:
        maturity = {
            "campaign_version": CAMPAIGN_VERSION,
            "current_stage": "mechanism_qualified",
            "status": "active",
            "scientific_runs_launched": 2,
            "invalid": False,
        }
    else:
        verdict = {
            "campaign_version": CAMPAIGN_VERSION,
            "verdict": "NO-GO-QUALITY-AA-v2",
            "selected_backend": None,
            "reason": "no synthetic route passed the preregistered mechanism gate",
            "mechanism_gate_path": str(SUMMARY_PATH.relative_to(ROOT)),
        }
        write_new_json(CAMPAIGN_DIR / "VERDICT.json", verdict)
        maturity = {
            "campaign_version": CAMPAIGN_VERSION,
            "current_stage": "terminal_no_go",
            "status": "terminal_no_go",
            "scientific_runs_launched": 2,
            "invalid": False,
        }
    replace_json(CAMPAIGN_DIR / "MATURITY.json", maturity)
    print(json.dumps(gate, allow_nan=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
