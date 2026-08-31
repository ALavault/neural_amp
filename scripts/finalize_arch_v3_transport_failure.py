#!/usr/bin/env python3
"""Close the interrupted AMP-QUALITY-ARCH-v3 round without resuming it."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fssr_nam.campaign.amp_arch_v3_registry import append_gate_event, gate_decisions

from fssr_nam.campaign.amp_quality_arch_v3 import CAMPAIGN_VERSION
from fssr_nam.campaign.quality_aa_provenance import (
    replace_json,
    write_new_json,
)
from fssr_nam.reporting.ledger import append_run, read_runs

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / ".codex_campaign/amp_quality_arch_v3"
RUNS_DIR = ROOT / "experiments/runs"
LEDGER = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
GATE_LEDGER = CAMPAIGN_DIR / "GATE_LEDGER.jsonl"
RUN_ID = "arch_v3_round_1_two_clippers_primary__slow_state_micro_tcn_x2_seed0_v1"
RUN_DIR = RUNS_DIR / RUN_ID
CPJ_JOB_ID = "job-mte87wc2-6e549f1d"
REASON = (
    "Codex Process Jobs reported the critical round-one worker terminal as failed: "
    "Tracked process ended without reporting a terminal status. The worker stopped "
    "during this trajectory after 15 earlier trajectories completed; protocol forbids "
    "resume or retry."
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _append_markdown(path: Path, text: str) -> None:
    existing = path.read_text(encoding="utf-8")
    if text.strip() in existing:
        raise RuntimeError(f"closure text already exists in {path}")
    path.write_text(existing.rstrip() + "\n\n" + text.strip() + "\n", encoding="utf-8")


def main() -> None:
    status_path = RUN_DIR / "status.json"
    if not status_path.is_file():
        raise RuntimeError("the interrupted v3 run directory is missing")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "running" or status.get("finished_at") is not None:
        raise RuntimeError(
            "the interrupted v3 run is not in its observed running state"
        )
    if (RUN_DIR / "result.json").exists():
        raise RuntimeError("the interrupted v3 run unexpectedly contains a result")

    ledger_rows = read_runs(LEDGER)
    v3_rows = [
        row for row in ledger_rows if row.get("phase") == "AMP-QUALITY-ARCH-v3-ROUND-1"
    ]
    if len(v3_rows) != 15 or any(row.get("status") != "completed" for row in v3_rows):
        raise RuntimeError("expected exactly 15 completed v3 round-one ledger rows")
    if any(row.get("run_id") == RUN_ID for row in ledger_rows):
        raise RuntimeError("the interrupted v3 run is already registered")
    run_dirs = sorted(path.name for path in RUNS_DIR.glob("arch_v3_round_1_*"))
    if len(run_dirs) != 16 or RUN_ID not in run_dirs:
        raise RuntimeError("v3 round-one run-directory census changed")
    if "round_1" in gate_decisions(GATE_LEDGER):
        raise RuntimeError("the v3 round-one gate is already registered")
    if (CAMPAIGN_DIR / "VERDICT.json").exists():
        raise RuntimeError("the v3 verdict already exists")

    template = next(
        row
        for row in v3_rows
        if row["run_id"].startswith(
            "arch_v3_round_1_two_clippers_primary__gainhead_micro_tcn_x2_"
        )
    )
    manifest = json.loads(
        (RUN_DIR / "source_snapshot_manifest.json").read_text(encoding="utf-8")
    )
    finished_at = _now()
    failure = {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "stage": "round_1",
        "status": "invalid",
        "verdict": "INVALID",
        "reason": REASON,
        "cpj_job_id": CPJ_JOB_ID,
        "completed_trajectories": 15,
        "failed_trajectories": 1,
        "unlaunched_trajectories": 11,
        "run_directories_created": run_dirs,
        "partial_results_eligible_for_selection": False,
        "resume_allowed": False,
        "validation_source_accessed": False,
        "physical_audio_samples_read": 0,
    }
    failure_path = CAMPAIGN_DIR / "ROUND_1_INVALID.json"
    write_new_json(RUN_DIR / "failure.json", failure)
    write_new_json(failure_path, failure)
    replace_json(
        status_path,
        {
            "status": "failed",
            "started_at": status["started_at"],
            "finished_at": finished_at,
            "failure_reason": REASON,
        },
    )
    append_run(
        LEDGER,
        {
            "date": finished_at,
            "run_id": RUN_ID,
            "phase": "AMP-QUALITY-ARCH-v3-ROUND-1",
            "model": "slow_state_micro_tcn_x2",
            "device": "synthetic_48khz_internal_dev",
            "seed": 0,
            "commit": manifest["git_head"],
            "config_sha256": template["config_sha256"],
            "data_sha256": template["data_sha256"],
            "status": "failed",
            "failure_reason": REASON,
            "results_path": str(RUN_DIR.relative_to(ROOT)),
        },
    )
    append_gate_event(
        GATE_LEDGER,
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "round_1",
            "status": "invalid",
            "evidence_path": str(failure_path.relative_to(ROOT)),
        },
    )
    write_new_json(
        CAMPAIGN_DIR / "VERDICT.json",
        {
            "campaign_version": CAMPAIGN_VERSION,
            "verdict": "INVALID",
            "terminal_stage": "round_1",
            "valid_scientific_gate_result": False,
            "evidence_path": str(failure_path.relative_to(ROOT)),
        },
    )
    replace_json(
        CAMPAIGN_DIR / "MATURITY.json",
        {
            "campaign_version": CAMPAIGN_VERSION,
            "current_stage": "round_1_invalid",
            "exploration_rounds_completed": 0,
            "physical_internal_dev_accessed": False,
            "scientific_runs_launched": 16,
            "status": "terminal_invalid",
            "verdict": "INVALID",
        },
    )
    (CAMPAIGN_DIR / "STATE.md").write_text(
        "# État\n\n"
        "- Lignée : `AMP-QUALITY-ARCH-v3`.\n"
        "- Statut : terminal `INVALID` au round 1.\n"
        "- Trajectoires : 15 complètes, 1 interrompue, 11 jamais lancées.\n"
        "- Résultats partiels : hypothèses historiques seulement; "
        "sélection interdite.\n"
        "- Reprise ou retry : interdits.\n"
        "- Validation et audio physique lus : 0 échantillon.\n"
        "- Blackstar, UA1176 et EXTERNAL_REPORT_ONLY : verrouillés.\n",
        encoding="utf-8",
    )
    (CAMPAIGN_DIR / "HANDOFF.md").write_text(
        "# Handoff\n\n"
        "Le round 1 v3 est terminal `INVALID` après disparition du worker externe. "
        "Ne reprendre, compléter ou comparer aucune trajectoire v3. Les 15 résultats "
        "complets peuvent seulement motiver une nouvelle lignée prospective.\n",
        encoding="utf-8",
    )
    _append_markdown(
        CAMPAIGN_DIR / "FAILURES.md",
        "## ARCH3-F-003 — Interruption externe du round 1\n\n"
        f"Le job `{CPJ_JOB_ID}` a disparu pendant `{RUN_ID}` après 15 trajectoires "
        "complètes. Aucun résultat de la trajectoire interrompue n'existe. La round "
        "entière est `INVALID`; reprise et retry restent interdits.",
    )
    _append_markdown(
        CAMPAIGN_DIR / "DECISIONS.md",
        "- `ARCH3-D-011` — Fermer v3 en `INVALID` sans reprendre la matrice; conserver "
        "les 15 trajectoires complètes comme hypothèses historiques uniquement.",
    )
    _append_markdown(
        CAMPAIGN_DIR / "CLAIMS.md",
        "- `ARCH3-C-005` — Une supériorité architecturale v3 reste non testable: la "
        "matrice prospective est incomplète et invalide.",
    )


if __name__ == "__main__":
    main()
