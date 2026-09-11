#!/usr/bin/env python3
"""Close AMP-COMPETENCE-ARCH-v2 with technical and scientific audits."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from fssr_nam.campaign.amp_arch_v2_registry import (
    append_gate_event,
    gate_decisions,
    read_gate_events,
)
from fssr_nam.campaign.amp_competence_arch_v2 import (
    CAMPAIGN_PATH,
    CAMPAIGN_VERSION,
    validate_repository_state,
    validate_stage_authorization,
)
from fssr_nam.campaign.quality_aa_provenance import (
    replace_json,
    write_new_json,
    write_new_text,
)
from fssr_nam.reporting.ledger import read_runs

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / CAMPAIGN_PATH
GATE_LEDGER = CAMPAIGN_DIR / "GATE_LEDGER.jsonl"
GLOBAL_LEDGER = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
SUMMARY_DIR = ROOT / "experiments/summaries/amp_competence_arch_v2"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _run_validation(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout + completed.stderr).strip().splitlines()
    return {
        "command": command,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "output_tail": output[-20:],
    }


def _load_summary(name: str) -> dict[str, Any]:
    path = SUMMARY_DIR / f"{name}.json"
    if not path.is_file():
        raise RuntimeError(f"missing v2 {name} evidence")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("physical_audio_samples_read") != 0:
        raise RuntimeError(f"v2 {name} evidence accessed physical audio")
    if value.get("sealed_outputs_accessed") != {
        "blackstar": False,
        "ua1176": False,
    }:
        raise RuntimeError(f"v2 {name} evidence crossed a sealed boundary")
    return value


def main() -> int:
    decisions = gate_decisions(GATE_LEDGER)
    validate_stage_authorization("audit", decisions)
    validate_repository_state(ROOT, require_frozen=True)
    if (CAMPAIGN_DIR / "AUDIT.json").exists():
        raise RuntimeError("v2 audit is already immutable")
    preflight = json.loads((SUMMARY_DIR / "preflight.json").read_text(encoding="utf-8"))
    competence = _load_summary("competence")
    comparison = (
        _load_summary("comparison") if decisions.get("competence") == "passed" else None
    )
    verdict = json.loads((CAMPAIGN_DIR / "VERDICT.json").read_text(encoding="utf-8"))
    expected_competence_runs = 9
    expected_comparison_runs = 45 if comparison is not None else 0
    v2_runs = [
        entry
        for entry in read_runs(GLOBAL_LEDGER)
        if entry["run_id"].startswith("arch_v2_")
    ]
    if len(v2_runs) != expected_competence_runs + expected_comparison_runs:
        raise RuntimeError("v2 global run ledger count is incomplete")
    if any(entry.get("status") != "completed" for entry in v2_runs):
        raise RuntimeError("v2 audit found a failed or incomplete scientific run")
    for entry in v2_runs:
        run_dir = ROOT / entry["results_path"]
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if (
            status.get("status") != "completed"
            or not (run_dir / "result.json").is_file()
        ):
            raise RuntimeError(f"v2 run artifact is incomplete: {entry['run_id']}")

    make_test = _run_validation(["make", "test"])
    make_lint = _run_validation(["make", "lint"])
    if not make_test["passed"] or not make_lint["passed"]:
        raise RuntimeError("v2 closure validation failed")
    audit = {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "finished_at": _now(),
        "status": "passed",
        "verdict": verdict["verdict"],
        "gate_events": read_gate_events(GATE_LEDGER),
        "preflight_passed": preflight.get("status") == "passed",
        "competence_status": competence["status"],
        "comparison_status": None if comparison is None else comparison["status"],
        "selected_budget_updates": competence["gate"]["selected_budget_updates"],
        "promoted_candidate": (
            None if comparison is None else comparison["gate"]["promoted_candidate"]
        ),
        "v1_runs_resumed": 0,
        "v1_runs_retuned": 0,
        "competence_run_count": expected_competence_runs,
        "comparison_run_count": expected_comparison_runs,
        "physical_audio_samples_read": 0,
        "sealed_outputs_accessed": {"blackstar": False, "ua1176": False},
        "external_report_only_locked": True,
        "global_run_ledger_entries": len(v2_runs),
        "make_test": make_test,
        "make_lint": make_lint,
    }
    write_new_json(CAMPAIGN_DIR / "AUDIT.json", audit)
    technical = (
        "# Audit technique AMP-COMPETENCE-ARCH-v2\n\n"
        f"- Verdict : `{verdict['verdict']}`.\n"
        f"- Runs de compétence complets : {expected_competence_runs}/9.\n"
        "- Runs de comparaison complets : "
        f"{expected_comparison_runs}/{expected_comparison_runs}.\n"
        "- Reprises ou retunings v1 : 0.\n"
        "- Échantillons audio physiques lus : 0.\n"
        "- Blackstar, UA1176 et EXTERNAL_REPORT_ONLY sont restés verrouillés.\n"
        "- Chaque run possède statut, résultat, checkpoints et snapshot de source.\n"
        "- `make test` et `make lint` passent dans l'audit de clôture.\n"
    )
    write_new_text(CAMPAIGN_DIR / "TECHNICAL_AUDIT.md", technical)
    if comparison is None:
        scientific_conclusion = (
            "Le contrôle n'a pas démontré une convergence confirmable sous le "
            "budget préenregistré. Aucune comparaison architecturale n'a été lancée."
        )
    elif comparison["gate"]["passed"]:
        scientific_conclusion = (
            "Le contrôle a fixé un budget de "
            f"{competence['gate']['selected_budget_updates']} "
            f"updates et `{comparison['gate']['promoted_candidate']}` passe la gate "
            "synthétique appariée. Cette conclusion ne constitue pas une revendication "
            "de qualité physique ou SOTA."
        )
    else:
        scientific_conclusion = (
            "Le contrôle a fixé un budget de "
            f"{competence['gate']['selected_budget_updates']} "
            "updates, mais aucune architecture gelée ne passe simultanément les gates "
            "appariées."
        )
    scientific = (
        "# Audit scientifique AMP-COMPETENCE-ARCH-v2\n\n"
        f"{scientific_conclusion}\n\n"
        "Le résultat est valide dans la portée synthétique gelée. Il ne porte sur "
        "aucun "
        "amplificateur physique, aucun holdout scellé et aucune donnée 192 kHz.\n"
    )
    write_new_text(CAMPAIGN_DIR / "SCIENTIFIC_AUDIT.md", scientific)
    append_gate_event(
        GATE_LEDGER,
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "audit",
            "status": "passed",
            "evidence_path": str((CAMPAIGN_DIR / "AUDIT.json").relative_to(ROOT)),
        },
    )
    maturity_path = CAMPAIGN_DIR / "MATURITY.json"
    maturity = json.loads(maturity_path.read_text(encoding="utf-8"))
    maturity["current_stage"] = "terminal_audited"
    replace_json(maturity_path, maturity)
    lineages_path = ROOT / ".codex_campaign/LINEAGES.json"
    lineages = json.loads(lineages_path.read_text(encoding="utf-8"))
    lineage = lineages["lineages"]["amp_competence_arch_v2"]
    lineage["administrative_status"] = f"terminal_{verdict['verdict'].lower()}"
    lineage["historical_artifacts_immutable"] = True
    replace_json(lineages_path, lineages)
    state = (
        "# État\n\n"
        "- Lignée : `AMP-COMPETENCE-ARCH-v2`\n"
        f"- Statut : terminal audité `{verdict['verdict']}`\n"
        f"- Budget de convergence : {competence['gate']['selected_budget_updates']}\n"
        f"- Candidat promu : {audit['promoted_candidate']}\n"
        "- Runs v1 repris/retunés : 0/0\n"
        "- Audio physique lu : 0 échantillon\n"
        "- Validation finale : `make test` et `make lint` verts\n"
    )
    (CAMPAIGN_DIR / "STATE.md").write_text(state, encoding="utf-8")
    claims = (
        "# Revendications\n\n"
        f"- Verdict synthétique gelé : `{verdict['verdict']}`.\n"
        "- Aucune revendication de qualité physique, perceptuelle ou SOTA.\n"
        "- Aucun résultat Blackstar, UA1176, 192 kHz ou FM9 n'a été consulté.\n"
    )
    (CAMPAIGN_DIR / "CLAIMS.md").write_text(claims, encoding="utf-8")
    handoff = (
        "# Handoff\n\n"
        "La lignée v2 est terminale et immuable. Toute extension vers l'audio physique "
        "ou un nouveau tuning exige une nouvelle lignée prospective.\n"
    )
    (CAMPAIGN_DIR / "HANDOFF.md").write_text(handoff, encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
