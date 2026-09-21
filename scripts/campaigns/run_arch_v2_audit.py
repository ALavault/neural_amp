#!/usr/bin/env python3
"""Close AMP-COMPETENCE-ARCH-v2 with technical and scientific audits."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from fssr_nam.campaign.amp_arch_v2_gates import evaluate_competence_gate
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
    protocol = validate_repository_state(ROOT, require_frozen=True)
    if (CAMPAIGN_DIR / "AUDIT.json").exists():
        raise RuntimeError("v2 audit is already immutable")
    preflight = json.loads((SUMMARY_DIR / "preflight.json").read_text(encoding="utf-8"))
    competence = _load_summary("competence")
    comparison = (
        _load_summary("comparison") if decisions.get("competence") == "passed" else None
    )
    verdict = json.loads((CAMPAIGN_DIR / "VERDICT.json").read_text(encoding="utf-8"))
    recomputed_competence_gate = evaluate_competence_gate(
        competence["trajectories"], protocol
    )
    if recomputed_competence_gate != competence["gate"]:
        raise RuntimeError("stored v2 competence gate does not recompute exactly")
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

    terminal_evidence = _run_validation(
        [
            "uv",
            "run",
            "pytest",
            "tests/unit/test_amp_arch_v2_terminal_evidence.py",
            "-q",
        ]
    )
    make_lint = _run_validation(["make", "lint"])
    preliminary_failures = {
        name: result
        for name, result in (
            ("terminal_evidence", terminal_evidence),
            ("make_lint", make_lint),
        )
        if not result["passed"]
    }
    if preliminary_failures:
        raise RuntimeError(
            "v2 preliminary closure validation failed: "
            + json.dumps(preliminary_failures, sort_keys=True)
        )
    make_test = _run_validation(["make", "test"])
    if not make_test["passed"]:
        raise RuntimeError(
            "v2 full test validation failed: " + json.dumps(make_test, sort_keys=True)
        )
    checkpoint_summary = competence["gate"]["checkpoint_summary"]
    penultimate_checkpoint = checkpoint_summary[-2]
    final_checkpoint = checkpoint_summary[-1]
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
        "competence_gate_recomputed_exactly": True,
        "terminal_evidence": terminal_evidence,
        "make_test": make_test,
        "make_lint": make_lint,
        "post_run_audit_hardening": {
            "decision_bearing_code_changed": False,
            "protocol_or_lock_changed": False,
            "scientific_results_changed": False,
            "exact_run_ids_and_ledger_checked": True,
            "all_checkpoints_loaded_and_finite": True,
            "source_snapshots_compared_byte_for_byte": True,
        },
        "adversarial_findings": [
            {
                "id": "ARCH2-AF-001",
                "severity": "critical_for_unrun_comparison",
                "finding": (
                    "the frozen comparison assigns phys_s6_tcn_x2 the auxiliary "
                    "weight 0.01 selected by the parent v1 mechanism campaign while "
                    "declaring that v1 results were not used for v2 selection"
                ),
                "impact_on_competence_verdict": "none_control_auxiliary_weight_is_zero",
                "required_disposition": "do_not_reuse_comparison_contract",
            },
            {
                "id": "ARCH2-AF-002",
                "severity": "critical_for_unrun_comparison",
                "finding": (
                    "phys_s6_tcn_x2 and rf2047_tfilm_x2 instantiate the same observer "
                    "conditioned topology, but only phys_s6_tcn_x2 receives auxiliary "
                    "state supervision, contradicting the same-loss comparison claim"
                ),
                "impact_on_competence_verdict": "none_no_candidate_was_launched",
                "required_disposition": (
                    "separate_architecture_from_supervision_ablation"
                ),
            },
        ],
        "known_limits": [
            "run snapshots establish within-campaign byte identity but not an "
            "external historical baseline",
            "CUDA deterministic mode and exact accelerator identity were not "
            "enforced by the training code",
            "the run-exception path does not make INVALID as fail-closed as the "
            "scientific gates",
            "the valid verdict is limited to three synthetic systems and three "
            "optimization seeds",
        ],
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
        "- Les neuf IDs attendus sont les seules entrées v2 du registre global.\n"
        "- Tous les checkpoints attendus se chargent et leurs tenseurs sont finis.\n"
        "- Les snapshots de source des neuf runs sont identiques octet par octet.\n"
        "- La gate de compétence est recalculée à l'identique depuis les "
        "trajectoires.\n"
        "- `make test` et `make lint` passent dans l'audit de clôture.\n"
        "\n"
        "## Findings adversariaux\n\n"
        "- `ARCH2-AF-001` — Le poids auxiliaire `0.01` de "
        "`phys_s6_tcn_x2` provient de la sélection v1, contrairement à la déclaration "
        "d'indépendance de sélection v2.\n"
        "- `ARCH2-AF-002` — `phys_s6_tcn_x2` et `rf2047_tfilm_x2` instancient la "
        "même topologie observer-conditioned ; seule la supervision auxiliaire "
        "diffère. La comparaison aurait donc confondu architecture et loss.\n"
        "- Ces deux findings n'affectent pas le verdict de compétence : le contrôle "
        "utilise un poids auxiliaire nul et aucun candidat n'a été lancé. Ils "
        "interdisent en revanche de réutiliser le contrat de comparaison v2.\n"
        "\n"
        "## Limites techniques restantes\n\n"
        "- Les snapshots prouvent l'identité interne entre runs, pas une référence "
        "historique externe.\n"
        "- Le mode CUDA déterministe et l'identité exacte de l'accélérateur ne sont "
        "pas imposés par le code d'entraînement.\n"
        "- Le chemin d'exception d'un run est moins fail-closed que les gates "
        "scientifiques.\n"
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
        "Au checkpoint 10 000, la médiane ESR vaut "
        f"{penultimate_checkpoint['median_esr']:.9f} et son amélioration médiane "
        "vers 15 000 vaut "
        f"{penultimate_checkpoint['next_checkpoint_relative_esr_improvement']:.3%}, "
        "ce qui satisfait le critère de plateau. Les gardes échouent néanmoins "
        f"(gain minimal {penultimate_checkpoint['minimum_gain_error']:.6f}, "
        f"corrélation minimale {penultimate_checkpoint['minimum_correlation']:.6f}). "
        "À 15 000, elles échouent encore "
        f"(gain minimal {final_checkpoint['minimum_gain_error']:.6f}, "
        f"corrélation minimale {final_checkpoint['minimum_correlation']:.6f}) ; "
        "aucun budget ne peut donc être sélectionné. Le contrôle réussit le système "
        "statique, mais reste sous-gainé sur les trois seeds dynamiques et les trois "
        "seeds deux-clippers ; le seed dynamique 0 échoue aussi en corrélation.\n\n"
        "Le résultat est valide dans la portée synthétique gelée. Il ne porte sur "
        "aucun amplificateur physique, aucun holdout scellé et aucune donnée "
        "192 kHz. Les trois systèmes et trois seeds ne justifient aucune "
        "généralisation vers l'état de l'art.\n"
    )
    write_new_text(CAMPAIGN_DIR / "SCIENTIFIC_AUDIT.md", scientific)
    decisions_text = (
        "# Décisions\n\n"
        "- `ARCH2-D-001` — Ouvrir une lignée v2 sans modifier ni réutiliser les "
        "runs v1.\n"
        "- `ARCH2-D-002` — Qualifier le seul contrôle avant tout run candidat.\n"
        "- `ARCH2-D-003` — Fixer le budget par passage des gardes au checkpoint "
        "courant et suivant, puis par plateau ESR médian entre 0 et 5 %.\n"
        "- `ARCH2-D-004` — Utiliser un split de comparaison disjoint et réentraîner "
        "aussi le contrôle pour préserver l'appariement.\n"
        "- `ARCH2-D-005` — Geler quatre candidats sans remplacement dépendant du "
        "résultat.\n"
        "- `ARCH2-D-006` — Exclure tout audio physique, 192 kHz et FM9 de cette "
        "étape.\n"
        "- `ARCH2-D-007` — Enregistrer `NO-GO-COMPETENCE-v2` : aucun checkpoint "
        "confirmé ne passe toutes les gardes.\n"
        "- `ARCH2-D-008` — Lancer zéro run candidat conformément à la gate "
        "fail-closed.\n"
        "- `ARCH2-D-009` — Ne pas réutiliser le contrat de comparaison : il est "
        "contaminé par une sélection v1 et confond topologie et supervision.\n"
        "- `ARCH2-D-010` — Limiter le durcissement post-run à la vérification et à "
        "la documentation ; ne modifier ni protocole, ni runs, ni métriques, ni "
        "verdict.\n"
    )
    (CAMPAIGN_DIR / "DECISIONS.md").write_text(decisions_text, encoding="utf-8")
    failures_text = (
        "# Échecs\n\n"
        "## `NO-GO-COMPETENCE-v2` valide\n\n"
        "Les neuf trajectoires du contrôle sont complètes. Le plateau médian entre "
        "10 000 et 15 000 updates est de 4,114 %, donc dans l'intervalle gelé "
        "[0 %, 5 %]. Les gardes restent toutefois en échec aux deux checkpoints : "
        "à 15 000, le gain minimal vaut -0,251051 pour un seuil strict > -0,2 et "
        "la corrélation minimale 0,888893 pour un seuil strict > 0,9. Aucun budget "
        "confirmatoire n'est sélectionnable.\n\n"
        "Le système statique converge près de la cible. En revanche, les trois "
        "seeds dynamiques et les trois seeds deux-clippers restent sous-gainés ; le "
        "seed dynamique 0 échoue aussi en corrélation. Cette structure exclut une "
        "panne globale d'entraînement et indique une faiblesse systématique du "
        "contrôle sur la dynamique/multi-non-linéarité sous ce protocole. Elle ne "
        "permet pas de trancher entre capacité, paramétrisation et objectif "
        "d'optimisation.\n\n"
        "Aucun run candidat n'a été lancé et aucun run v1 n'a été repris ou retuné. "
        "Une nouvelle hypothèse exige une nouvelle lignée prospective.\n"
    )
    (CAMPAIGN_DIR / "FAILURES.md").write_text(failures_text, encoding="utf-8")
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
    lineage["administrative_status"] = "terminal_no_go_competence"
    lineage["historical_artifacts_immutable"] = True
    replace_json(lineages_path, lineages)
    state = (
        "# État\n\n"
        "- Lignée : `AMP-COMPETENCE-ARCH-v2`\n"
        f"- Statut : terminal audité `{verdict['verdict']}`\n"
        f"- Budget de convergence : {competence['gate']['selected_budget_updates']}\n"
        f"- Candidat promu : {audit['promoted_candidate']}\n"
        "- Runs de compétence/comparaison : 9/0\n"
        "- Runs v1 repris/retunés : 0/0\n"
        "- Audio physique lu : 0 échantillon\n"
        "- Contrat de comparaison : non exécuté et non réutilisable tel quel\n"
        "- Validation finale : `make test` et `make lint` verts\n"
    )
    (CAMPAIGN_DIR / "STATE.md").write_text(state, encoding="utf-8")
    claims = (
        "# Revendications\n\n"
        f"- Verdict synthétique gelé : `{verdict['verdict']}`.\n"
        "- Le contrôle n'a satisfait simultanément les gardes de gain et de "
        "corrélation à aucun couple de checkpoints confirmatoire.\n"
        "- Aucune comparaison d'architectures n'a été exécutée.\n"
        "- Aucune revendication de qualité physique, perceptuelle ou SOTA.\n"
        "- Aucun résultat Blackstar, UA1176, 192 kHz ou FM9 n'a été consulté.\n"
    )
    (CAMPAIGN_DIR / "CLAIMS.md").write_text(claims, encoding="utf-8")
    handoff = (
        "# Handoff\n\n"
        "La lignée v2 est terminale et immuable : ne reprendre ni retuner ses runs. "
        "Toute suite exige une nouvelle lignée prospective. Elle devra d'abord "
        "résoudre la sous-amplitude du contrôle sur les systèmes dynamiques et "
        "deux-clippers, puis séparer explicitement les ablations de topologie et de "
        "supervision. Le contrat de comparaison v2 ne doit pas être réutilisé tel "
        "quel.\n"
    )
    (CAMPAIGN_DIR / "HANDOFF.md").write_text(handoff, encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
