from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

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
from fssr_nam.campaign.quality_aa_v2_gates import (
    evaluate_mechanism_gate,
    evaluate_native_gate,
    evaluate_preflight_gate,
    final_backend_decision,
)
from fssr_nam.campaign.quality_aa_v2_registry import (
    append_gate_event,
    gate_decisions,
)
from fssr_nam.reporting.ledger import append_run, read_runs

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / ".codex_campaign/quality_aa_v2"
GLOBAL_LEDGER = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
SUMMARY_DIR = ROOT / "experiments/summaries/quality_aa_v2"
RUN_ID = make_run_id("audit")
RUN_DIR = ROOT / "experiments/runs" / RUN_ID
SUMMARY_PATH = SUMMARY_DIR / "audit.json"


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return value


def _snapshot_check(run_dir: Path) -> bool:
    manifest_path = run_dir / "source_snapshot_manifest.json"
    environment_path = run_dir / "environment.json"
    if not manifest_path.is_file() or not environment_path.is_file():
        return False
    manifest = _load_json(manifest_path, "source snapshot manifest")
    snapshot = run_dir / "source_snapshot"
    files = manifest.get("files")
    return (
        manifest.get("method") == "exact-file-copy-v1"
        and isinstance(files, list)
        and len(files) >= 10
        and all((snapshot / relative).is_file() for relative in files)
    )


def _write_campaign_documents(decision: dict[str, Any], audit_path: str) -> None:
    selected = decision["selected_backend"]
    (CAMPAIGN_DIR / "STATE.md").write_text(
        "# État\n\n"
        f"- Lignée : `FSSR-QUALITY-AA-v2`\n"
        f"- Statut : terminal `{decision['verdict']}`\n"
        f"- Backend sélectionné : `{selected}`\n"
        "- Portée : mécanisme AA synthétique et profil natif seulement\n"
        "- Audio physique lu : 0 échantillon\n"
        "- `EXTERNAL_REPORT_ONLY` : verrouillé\n"
        f"- Audit : `{audit_path}`\n",
        encoding="utf-8",
    )
    (CAMPAIGN_DIR / "DECISIONS.md").write_text(
        "# Décisions\n\n"
        "- `full_island_x2` et `teacher_x4` passent le gate scientifique.\n"
        "- Les deux routes atteignent le plancher de résidu connu et sont dans "
        "la bande d'indifférence qualité de 0,5 dB.\n"
        "- `teacher_x4` est rejeté du profil temps réel : son p95 bloc 64 "
        "dépasse la limite préenregistrée.\n"
        "- `full_island_x2` est sélectionné : parité, latence 32 et p95 temps "
        "réel passent.\n"
        "- ADAA reste exclue sans affecter x2/x4 faute d'un délai global "
        "compensable et d'une parité de compensation native.\n",
        encoding="utf-8",
    )
    (CAMPAIGN_DIR / "CLAIMS.md").write_text(
        "# Revendications\n\n"
        "## Supported\n\n"
        "Sous les fixtures et gates prospectifs `FSSR-QUALITY-AA-v2`, "
        "`full_island_x2` est un backend AA synthétiquement qualifié et "
        "compatible avec le profil natif préenregistré.\n\n"
        "## Not supported\n\n"
        "Aucune conclusion n'est permise sur une architecture d'ampli, une "
        "capture matérielle, une fidélité physique ou l'état de l'art global.\n",
        encoding="utf-8",
    )
    (CAMPAIGN_DIR / "FAILURES.md").write_text(
        "# Échecs\n\n"
        "Aucune panne d'instrumentation v2. `teacher_x4` est un rejet valide du "
        "gate runtime, pas un run invalide. Le transport v1 reste `INVALID` "
        "dans sa lignée immuable.\n",
        encoding="utf-8",
    )
    (CAMPAIGN_DIR / "HANDOFF.md").write_text(
        "# Reprise\n\n"
        "Le prochain goal peut ouvrir la frontière de qualité des architectures "
        "d'ampli en utilisant `full_island_x2` comme backend AA qualifié. Il "
        "devra figer ses comparateurs état de l'art, données, enveloppe runtime "
        "et protocole perceptuel dans une nouvelle lignée. Ne pas transformer "
        "ce verdict synthétique en revendication matérielle.\n",
        encoding="utf-8",
    )


def main() -> int:
    decisions = gate_decisions(CAMPAIGN_DIR / "GATE_LEDGER.jsonl")
    validate_stage_authorization("audit", decisions)
    if "audit" in decisions:
        raise RuntimeError("QUALITY-AA-v2 audit is already recorded")
    if (
        RUN_DIR.exists()
        or SUMMARY_PATH.exists()
        or any(run["run_id"] == RUN_ID for run in read_runs(GLOBAL_LEDGER))
    ):
        raise RuntimeError("QUALITY-AA-v2 audit run ID is already reserved")
    protocol = validate_repository_configs(ROOT)
    preflight = _load_json(SUMMARY_DIR / "preflight.json", "preflight")
    mechanism = _load_json(SUMMARY_DIR / "mechanism.json", "mechanism")
    native = _load_json(SUMMARY_DIR / "native.json", "native")
    preflight_gate = evaluate_preflight_gate(preflight)
    mechanism_gate = evaluate_mechanism_gate(mechanism)
    native_gate = evaluate_native_gate(native, mechanism_gate)
    decision = final_backend_decision(mechanism_gate, native_gate)
    ledger = {run["run_id"]: run for run in read_runs(GLOBAL_LEDGER)}
    run_ids = {
        "preflight": make_run_id("preflight"),
        "mechanism": make_run_id("mechanism"),
        "native": make_run_id("native"),
    }
    provenance_checks = {
        stage: run_id in ledger
        and ledger[run_id]["status"] == "completed"
        and _snapshot_check(ROOT / ledger[run_id]["results_path"])
        for stage, run_id in run_ids.items()
    }
    parent = _load_json(
        ROOT / ".codex_campaign/quality_aa_v1/VERDICT.json", "v1 verdict"
    )
    requirements = {
        "v1_preserved_invalid": parent.get("status") == "INVALID"
        and parent.get("rerun_authorized") is False,
        "preflight_valid": preflight_gate["passed"] is True,
        "dc_separated_metric": mechanism.get("floor_db")
        == preflight_gate["locked_floor_db"],
        "reference_converged": preflight_gate["checks"]["reference_x8_x16_convergence"],
        "adaa_independent": preflight_gate["adaa_status"] == "excluded_unalignable"
        and mechanism_gate["route_decisions_independent"] is True,
        "mechanism_route_available": bool(mechanism_gate["passing_routes"]),
        "native_route_available": bool(native_gate["passing_routes"]),
        "selected_backend": decision["selected_backend"] == "full_island_x2",
        "latency_within_48": native_gate["routes"]["full_island_x2"]["latency_samples"]
        <= 48,
        "python_cpp_parity": native_gate["routes"]["full_island_x2"]["checks"][
            "python_cpp_parity"
        ],
        "block64_realtime": native_gate["routes"]["full_island_x2"]["checks"][
            "block64_realtime"
        ],
        "provenance_all_runs": all(provenance_checks.values()),
        "physical_audio_absent": all(
            artifact.get("physical_audio_samples_read") == 0
            for artifact in (preflight, mechanism, native)
        ),
        "external_report_only_locked": protocol["scope"]["external_report_only_locked"]
        is True,
        "claim_boundary": protocol["scope"]["global_state_of_the_art_claim_allowed"]
        is False,
    }
    audit_passed = all(requirements.values())
    if not audit_passed:
        raise RuntimeError(
            "completion audit failed: "
            + ", ".join(name for name, passed in requirements.items() if not passed)
        )
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    provenance = capture_provenance(
        ROOT,
        RUN_DIR,
        ["uv", "run", "python", "scripts/campaigns/run_quality_aa_audit.py"],
    )
    write_new_json(
        RUN_DIR / "status.json",
        {"status": "running", "started_at": started_at, "finished_at": None},
    )
    evidence = {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "run_id": RUN_ID,
        "status": "supported",
        "requirements": requirements,
        "run_provenance": provenance_checks,
        "preflight_gate": preflight_gate,
        "mechanism_gate": mechanism_gate,
        "native_gate": native_gate,
        "decision": decision,
        "physical_audio_samples_read": 0,
        "external_report_only_accessed": False,
        "provenance": provenance,
    }
    write_new_json(RUN_DIR / "audit.json", evidence)
    write_new_json(SUMMARY_PATH, evidence)
    write_new_json(CAMPAIGN_DIR / "VERDICT.json", decision)
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    append_run(
        GLOBAL_LEDGER,
        {
            "date": finished_at,
            "run_id": RUN_ID,
            "phase": "QUALITY-AA-AUDIT",
            "model": "evidence_audit",
            "device": "synthetic_and_host_cpu",
            "seed": 0,
            "commit": provenance["git_head"],
            "config_sha256": digest_text(strict_json(protocol)),
            "data_sha256": digest_text(strict_json({"inputs": list(run_ids.values())})),
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
            "stage": "audit",
            "status": "passed",
            "evidence_path": str(SUMMARY_PATH.relative_to(ROOT)),
        },
    )
    replace_json(
        CAMPAIGN_DIR / "MATURITY.json",
        {
            "campaign_version": CAMPAIGN_VERSION,
            "current_stage": "terminal_go",
            "status": "terminal_go",
            "scientific_runs_launched": 4,
            "invalid": False,
        },
    )
    _write_campaign_documents(decision, str(SUMMARY_PATH.relative_to(ROOT)))
    print(json.dumps(decision, allow_nan=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
