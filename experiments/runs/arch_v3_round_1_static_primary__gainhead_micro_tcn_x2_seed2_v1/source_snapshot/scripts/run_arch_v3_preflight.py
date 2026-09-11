#!/usr/bin/env python3
"""Freeze and execute the train-only AMP-QUALITY-ARCH-v3 preflight."""

from __future__ import annotations

import json
from pathlib import Path

from fssr_nam.campaign.amp_arch_v3_gates import (
    analyze_history_collisions,
    analyze_train_ranges,
    evaluate_representability_gate,
)
from fssr_nam.campaign.amp_arch_v3_registry import append_gate_event, gate_decisions
from fssr_nam.campaign.amp_quality_arch_v3 import (
    CAMPAIGN_PATH,
    CAMPAIGN_VERSION,
    INITIAL_FAMILIES,
    validate_repository_state,
    validate_stage_authorization,
)
from fssr_nam.campaign.quality_aa_provenance import (
    replace_json,
    write_new_json,
    write_new_text,
)
from fssr_nam.data.arch_v3_fixtures import build_arch_v3_episodes
from fssr_nam.models import build_arch_v3_candidate
from fssr_nam.reporting.ledger import read_runs

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / CAMPAIGN_PATH
GATE_LEDGER = CAMPAIGN_DIR / "GATE_LEDGER.jsonl"
GLOBAL_LEDGER = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
SUMMARY = ROOT / "experiments/summaries/amp_quality_arch_v3/preflight.json"


def main() -> int:
    decisions = gate_decisions(GATE_LEDGER)
    validate_stage_authorization("preflight", decisions)
    protocol = validate_repository_state(ROOT)
    lock_path = CAMPAIGN_DIR / "PROTOCOL_LOCK.yaml"
    if lock_path.exists() or SUMMARY.exists():
        raise RuntimeError("v3 preflight evidence is already frozen")
    if any(
        entry["run_id"].startswith("arch_v3_") for entry in read_runs(GLOBAL_LEDGER)
    ):
        raise RuntimeError("v3 preflight found a prior scientific ledger entry")
    if any((ROOT / "experiments/runs").glob("arch_v3_*")):
        raise RuntimeError("v3 preflight found an unregistered run directory")

    source_text = (ROOT / "configs/amp_quality_arch_v3/protocol.yaml").read_text(
        encoding="utf-8"
    )
    write_new_text(lock_path, source_text)
    data = protocol["synthetic_data"]
    episodes = build_arch_v3_episodes(
        source_seed=int(data["train_source_seed"]),
        episodes=int(data["train_episodes"]),
        samples=int(data["episode_samples"]),
    )
    range_rows = analyze_train_ranges(episodes, protocol)
    memory_rows = analyze_history_collisions(protocol)
    gate = evaluate_representability_gate(range_rows, memory_rows, protocol)
    initial_scale = max(
        float(row["recommended_initial_residual_scale"])
        for row in range_rows
        if row["primary"]
    )
    model_rows = []
    for family in INITIAL_FAMILIES:
        model = build_arch_v3_candidate(
            family,
            profile=str(protocol["architectures"]["profile"]),
            initial_residual_scale=initial_scale,
        )
        expected = protocol["architectures"][family]
        if model.receptive_field_samples != int(
            expected["physical_receptive_field_samples"]
        ):
            raise RuntimeError(f"v3 receptive field changed: {family}")
        if model.latency_samples != int(protocol["architectures"]["latency_samples"]):
            raise RuntimeError(f"v3 latency changed: {family}")
        model_rows.append(
            {
                "family": family,
                "parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "physical_receptive_field_samples": model.receptive_field_samples,
                "latency_samples": model.latency_samples,
                "initial_residual_scale": float(model.residual_scale.detach()),
                "slow_state": model.uses_slow_state,
            }
        )
    status = "passed" if gate["passed"] else "failed"
    summary = {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "stage": "preflight",
        "status": status,
        "protocol_lock_written_before_train_analysis": True,
        "evidence_tier": "train_only",
        "train_source_seed": data["train_source_seed"],
        "validation_source_accessed": False,
        "physical_audio_samples_read": 0,
        "sealed_outputs_accessed": {"blackstar": False, "ua1176": False},
        "external_report_only_locked": True,
        "scientific_runs_launched": 0,
        "initial_residual_scale": initial_scale,
        "models": model_rows,
        "gate": gate,
    }
    write_new_json(SUMMARY, summary)
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
        "preflight_passed" if gate["passed"] else "representability_failed"
    )
    if not gate["passed"]:
        maturity["status"] = "terminal_no_go"
        maturity["verdict"] = "NO-GO-REPRESENTABILITY-v3"
        write_new_json(
            CAMPAIGN_DIR / "VERDICT.json",
            {
                "campaign_version": CAMPAIGN_VERSION,
                "verdict": "NO-GO-REPRESENTABILITY-v3",
                "valid_scientific_gate_result": True,
                "evidence_path": str(SUMMARY.relative_to(ROOT)),
            },
        )
    replace_json(maturity_path, maturity)
    state = (
        "# État\n\n"
        "- Lignée : `AMP-QUALITY-ARCH-v3`.\n"
        f"- Preflight de représentabilité : `{status}`.\n"
        "- Protocole : gelé avant l'analyse train-only.\n"
        "- Runs scientifiques v3 : 0.\n"
        f"- Familles entraînables : {len(gate['training_eligible_families'])}.\n"
        f"- Familles compatibles mémoire : {len(gate['memory_capable_families'])}.\n"
        "- Audio physique lu : 0 échantillon.\n"
        "- Validation, Blackstar, UA1176 et EXTERNAL_REPORT_ONLY : verrouillés.\n"
    )
    (CAMPAIGN_DIR / "STATE.md").write_text(state, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
