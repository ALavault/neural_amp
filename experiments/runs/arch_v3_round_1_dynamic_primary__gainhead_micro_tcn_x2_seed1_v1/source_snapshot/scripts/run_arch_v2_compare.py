#!/usr/bin/env python3
"""Run the frozen paired architecture comparison after competence passes."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from fssr_nam.campaign.amp_arch_v2_gates import evaluate_comparison_gate
from fssr_nam.campaign.amp_arch_v2_registry import append_gate_event, gate_decisions
from fssr_nam.campaign.amp_competence_arch_v2 import (
    ALL_FAMILIES,
    CAMPAIGN_PATH,
    CAMPAIGN_VERSION,
    SEEDS,
    SYSTEMS,
    make_run_id,
    validate_repository_state,
    validate_stage_authorization,
)
from fssr_nam.campaign.quality_aa_provenance import (
    capture_provenance,
    digest_text,
    replace_json,
    strict_json,
    write_new_json,
)
from fssr_nam.data.arch_v2_fixtures import build_arch_v2_episodes
from fssr_nam.reporting.ledger import append_run, read_runs
from fssr_nam.training.arch_v2 import train_checkpointed_trajectory

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / CAMPAIGN_PATH
GATE_LEDGER = CAMPAIGN_DIR / "GATE_LEDGER.jsonl"
GLOBAL_LEDGER = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
COMPETENCE_SUMMARY = (
    ROOT / "experiments/summaries/amp_competence_arch_v2/competence.json"
)
SUMMARY = ROOT / "experiments/summaries/amp_competence_arch_v2/comparison.json"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _ledger_entry(
    *,
    run_id: str,
    family: str,
    system: str,
    seed: int,
    status: str,
    failure_reason: str,
    run_dir: Path,
    provenance: dict[str, Any],
    config_digest: str,
    data_digest: str,
) -> dict[str, Any]:
    return {
        "date": _now(),
        "run_id": run_id,
        "phase": "AMP-COMPETENCE-ARCH-v2-COMPARISON",
        "model": family,
        "device": f"synthetic_{system}",
        "seed": seed,
        "commit": provenance["git_head"],
        "config_sha256": config_digest,
        "data_sha256": data_digest,
        "status": status,
        "failure_reason": failure_reason,
        "results_path": str(run_dir.relative_to(ROOT)),
    }


def main() -> int:
    decisions = gate_decisions(GATE_LEDGER)
    validate_stage_authorization("comparison", decisions)
    protocol = validate_repository_state(ROOT, require_frozen=True)
    if SUMMARY.exists():
        raise RuntimeError("v2 comparison summary already exists")
    competence = json.loads(COMPETENCE_SUMMARY.read_text(encoding="utf-8"))
    budget = competence.get("gate", {}).get("selected_budget_updates")
    if not isinstance(budget, int) or budget < 1:
        raise RuntimeError("v2 comparison lacks a valid competence-selected budget")
    run_ids = [
        make_run_id("comparison", system, family, seed)
        for family in ALL_FAMILIES
        for system in SYSTEMS
        for seed in SEEDS
    ]
    if len(run_ids) > int(
        protocol["resource_budget"]["comparison_run_directories_maximum"]
    ):
        raise RuntimeError("v2 comparison run cap is exceeded")
    prior_ids = {entry["run_id"] for entry in read_runs(GLOBAL_LEDGER)}
    run_dirs = [ROOT / "experiments/runs" / run_id for run_id in run_ids]
    if prior_ids.intersection(run_ids) or any(path.exists() for path in run_dirs):
        raise RuntimeError("v2 comparison run reuse or resume is forbidden")
    if not torch.cuda.is_available():
        raise RuntimeError("frozen CUDA comparison device is unavailable")

    data_config = protocol["synthetic_data"]
    train_arrays = build_arch_v2_episodes(
        source_seed=int(data_config["comparison_train_source_seed"]),
        episodes=int(data_config["train_episodes"]),
        samples=int(data_config["episode_samples"]),
    )
    validation_arrays = build_arch_v2_episodes(
        source_seed=int(data_config["comparison_validation_source_seed"]),
        episodes=int(data_config["validation_episodes"]),
        samples=int(data_config["episode_samples"]),
    )
    config_digest = digest_text(strict_json(protocol))
    data_digest = digest_text(
        strict_json(
            {
                "generator": data_config["generator"],
                "train_source_seed": data_config["comparison_train_source_seed"],
                "validation_source_seed": data_config[
                    "comparison_validation_source_seed"
                ],
                "episode_samples": data_config["episode_samples"],
                "train_episodes": data_config["train_episodes"],
                "validation_episodes": data_config["validation_episodes"],
            }
        )
    )
    optimization = protocol["optimization"]
    auxiliary_weights = protocol["comparison"]["auxiliary_weight_by_family"]
    trajectories: list[dict[str, Any]] = []
    for family in ALL_FAMILIES:
        for system in SYSTEMS:
            for seed in SEEDS:
                run_id = make_run_id("comparison", system, family, seed)
                run_dir = ROOT / "experiments/runs" / run_id
                started_at = _now()
                run_dir.mkdir(parents=True, exist_ok=False)
                provenance = capture_provenance(
                    ROOT,
                    run_dir,
                    ["uv", "run", "python", "scripts/run_arch_v2_compare.py"],
                )
                write_new_json(
                    run_dir / "status.json",
                    {
                        "status": "running",
                        "started_at": started_at,
                        "finished_at": None,
                    },
                )
                try:
                    model, result, states = train_checkpointed_trajectory(
                        family=family,
                        system=system,
                        train_inputs=torch.from_numpy(train_arrays[system][0]),
                        train_targets=torch.from_numpy(train_arrays[system][1]),
                        validation_inputs=torch.from_numpy(
                            validation_arrays[system][0]
                        ),
                        validation_targets=torch.from_numpy(
                            validation_arrays[system][1]
                        ),
                        profile=str(optimization["profile"]),
                        checkpoints=(budget,),
                        tail_samples=int(optimization["loss_tail_samples"]),
                        learning_rate=float(optimization["learning_rate"]),
                        auxiliary_weight=float(auxiliary_weights[family]),
                        device=torch.device("cuda"),
                        seed=seed,
                    )
                    payload = asdict(result)
                    checkpoint_dir = run_dir / "checkpoints"
                    checkpoint_dir.mkdir(parents=True, exist_ok=False)
                    torch.save(states[budget], checkpoint_dir / f"update_{budget}.pt")
                    write_new_json(run_dir / "result.json", payload)
                    finished_at = _now()
                    replace_json(
                        run_dir / "status.json",
                        {
                            "status": "completed",
                            "started_at": started_at,
                            "finished_at": finished_at,
                        },
                    )
                    append_run(
                        GLOBAL_LEDGER,
                        _ledger_entry(
                            run_id=run_id,
                            family=family,
                            system=system,
                            seed=seed,
                            status="completed",
                            failure_reason="",
                            run_dir=run_dir,
                            provenance=provenance,
                            config_digest=config_digest,
                            data_digest=data_digest,
                        ),
                    )
                    validation = payload["checkpoints"][0]["validation"]
                    trajectories.append(
                        {
                            "run_id": run_id,
                            "family": family,
                            "system": system,
                            "seed": seed,
                            "budget_updates": budget,
                            "validation": validation,
                        }
                    )
                    print(
                        f"arch_v2_comparison_completed:{family}:{system}:seed={seed}:"
                        f"esr={validation['esr']:.9g}",
                        flush=True,
                    )
                    del model
                    torch.cuda.empty_cache()
                except Exception as error:
                    finished_at = _now()
                    replace_json(
                        run_dir / "status.json",
                        {
                            "status": "failed",
                            "started_at": started_at,
                            "finished_at": finished_at,
                            "failure_reason": str(error),
                        },
                    )
                    append_run(
                        GLOBAL_LEDGER,
                        _ledger_entry(
                            run_id=run_id,
                            family=family,
                            system=system,
                            seed=seed,
                            status="failed",
                            failure_reason=str(error),
                            run_dir=run_dir,
                            provenance=provenance,
                            config_digest=config_digest,
                            data_digest=data_digest,
                        ),
                    )
                    raise

    gate = evaluate_comparison_gate(trajectories, protocol)
    status = "passed" if gate["passed"] else "failed"
    verdict = "GO-ARCH-v2" if gate["passed"] else "NO-GO-ARCH-v2"
    summary = {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "stage": "comparison",
        "status": status,
        "verdict": verdict,
        "selected_budget_updates": budget,
        "run_ids": run_ids,
        "trajectory_count": len(trajectories),
        "trajectories": trajectories,
        "gate": gate,
        "physical_audio_samples_read": 0,
        "sealed_outputs_accessed": {"blackstar": False, "ua1176": False},
        "external_report_only_locked": True,
    }
    write_new_json(SUMMARY, summary)
    append_gate_event(
        GATE_LEDGER,
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "comparison",
            "status": status,
            "evidence_path": str(SUMMARY.relative_to(ROOT)),
        },
    )
    write_new_json(
        CAMPAIGN_DIR / "VERDICT.json",
        {
            "campaign_version": CAMPAIGN_VERSION,
            "verdict": verdict,
            "terminal_stage": "comparison",
            "valid_scientific_gate_result": True,
            "evidence_path": str(SUMMARY.relative_to(ROOT)),
        },
    )
    maturity_path = CAMPAIGN_DIR / "MATURITY.json"
    maturity = json.loads(maturity_path.read_text(encoding="utf-8"))
    maturity.update(
        {
            "comparison_runs_launched": len(run_ids),
            "current_stage": "comparison_complete",
            "status": "terminal_go" if gate["passed"] else "terminal_no_go",
            "verdict": verdict,
        }
    )
    replace_json(maturity_path, maturity)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"architecture v2 comparison failed closed: {error}", file=sys.stderr)
        raise
