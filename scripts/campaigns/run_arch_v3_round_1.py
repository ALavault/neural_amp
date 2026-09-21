#!/usr/bin/env python3
"""Run the immutable paired AMP-QUALITY-ARCH-v3 round-one matrix."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from fssr_nam.campaign.amp_arch_v3_gates import evaluate_round_one_gate
from fssr_nam.campaign.amp_arch_v3_registry import append_gate_event, gate_decisions
from fssr_nam.campaign.amp_quality_arch_v3 import (
    CAMPAIGN_PATH,
    CAMPAIGN_VERSION,
    INITIAL_FAMILIES,
    PRIMARY_SYSTEMS,
    ROUND_ONE_CHECKPOINTS,
    ROUND_ONE_SEEDS,
    load_round_one_lock,
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
from fssr_nam.data.arch_v3_fixtures import build_arch_v3_episodes
from fssr_nam.reporting.ledger import append_run, read_runs
from fssr_nam.training.arch_v3 import train_arch_v3_trajectory

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / CAMPAIGN_PATH
GATE_LEDGER = CAMPAIGN_DIR / "GATE_LEDGER.jsonl"
GLOBAL_LEDGER = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
SUMMARY = ROOT / "experiments/summaries/amp_quality_arch_v3/round_1.json"
PREFLIGHT = ROOT / "experiments/summaries/amp_quality_arch_v3/preflight.json"
FEASIBILITY = CAMPAIGN_DIR / "TRAINING_FEASIBILITY_V2.json"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _save_states(run_dir: Path, states: dict[int, dict[str, torch.Tensor]]) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    for update, state in sorted(states.items()):
        torch.save(state, checkpoint_dir / f"update_{update}.pt")


def _ledger_entry(
    *,
    run_id: str,
    family: str,
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
        "phase": "AMP-QUALITY-ARCH-v3-ROUND-1",
        "model": family,
        "device": "synthetic_48khz_internal_dev",
        "seed": seed,
        "commit": provenance["git_head"],
        "config_sha256": config_digest,
        "data_sha256": data_digest,
        "status": status,
        "failure_reason": failure_reason,
        "results_path": str(run_dir.relative_to(ROOT)),
    }


def _record_invalid(error: Exception) -> None:
    if SUMMARY.exists() or (CAMPAIGN_DIR / "ROUND_1_INVALID.json").exists():
        return
    launched = sorted(
        path.name for path in (ROOT / "experiments/runs").glob("arch_v3_round_1_*")
    )
    if not launched:
        return
    evidence = {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "stage": "round_1",
        "status": "invalid",
        "verdict": "INVALID",
        "reason": str(error),
        "run_directories_created": launched,
        "resume_allowed": False,
        "validation_source_accessed": False,
        "physical_audio_samples_read": 0,
    }
    invalid_path = CAMPAIGN_DIR / "ROUND_1_INVALID.json"
    write_new_json(invalid_path, evidence)
    append_gate_event(
        GATE_LEDGER,
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "round_1",
            "status": "invalid",
            "evidence_path": str(invalid_path.relative_to(ROOT)),
        },
    )
    write_new_json(
        CAMPAIGN_DIR / "VERDICT.json",
        {
            "campaign_version": CAMPAIGN_VERSION,
            "verdict": "INVALID",
            "terminal_stage": "round_1",
            "valid_scientific_gate_result": False,
            "evidence_path": str(invalid_path.relative_to(ROOT)),
        },
    )
    maturity_path = CAMPAIGN_DIR / "MATURITY.json"
    maturity = json.loads(maturity_path.read_text(encoding="utf-8"))
    maturity.update(
        {
            "current_stage": "round_1_invalid",
            "scientific_runs_launched": len(launched),
            "status": "terminal_invalid",
            "verdict": "INVALID",
        }
    )
    replace_json(maturity_path, maturity)


def main() -> int:
    decisions = gate_decisions(GATE_LEDGER)
    validate_stage_authorization("round_1", decisions)
    protocol = validate_repository_state(ROOT, require_frozen=True)
    round_lock = load_round_one_lock(ROOT, protocol)
    if SUMMARY.exists():
        raise RuntimeError("v3 round-one summary already exists")
    feasibility = json.loads(FEASIBILITY.read_text(encoding="utf-8"))
    if feasibility.get("status") != "passed" or not all(
        feasibility.get("checks", {}).values()
    ):
        raise RuntimeError("v3 frozen training feasibility has not passed")
    if feasibility.get("validation_source_accessed") is not False:
        raise RuntimeError("v3 feasibility crossed the validation boundary")
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed":
        raise RuntimeError("v3 representability preflight has not passed")
    initial_scale = float(preflight["initial_residual_scale"])
    family_parameters = {
        row["family"]: int(row["parameters"]) for row in preflight["models"]
    }
    run_ids = [
        make_run_id("round_1", system, family, seed)
        for family in INITIAL_FAMILIES
        for system in PRIMARY_SYSTEMS
        for seed in ROUND_ONE_SEEDS
    ]
    prior_ids = {entry["run_id"] for entry in read_runs(GLOBAL_LEDGER)}
    run_dirs = [ROOT / "experiments/runs" / run_id for run_id in run_ids]
    if prior_ids.intersection(run_ids) or any(path.exists() for path in run_dirs):
        raise RuntimeError("v3 round-one run reuse or resume is forbidden")
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen CUDA device is unavailable for v3 round one")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    data = protocol["synthetic_data"]
    train_arrays = build_arch_v3_episodes(
        source_seed=int(data["train_source_seed"]),
        episodes=int(data["train_episodes"]),
        samples=int(data["episode_samples"]),
    )
    internal_dev_arrays = build_arch_v3_episodes(
        source_seed=int(data["internal_dev_source_seed"]),
        episodes=int(data["internal_dev_episodes"]),
        samples=int(data["episode_samples"]),
    )
    config_digest = digest_text(
        strict_json({"protocol": protocol, "round_1_lock": round_lock})
    )
    optimization = round_lock["optimization"]
    trajectories: list[dict[str, Any]] = []
    campaign_started = time.perf_counter()
    for family in INITIAL_FAMILIES:
        for system in PRIMARY_SYSTEMS:
            data_digest = digest_text(
                strict_json(
                    {
                        "generator": data["generator"],
                        "system": system,
                        "train_source_seed": data["train_source_seed"],
                        "internal_dev_source_seed": data["internal_dev_source_seed"],
                        "episode_samples": data["episode_samples"],
                        "train_episodes": data["train_episodes"],
                        "internal_dev_episodes": data["internal_dev_episodes"],
                        "normalization": data["normalization"],
                    }
                )
            )
            for seed in ROUND_ONE_SEEDS:
                run_id = make_run_id("round_1", system, family, seed)
                run_dir = ROOT / "experiments/runs" / run_id
                started_at = _now()
                trajectory_started = time.perf_counter()
                run_dir.mkdir(parents=True, exist_ok=False)
                provenance = capture_provenance(
                    ROOT,
                    run_dir,
                    ["uv", "run", "python", "scripts/campaigns/run_arch_v3_round_1.py"],
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
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    model, result, states = train_arch_v3_trajectory(
                        family=family,
                        system=system,
                        train_inputs=train_arrays[system][0],
                        train_targets=train_arrays[system][1],
                        internal_dev_inputs=internal_dev_arrays[system][0],
                        internal_dev_targets=internal_dev_arrays[system][1],
                        profile=str(optimization["profile"]),
                        initial_residual_scale=initial_scale,
                        checkpoints=ROUND_ONE_CHECKPOINTS,
                        scored_start=int(data["preroll_samples"]),
                        chunk_samples=int(optimization["training_chunk_samples"]),
                        learning_rate=float(optimization["learning_rate"]),
                        projection_gain_weight=float(
                            optimization["projection_gain_weight"]
                        ),
                        device=torch.device("cuda"),
                        seed=seed,
                    )
                    payload = asdict(result)
                    _save_states(run_dir, states)
                    write_new_json(run_dir / "result.json", payload)
                    write_new_json(
                        run_dir / "resources.json",
                        {
                            "elapsed_seconds": time.perf_counter() - trajectory_started,
                            "peak_cuda_memory_bytes": int(
                                torch.cuda.max_memory_allocated()
                            ),
                            "cuda_device": torch.cuda.get_device_name(0),
                        },
                    )
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
                            seed=seed,
                            status="completed",
                            failure_reason="",
                            run_dir=run_dir,
                            provenance=provenance,
                            config_digest=config_digest,
                            data_digest=data_digest,
                        ),
                    )
                    trajectories.append(payload)
                    final_metrics = payload["checkpoints"][-1]["internal_dev"]
                    print(
                        f"arch_v3_round_1_completed:{family}:{system}:seed={seed}:"
                        f"esr={final_metrics['esr']:.9g}:"
                        f"gain={final_metrics['gain_error']:.9g}:"
                        f"corr={final_metrics['correlation']:.9g}",
                        flush=True,
                    )
                    del model, states
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

    gate = evaluate_round_one_gate(
        trajectories, protocol, round_lock, family_parameters
    )
    status = "passed" if gate["passed"] else "failed"
    summary = {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "stage": "round_1",
        "status": status,
        "run_ids": run_ids,
        "trajectory_count": len(trajectories),
        "trajectories": trajectories,
        "gate": gate,
        "elapsed_seconds": time.perf_counter() - campaign_started,
        "scientific_runs_launched": len(run_ids),
        "evidence_tier": "INTERNAL_DEV",
        "validation_source_accessed": False,
        "physical_audio_samples_read": 0,
        "sealed_outputs_accessed": {"blackstar": False, "ua1176": False},
        "external_report_only_locked": True,
        "result_dependent_retries": 0,
    }
    write_new_json(SUMMARY, summary)
    append_gate_event(
        GATE_LEDGER,
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "round_1",
            "status": status,
            "evidence_path": str(SUMMARY.relative_to(ROOT)),
        },
    )
    maturity_path = CAMPAIGN_DIR / "MATURITY.json"
    maturity = json.loads(maturity_path.read_text(encoding="utf-8"))
    maturity.update(
        {
            "current_stage": f"round_1_{status}",
            "exploration_rounds_completed": 1,
            "scientific_runs_launched": len(run_ids),
            "round_1_selected_family": gate["selected_family"],
        }
    )
    replace_json(maturity_path, maturity)
    state = (
        "# État\n\n"
        "- Lignée : `AMP-QUALITY-ARCH-v3`.\n"
        "- Préflight de représentabilité : `passed`.\n"
        "- Faisabilité GPU : `passed`.\n"
        f"- Round 1 : `{status}` ({len(run_ids)} trajectoires immuables).\n"
        f"- Famille retenue : `{gate['selected_family']}`.\n"
        "- Validation et audio physique lus : 0 échantillon.\n"
        "- Blackstar, UA1176 et EXTERNAL_REPORT_ONLY : verrouillés.\n"
    )
    (CAMPAIGN_DIR / "STATE.md").write_text(state, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        _record_invalid(error)
        print(f"architecture v3 round one failed closed: {error}", file=sys.stderr)
        raise
