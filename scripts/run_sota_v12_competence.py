#!/usr/bin/env python3
"""Run one authorized three-seed competence system for SOTA v1.2."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fssr_nam.campaign.amp_sota_prototype_v1_2 import (
    CAMPAIGN_PATH,
    CAMPAIGN_VERSION,
    CANDIDATE_FAMILY,
    DECISION_SOURCE_FILES,
    EVALUATION_UPDATES,
    SEEDS,
    SNAPSHOT_UPDATES,
    SYSTEMS,
    make_run_id,
    validate_clean_worktree,
    validate_cuda_device_properties,
    validate_repository_state,
    validate_stage_authorization,
)
from fssr_nam.campaign.amp_sota_v12_gates import (
    evaluate_competence_system_gate,
)
from fssr_nam.campaign.amp_sota_v12_registry import (
    append_gate_event,
    gate_decisions,
)
from fssr_nam.campaign.quality_aa_provenance import (
    capture_provenance,
    digest_array_bundle,
    digest_text,
    replace_json,
    strict_json,
    write_json_once_or_equal,
    write_new_json,
)
from fssr_nam.data.sota_v12_fixtures import build_sota_v12_system_episodes
from fssr_nam.reporting.ledger import append_run_once_or_equal, read_runs
from fssr_nam.training.sota_v12 import (
    SotaV12StabilityError,
    train_sota_v12_trajectory,
)

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / CAMPAIGN_PATH
GATE_LEDGER = CAMPAIGN_DIR / "GATE_LEDGER.jsonl"
GLOBAL_LEDGER = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
SUMMARY_DIR = ROOT / "experiments/summaries/amp_sota_prototype_v1_2"
STAGE_BY_SYSTEM = {
    "dynamic_primary": "competence_dynamic",
    "two_clippers_primary": "competence_two_clippers",
    "static_primary": "competence_static",
}
SOURCE_FILES = (*DECISION_SOURCE_FILES, "scripts/run_sota_v12_competence.py")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _summary_path(system: str) -> Path:
    return SUMMARY_DIR / f"competence_{system}.json"


def _save_states(run_dir: Path, states: dict[int, dict[str, torch.Tensor]]) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    for update, state in sorted(states.items()):
        torch.save(state, checkpoint_dir / f"update_{update}.pt")


def _save_predictions(
    model: torch.nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    device: torch.device,
    run_dir: Path,
) -> None:
    output_dir = run_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=False)
    model.eval()
    with torch.inference_mode():
        for episode in range(len(inputs)):
            source = torch.from_numpy(inputs[episode : episode + 1]).to(device)
            prediction = model(source)[0].detach().cpu().numpy().astype(np.float32)
            np.savez_compressed(
                output_dir / f"internal_dev_episode_{episode}.npz",
                input=inputs[episode],
                target=targets[episode],
                prediction=prediction,
            )


def _run_disk_gib() -> float:
    total = sum(
        child.stat().st_size
        for path in (ROOT / "experiments/runs").glob("sota_v1_2_*")
        if path.is_dir()
        for child in path.rglob("*")
        if child.is_file()
    )
    return total / (1024.0**3)


def _directory_gib(path: Path) -> float:
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file()) / (
        1024.0**3
    )


def _used_gpu_hours() -> float:
    return sum(
        float(row.get("elapsed_seconds", 0.0)) / 3600.0
        for row in read_runs(GLOBAL_LEDGER)
        if str(row.get("phase", "")).startswith("AMP-SOTA-PROTOTYPE-v1.2")
    )


def _v12_run_count() -> int:
    return sum(
        1
        for row in read_runs(GLOBAL_LEDGER)
        if str(row.get("phase", "")).startswith("AMP-SOTA-PROTOTYPE-v1.2")
    )


def _selection_sample_total() -> int:
    total = 0
    for path in SUMMARY_DIR.glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        total += int(value.get("selection_eligible_synthetic_samples_generated", 0))
    return total


def _ledger_entry(
    *,
    run_id: str,
    system: str,
    seed: int,
    status: str,
    failure_reason: str,
    run_dir: Path,
    provenance: dict[str, Any] | None,
    config_digest: str,
    data_digest: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "date": _now(),
        "run_id": run_id,
        "phase": "AMP-SOTA-PROTOTYPE-v1.2-COMPETENCE",
        "model": CANDIDATE_FAMILY,
        "device": f"synthetic_48khz_{system}",
        "seed": seed,
        "commit": (
            provenance["git_head"]
            if provenance is not None
            else "PROVENANCE_CAPTURE_FAILED"
        ),
        "config_sha256": config_digest,
        "data_sha256": data_digest,
        "status": status,
        "failure_reason": failure_reason,
        "results_path": str(run_dir.relative_to(ROOT)),
        "elapsed_seconds": elapsed_seconds,
        "gpu_hours": elapsed_seconds / 3600.0,
    }


def _write_terminal_verdict(verdict: str, summary: Path) -> None:
    write_json_once_or_equal(
        CAMPAIGN_DIR / "VERDICT.json",
        {
            "campaign_version": CAMPAIGN_VERSION,
            "verdict": verdict,
            "terminal_stage": summary.stem,
            "valid_scientific_gate_result": verdict != "INVALID",
            "evidence_path": str(summary.relative_to(ROOT)),
            "confirmation_opened": False,
            "physical_audio_samples_read": 0,
        },
    )


def _finalize_invalid(
    *,
    system: str,
    run_ids: list[str],
    trajectories: list[dict[str, Any]],
    generated_samples: int,
    error: Exception,
) -> None:
    stage = STAGE_BY_SYSTEM[system]
    summary_path = _summary_path(system)
    reason = f"{type(error).__name__}: {error}"
    summary = {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "stage": stage,
        "system": system,
        "status": "invalid",
        "verdict": "INVALID",
        "failure_reason": reason,
        "run_ids": run_ids,
        "trajectories": trajectories,
        "selection_eligible_synthetic_samples_generated": generated_samples,
        "physical_audio_samples_read": 0,
        "confirmation_opened": False,
    }
    write_json_once_or_equal(summary_path, summary)
    maturity_path = CAMPAIGN_DIR / "MATURITY.json"
    maturity = json.loads(maturity_path.read_text(encoding="utf-8"))
    maturity.update(
        {
            "current_stage": f"{stage}_invalid",
            "status": "terminal_invalid",
            "verdict": "INVALID",
            "scientific_runs_launched": _v12_run_count(),
            "selection_eligible_synthetic_samples_generated": (
                _selection_sample_total()
            ),
        }
    )
    replace_json(maturity_path, maturity)
    _write_terminal_verdict("INVALID", summary_path)
    append_gate_event(
        GATE_LEDGER,
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": stage,
            "status": "invalid",
            "evidence_path": str(summary_path.relative_to(ROOT)),
        },
    )


def run(system: str) -> int:
    validate_clean_worktree(ROOT)
    stage = STAGE_BY_SYSTEM[system]
    decisions = gate_decisions(GATE_LEDGER)
    validate_stage_authorization(stage, decisions)
    summary_path = _summary_path(system)
    if stage in decisions or summary_path.exists():
        raise RuntimeError(f"v1.2 {stage} is already frozen")
    protocol = validate_repository_state(ROOT)
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen CUDA device is unavailable")
    device_index = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device_index)
    validate_cuda_device_properties(
        protocol,
        name=device_properties.name,
        total_memory_bytes=device_properties.total_memory,
    )
    used_gpu_hours = _used_gpu_hours()
    remaining_gpu_hours = (
        float(protocol["resource_budget"]["v1_2_gpu_hours_maximum"]) - used_gpu_hours
    )
    if remaining_gpu_hours < len(SEEDS) * float(
        protocol["resource_budget"]["gpu_hours_per_trajectory_maximum"]
    ):
        raise RuntimeError("v1.2 GPU budget cannot cover the authorized triplet")
    if _run_disk_gib() >= float(protocol["resource_budget"]["v1_2_disk_gib_maximum"]):
        raise RuntimeError("v1.2 disk budget is exhausted")

    run_ids = [
        make_run_id(
            stage="competence",
            system=system,
            family=CANDIDATE_FAMILY,
            seed=seed,
        )
        for seed in SEEDS
    ]
    prior_ids = {row["run_id"] for row in read_runs(GLOBAL_LEDGER)}
    run_dirs = [ROOT / "experiments/runs" / run_id for run_id in run_ids]
    if prior_ids.intersection(run_ids) or any(path.exists() for path in run_dirs):
        raise RuntimeError("v1.2 run reuse or resume is forbidden")

    data = protocol["synthetic_data"]
    generated_samples = 0
    try:
        train_inputs, train_targets = build_sota_v12_system_episodes(
            system=system,
            source_seed=int(data["train_source_seed"]),
            episodes=int(data["train_episodes"]),
            samples=int(data["episode_samples"]),
        )
        generated_samples += 2 * train_inputs.size
        dev_inputs, dev_targets = build_sota_v12_system_episodes(
            system=system,
            source_seed=int(data["internal_dev_source_seed"]),
            episodes=int(data["internal_dev_episodes"]),
            samples=int(data["episode_samples"]),
        )
        generated_samples += 2 * dev_inputs.size
    except Exception as error:
        _finalize_invalid(
            system=system,
            run_ids=[],
            trajectories=[],
            generated_samples=generated_samples,
            error=error,
        )
        raise
    data_spec = {
        "generator": data["generator"],
        "target_renderer": data["target_renderer"],
        "system": system,
        "train_source_seed": data["train_source_seed"],
        "internal_dev_source_seed": data["internal_dev_source_seed"],
        "train_episodes": data["train_episodes"],
        "internal_dev_episodes": data["internal_dev_episodes"],
        "episode_samples": data["episode_samples"],
        "normalization": data["normalization"],
    }
    config_digest = digest_text(strict_json(protocol))
    training_data_digest = digest_array_bundle(
        {"train_inputs": train_inputs, "train_targets": train_targets}
    )
    evaluation_data_digest = digest_array_bundle(
        {"internal_dev_inputs": dev_inputs, "internal_dev_targets": dev_targets}
    )
    data_digest = digest_array_bundle(
        {
            "train_inputs": train_inputs,
            "train_targets": train_targets,
            "internal_dev_inputs": dev_inputs,
            "internal_dev_targets": dev_targets,
        }
    )
    data_manifest = data_spec | {
        "training_data_sha256": training_data_digest,
        "evaluation_data_sha256": evaluation_data_digest,
        "data_sha256": data_digest,
    }
    optimization = protocol["optimization"]
    device = torch.device("cuda", device_index)
    trajectories: list[dict[str, Any]] = []
    launched_ids: list[str] = []
    for seed, run_id, run_dir in zip(SEEDS, run_ids, run_dirs, strict=True):
        try:
            maximum_hours = float(
                protocol["resource_budget"]["gpu_hours_per_trajectory_maximum"]
            )
            if _used_gpu_hours() + maximum_hours > float(
                protocol["resource_budget"]["v1_2_gpu_hours_maximum"]
            ):
                raise RuntimeError("v1.2 GPU budget cannot cover the next trajectory")
            if _run_disk_gib() + float(
                protocol["resource_budget"]["run_disk_gib_maximum"]
            ) > float(protocol["resource_budget"]["v1_2_disk_gib_maximum"]):
                raise RuntimeError("v1.2 disk budget is exhausted before the next seed")
            run_dir.mkdir(parents=True, exist_ok=False)
        except Exception as error:
            _finalize_invalid(
                system=system,
                run_ids=launched_ids,
                trajectories=trajectories,
                generated_samples=generated_samples,
                error=error,
            )
            raise
        started_at = _now()
        started_clock = time.perf_counter()
        launched_ids.append(run_id)
        provenance: dict[str, Any] | None = None
        try:
            provenance = capture_provenance(
                ROOT,
                run_dir,
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/run_sota_v12_competence.py",
                    system,
                ],
                source_files=SOURCE_FILES,
                require_all_sources=True,
            )
            write_new_json(
                run_dir / "status.json",
                {
                    "status": "running",
                    "started_at": started_at,
                    "finished_at": None,
                },
            )
            write_new_json(run_dir / "data_manifest.json", data_manifest)
            model, result, states = train_sota_v12_trajectory(
                family=CANDIDATE_FAMILY,
                system=system,
                train_inputs=train_inputs,
                train_targets=train_targets,
                internal_dev_inputs=dev_inputs,
                internal_dev_targets=dev_targets,
                snapshot_updates=SNAPSHOT_UPDATES,
                evaluation_updates=EVALUATION_UPDATES,
                common_preroll_samples_after_alignment=int(
                    optimization["common_preroll_samples_after_alignment"]
                ),
                chunk_samples=int(optimization["training_chunk_samples"]),
                learning_rate=float(optimization["learning_rate"]),
                projection_gain_weight=float(optimization["projection_gain_weight"]),
                gradient_clip_norm=float(optimization["gradient_clip_norm"]),
                maximum_elapsed_seconds=(
                    maximum_hours * 3600.0
                    - float(
                        protocol["resource_budget"][
                            "artifact_finalization_reserve_seconds"
                        ]
                    )
                ),
                device=device,
                seed=seed,
                checkpoint_callback=lambda update, _row, seed_value=seed: print(
                    f"v12_checkpoint:{system}:seed={seed_value}:update={update}",
                    flush=True,
                ),
            )
            payload = asdict(result) | {
                "status": "completed",
                "config_sha256": config_digest,
                "data_sha256": data_digest,
                "training_data_sha256": training_data_digest,
                "evaluation_data_sha256": evaluation_data_digest,
                "evaluation_source_seed": data["internal_dev_source_seed"],
                "post_training_fit_applied": False,
            }
            _save_states(run_dir, states)
            _save_predictions(
                model, dev_inputs, dev_targets, device=device, run_dir=run_dir
            )
            if _directory_gib(run_dir) > float(
                protocol["resource_budget"]["run_disk_gib_maximum"]
            ):
                raise RuntimeError("v1.2 run exceeded its frozen disk allowance")
            write_new_json(run_dir / "result.json", payload)
            elapsed = time.perf_counter() - started_clock
            replace_json(
                run_dir / "status.json",
                {
                    "status": "completed",
                    "started_at": started_at,
                    "finished_at": _now(),
                    "failure_reason": "",
                },
            )
            append_run_once_or_equal(
                GLOBAL_LEDGER,
                _ledger_entry(
                    run_id=run_id,
                    system=system,
                    seed=seed,
                    status="completed",
                    failure_reason="",
                    run_dir=run_dir,
                    provenance=provenance,
                    config_digest=config_digest,
                    data_digest=data_digest,
                    elapsed_seconds=elapsed,
                ),
            )
            trajectories.append(payload)
            print(
                f"v12_completed:{system}:seed={seed}:hours={elapsed / 3600.0:.6f}",
                flush=True,
            )
            del model
            torch.cuda.empty_cache()
        except SotaV12StabilityError as error:
            elapsed = time.perf_counter() - started_clock
            reason = f"{type(error).__name__}: {error}"
            failure = {
                "family": CANDIDATE_FAMILY,
                "system": system,
                "seed": seed,
                "status": "stability_failed",
                "failure_reason": reason,
            }
            write_new_json(run_dir / "result.json", failure)
            replace_json(
                run_dir / "status.json",
                {
                    "status": "failed",
                    "failure_class": "scientific_stability",
                    "started_at": started_at,
                    "finished_at": _now(),
                    "failure_reason": reason,
                },
            )
            append_run_once_or_equal(
                GLOBAL_LEDGER,
                _ledger_entry(
                    run_id=run_id,
                    system=system,
                    seed=seed,
                    status="failed",
                    failure_reason=reason,
                    run_dir=run_dir,
                    provenance=provenance,
                    config_digest=config_digest,
                    data_digest=data_digest,
                    elapsed_seconds=elapsed,
                ),
            )
            trajectories.append(failure)
            torch.cuda.empty_cache()
        except Exception as error:
            elapsed = time.perf_counter() - started_clock
            reason = f"{type(error).__name__}: {error}"
            failed_status = {
                "status": "failed",
                "failure_class": "operational_or_instrumentation",
                "started_at": started_at,
                "finished_at": _now(),
                "failure_reason": reason,
            }
            status_path = run_dir / "status.json"
            if status_path.exists():
                replace_json(status_path, failed_status)
            else:
                write_new_json(status_path, failed_status)
            append_run_once_or_equal(
                GLOBAL_LEDGER,
                _ledger_entry(
                    run_id=run_id,
                    system=system,
                    seed=seed,
                    status="failed",
                    failure_reason=reason,
                    run_dir=run_dir,
                    provenance=provenance,
                    config_digest=config_digest,
                    data_digest=data_digest,
                    elapsed_seconds=elapsed,
                ),
            )
            _finalize_invalid(
                system=system,
                run_ids=launched_ids,
                trajectories=trajectories,
                generated_samples=generated_samples,
                error=error,
            )
            raise

    try:
        gate = evaluate_competence_system_gate(
            trajectories, system=system, protocol=protocol
        )
    except Exception as error:
        _finalize_invalid(
            system=system,
            run_ids=launched_ids,
            trajectories=trajectories,
            generated_samples=generated_samples,
            error=error,
        )
        raise
    status = "passed" if gate["passed"] else "no-go"
    summary = {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "stage": stage,
        "system": system,
        "status": status,
        "run_ids": run_ids,
        "trajectory_count": len(trajectories),
        "trajectories": trajectories,
        "gate": gate,
        "selection_eligible_synthetic_samples_generated": generated_samples,
        "physical_audio_samples_read": 0,
        "confirmation_opened": False,
        "fm9_outputs_accessed": False,
        "gpu_hours_used_after_stage": _used_gpu_hours(),
        "run_disk_gib_after_stage": _run_disk_gib(),
    }
    write_json_once_or_equal(summary_path, summary)
    maturity_path = CAMPAIGN_DIR / "MATURITY.json"
    maturity = json.loads(maturity_path.read_text(encoding="utf-8"))
    maturity["scientific_runs_launched"] = _v12_run_count()
    maturity["selection_eligible_synthetic_samples_generated"] = (
        _selection_sample_total()
    )
    if gate["passed"]:
        next_stage = {
            "dynamic_primary": "competence_two_clippers_authorized",
            "two_clippers_primary": "competence_static_authorized",
            "static_primary": "slow_value_authorized",
        }[system]
        maturity["current_stage"] = next_stage
    else:
        maturity.update(
            {
                "current_stage": f"{stage}_no_go",
                "status": "terminal_no_go",
                "verdict": gate["verdict"],
            }
        )
        _write_terminal_verdict(str(gate["verdict"]), summary_path)
    replace_json(maturity_path, maturity)
    append_gate_event(
        GATE_LEDGER,
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": stage,
            "status": status,
            "evidence_path": str(summary_path.relative_to(ROOT)),
        },
    )
    print(json.dumps(gate, allow_nan=False, indent=2, sort_keys=True))
    return 0 if gate["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("system", choices=SYSTEMS)
    arguments = parser.parse_args()
    return run(arguments.system)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"SOTA v1.2 competence failed closed: {error}", file=sys.stderr)
        raise
