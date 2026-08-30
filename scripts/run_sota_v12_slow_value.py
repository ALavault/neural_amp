#!/usr/bin/env python3
"""Run fresh candidate reevaluation, zero-modulation controls, and slow gate."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
    CONTROL_FAMILY,
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
from fssr_nam.campaign.amp_sota_v12_gates import evaluate_slow_value_gate
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
from fssr_nam.models.sota_v12 import build_sota_v12_model
from fssr_nam.reporting.ledger import append_run_once_or_equal, read_runs
from fssr_nam.training.sota_v12 import (
    SotaV12StabilityError,
    evaluate_sota_v12_model,
    train_sota_v12_trajectory,
)

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / CAMPAIGN_PATH
GATE_LEDGER = CAMPAIGN_DIR / "GATE_LEDGER.jsonl"
GLOBAL_LEDGER = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
SUMMARY_DIR = ROOT / "experiments/summaries/amp_sota_prototype_v1_2"
SUMMARY = SUMMARY_DIR / "slow_value.json"
SOURCE_FILES = (*DECISION_SOURCE_FILES, "scripts/run_sota_v12_slow_value.py")
PHASE_CANDIDATE_EVAL = "AMP-SOTA-PROTOTYPE-v1.2-SLOW-VALUE-EVALUATION"
PHASE_CONTROL = "AMP-SOTA-PROTOTYPE-v1.2-SLOW-VALUE"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=False)
    paths: list[str] = []
    model.eval()
    with torch.inference_mode():
        for episode in range(len(inputs)):
            source = torch.from_numpy(inputs[episode : episode + 1]).to(device)
            prediction = model(source)[0].detach().cpu().numpy().astype(np.float32)
            path = output_dir / f"slow_value_eval_episode_{episode}.npz"
            np.savez_compressed(
                path,
                input=inputs[episode],
                target=targets[episode],
                prediction=prediction,
            )
            paths.append(str(path.relative_to(ROOT)))
    return paths


def _ledger_entry(
    *,
    run_id: str,
    phase: str,
    family: str,
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
        "phase": phase,
        "model": family,
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


def _candidate_summaries() -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for system in SYSTEMS:
        path = SUMMARY_DIR / f"competence_{system}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("status") != "passed"
            or value.get("gate", {}).get("passed") is not True
        ):
            raise RuntimeError(f"candidate competence is not passed for {system}")
        expected_ids = [
            make_run_id(
                stage="competence",
                system=system,
                family=CANDIDATE_FAMILY,
                seed=seed,
            )
            for seed in SEEDS
        ]
        if value.get("run_ids") != expected_ids:
            raise RuntimeError(f"candidate run IDs changed for {system}")
        summaries[system] = value
    return summaries


def _assert_candidate_snapshot(run_dir: Path) -> None:
    snapshot = run_dir / "source_snapshot"
    for relative in DECISION_SOURCE_FILES:
        captured = snapshot / relative
        current = ROOT / relative
        if not captured.is_file() or not current.is_file():
            raise RuntimeError(f"candidate source snapshot is incomplete: {relative}")
        if captured.read_bytes() != current.read_bytes():
            raise RuntimeError(f"candidate decision source changed: {relative}")


def _candidate_evaluation(
    *,
    summary: dict[str, Any],
    system: str,
    seed: int,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    evaluation_inputs: np.ndarray,
    evaluation_targets: np.ndarray,
    training_data_digest: str,
    evaluation_data_digest: str,
    config_digest: str,
    protocol: dict[str, Any],
    device: torch.device,
    evaluation_run_id: str,
    prediction_dir: Path,
) -> dict[str, Any]:
    competence_run_id = make_run_id(
        stage="competence", system=system, family=CANDIDATE_FAMILY, seed=seed
    )
    matches = [
        row
        for row in summary.get("trajectories", [])
        if row.get("system") == system and row.get("seed") == seed
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"candidate trajectory is missing or duplicated: {competence_run_id}"
        )
    competence = matches[0]
    competence_dir = ROOT / "experiments/runs" / competence_run_id
    stored = json.loads((competence_dir / "result.json").read_text(encoding="utf-8"))
    if stored != competence:
        raise RuntimeError(
            f"candidate summary differs from immutable run: {competence_run_id}"
        )
    ledger_matches = [
        row
        for row in read_runs(GLOBAL_LEDGER)
        if row.get("run_id") == competence_run_id
    ]
    if len(ledger_matches) != 1:
        raise RuntimeError(
            f"candidate ledger entry is missing or duplicated: {competence_run_id}"
        )
    expected_contract = {
        "family": CANDIDATE_FAMILY,
        "system": system,
        "seed": seed,
        "status": "completed",
        "updates": SNAPSHOT_UPDATES[-1],
        "snapshot_updates": list(SNAPSHOT_UPDATES),
        "evaluation_updates": list(EVALUATION_UPDATES),
        "chunk_samples": protocol["optimization"]["training_chunk_samples"],
        "common_preroll_samples_after_alignment": protocol["optimization"][
            "common_preroll_samples_after_alignment"
        ],
        "scored_start": protocol["optimization"]["scored_start_samples"],
        "scored_samples_per_episode": protocol["optimization"][
            "scored_samples_per_episode"
        ],
        "config_sha256": config_digest,
        "training_data_sha256": training_data_digest,
        "post_training_fit_applied": False,
    }
    for key, expected in expected_contract.items():
        if competence.get(key) != expected:
            raise RuntimeError(
                f"candidate contract changed for {competence_run_id}: {key}"
            )
    manifest = json.loads(
        (competence_dir / "data_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("training_data_sha256") != training_data_digest:
        raise RuntimeError(f"candidate training sources changed: {competence_run_id}")
    ledger = ledger_matches[0]
    if (
        ledger.get("status") != "completed"
        or ledger.get("model") != CANDIDATE_FAMILY
        or ledger.get("config_sha256") != config_digest
        or ledger.get("data_sha256") != competence.get("data_sha256")
    ):
        raise RuntimeError(f"candidate ledger changed: {competence_run_id}")
    observed_training_digest = digest_array_bundle(
        {"train_inputs": train_inputs, "train_targets": train_targets}
    )
    if observed_training_digest != competence.get("training_data_sha256"):
        raise RuntimeError(
            f"candidate training arrays are not paired: {competence_run_id}"
        )
    _assert_candidate_snapshot(competence_dir)
    checkpoint = competence_dir / "checkpoints" / f"update_{SNAPSHOT_UPDATES[-1]}.pt"
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model = build_sota_v12_model(CANDIDATE_FAMILY).to(device)
    model.load_state_dict(state, strict=True)
    evaluation = evaluate_sota_v12_model(
        model,
        torch.from_numpy(evaluation_inputs).to(device),
        torch.from_numpy(evaluation_targets).to(device),
        common_preroll_samples_after_alignment=protocol["optimization"][
            "common_preroll_samples_after_alignment"
        ],
        include_spectral=True,
    )
    prediction_paths = _save_predictions(
        model,
        evaluation_inputs,
        evaluation_targets,
        device=device,
        output_dir=prediction_dir,
    )
    del model
    torch.cuda.empty_cache()
    return {
        "family": CANDIDATE_FAMILY,
        "system": system,
        "seed": seed,
        "status": "completed",
        "competence_run_id": competence_run_id,
        "evaluation_run_id": evaluation_run_id,
        "checkpoint_update": SNAPSHOT_UPDATES[-1],
        "training_data_sha256": training_data_digest,
        "slow_value_evaluation_data_sha256": evaluation_data_digest,
        "slow_value_evaluation_source_seed": protocol["slow_value"][
            "evaluation_source_seed"
        ],
        "slow_value_evaluation": evaluation,
        "prediction_paths": prediction_paths,
        "post_training_fit_applied": False,
    }


def _assert_prediction_pairing(
    candidate_rows: list[dict[str, Any]], control_rows: list[dict[str, Any]]
) -> None:
    def index(rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[str]]:
        result: dict[tuple[str, int], list[str]] = {}
        for row in rows:
            seed = row.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise RuntimeError("slow-value prediction seed is invalid")
            key = (str(row.get("system")), seed)
            paths = row.get("prediction_paths")
            if key in result or not isinstance(paths, list) or len(paths) != 4:
                raise RuntimeError("slow-value prediction evidence is incomplete")
            result[key] = paths
        return result

    candidate = index(candidate_rows)
    control = index(control_rows)
    if set(candidate) != set(control):
        raise RuntimeError("candidate/control prediction matrices differ")
    for key in sorted(candidate):
        for candidate_path, control_path in zip(
            candidate[key], control[key], strict=True
        ):
            with np.load(ROOT / candidate_path, allow_pickle=False) as candidate_npz:
                candidate_input = candidate_npz["input"]
                candidate_target = candidate_npz["target"]
            with np.load(ROOT / control_path, allow_pickle=False) as control_npz:
                if not np.array_equal(candidate_input, control_npz["input"]):
                    raise RuntimeError(f"slow-value inputs are not paired: {key}")
                if not np.array_equal(candidate_target, control_npz["target"]):
                    raise RuntimeError(f"slow-value targets are not paired: {key}")


def _write_terminal(
    *,
    verdict: str,
    status: str,
    candidate_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    run_ids: list[str],
    generated_samples: int,
    failure_reason: str,
) -> None:
    summary = {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "stage": "slow_value",
        "status": status,
        "verdict": verdict,
        "failure_reason": failure_reason,
        "run_ids": run_ids,
        "candidate_trajectories": candidate_rows,
        "control_trajectories": control_rows,
        "gate": None,
        "selection_eligible_synthetic_samples_generated": generated_samples,
        "physical_audio_samples_read": 0,
        "confirmation_opened": False,
    }
    write_json_once_or_equal(SUMMARY, summary)
    maturity_path = CAMPAIGN_DIR / "MATURITY.json"
    maturity = json.loads(maturity_path.read_text(encoding="utf-8"))
    maturity.update(
        {
            "scientific_runs_launched": _v12_run_count(),
            "selection_eligible_synthetic_samples_generated": (
                _selection_sample_total()
            ),
            "current_stage": f"slow_value_{status}",
            "status": ("terminal_invalid" if status == "invalid" else "terminal_no_go"),
            "verdict": verdict,
        }
    )
    replace_json(maturity_path, maturity)
    write_json_once_or_equal(
        CAMPAIGN_DIR / "VERDICT.json",
        {
            "campaign_version": CAMPAIGN_VERSION,
            "verdict": verdict,
            "terminal_stage": "slow_value",
            "valid_scientific_gate_result": status != "invalid",
            "evidence_path": str(SUMMARY.relative_to(ROOT)),
            "confirmation_opened": False,
            "physical_audio_samples_read": 0,
        },
    )
    append_gate_event(
        GATE_LEDGER,
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "slow_value",
            "status": status,
            "evidence_path": str(SUMMARY.relative_to(ROOT)),
        },
    )


def _mark_failed_run(
    *,
    run_dir: Path,
    payload: dict[str, Any],
    failure_class: str,
    started_at: str,
    ledger_entry: dict[str, Any],
) -> None:
    write_json_once_or_equal(run_dir / "result.json", payload)
    status = {
        "status": "failed",
        "failure_class": failure_class,
        "started_at": started_at,
        "finished_at": _now(),
        "failure_reason": payload["failure_reason"],
    }
    status_path = run_dir / "status.json"
    if status_path.exists():
        replace_json(status_path, status)
    else:
        write_new_json(status_path, status)
    append_run_once_or_equal(GLOBAL_LEDGER, ledger_entry)


def _ensure_capacity(protocol: dict[str, Any], maximum_hours: float) -> None:
    resources = protocol["resource_budget"]
    if _used_gpu_hours() + maximum_hours > float(resources["v1_2_gpu_hours_maximum"]):
        raise RuntimeError("v1.2 GPU budget cannot cover the next run")
    if _run_disk_gib() + float(resources["run_disk_gib_maximum"]) > float(
        resources["v1_2_disk_gib_maximum"]
    ):
        raise RuntimeError("v1.2 disk budget is exhausted")


def main() -> int:
    validate_clean_worktree(ROOT)
    decisions = gate_decisions(GATE_LEDGER)
    validate_stage_authorization("slow_value", decisions)
    if "slow_value" in decisions or SUMMARY.exists():
        raise RuntimeError("v1.2 slow-value gate is already frozen")
    protocol = validate_repository_state(ROOT)
    candidate_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    launched_ids: list[str] = []
    generated_samples = 0

    def close_invalid(error: Exception) -> None:
        _write_terminal(
            verdict="INVALID",
            status="invalid",
            candidate_rows=candidate_rows,
            control_rows=control_rows,
            run_ids=launched_ids,
            generated_samples=generated_samples,
            failure_reason=f"{type(error).__name__}: {error}",
        )

    try:
        candidate_summaries = _candidate_summaries()
        if not torch.cuda.is_available():
            raise RuntimeError("the frozen CUDA device is unavailable")
        device_index = torch.cuda.current_device()
        device_properties = torch.cuda.get_device_properties(device_index)
        validate_cuda_device_properties(
            protocol,
            name=device_properties.name,
            total_memory_bytes=device_properties.total_memory,
        )
        resources = protocol["resource_budget"]
        evaluation_maximum_hours = float(
            resources["candidate_reevaluation_gpu_hours_per_run_maximum"]
        )
        paired_maximum_hours = float(
            resources["slow_value_paired_candidate_eval_plus_control_gpu_hours_maximum"]
        )
        maximum_needed = len(SYSTEMS) * len(SEEDS) * paired_maximum_hours
        if (
            float(resources["v1_2_gpu_hours_maximum"]) - _used_gpu_hours()
            < maximum_needed
        ):
            raise RuntimeError(
                "v1.2 GPU budget cannot cover the paired slow-value matrix"
            )
        _ensure_capacity(protocol, 0.0)
    except Exception as error:
        close_invalid(error)
        raise

    candidate_evaluation_run_ids = [
        make_run_id(
            stage="slow_value_eval",
            system=system,
            family=CANDIDATE_FAMILY,
            seed=seed,
        )
        for system in SYSTEMS
        for seed in SEEDS
    ]
    control_run_ids = [
        make_run_id(
            stage="slow_value",
            system=system,
            family=CONTROL_FAMILY,
            seed=seed,
        )
        for system in SYSTEMS
        for seed in SEEDS
    ]
    planned_ids = [*candidate_evaluation_run_ids, *control_run_ids]
    planned_dirs = [ROOT / "experiments/runs" / run_id for run_id in planned_ids]
    prior_ids = {row["run_id"] for row in read_runs(GLOBAL_LEDGER)}
    if prior_ids.intersection(planned_ids) or any(
        path.exists() for path in planned_dirs
    ):
        error = RuntimeError("v1.2 slow-value run reuse or resume is forbidden")
        close_invalid(error)
        raise error

    data = protocol["synthetic_data"]
    optimization = protocol["optimization"]
    config_digest = digest_text(strict_json(protocol))
    device = torch.device("cuda", device_index)
    candidate_index = 0
    control_index = 0
    evaluation_elapsed_hours: dict[tuple[str, int], float] = {}

    for system in SYSTEMS:
        try:
            train_inputs, train_targets = build_sota_v12_system_episodes(
                system=system,
                source_seed=int(data["train_source_seed"]),
                episodes=int(data["train_episodes"]),
                samples=int(data["episode_samples"]),
            )
            generated_samples += 2 * train_inputs.size
            evaluation_inputs, evaluation_targets = build_sota_v12_system_episodes(
                system=system,
                source_seed=int(data["slow_value_eval_source_seed"]),
                episodes=int(data["slow_value_eval_episodes"]),
                samples=int(data["episode_samples"]),
            )
            generated_samples += 2 * evaluation_inputs.size
        except Exception as error:
            close_invalid(error)
            raise

        training_data_digest = digest_array_bundle(
            {"train_inputs": train_inputs, "train_targets": train_targets}
        )
        evaluation_data_digest = digest_array_bundle(
            {
                "slow_value_eval_inputs": evaluation_inputs,
                "slow_value_eval_targets": evaluation_targets,
            }
        )
        data_digest = digest_array_bundle(
            {
                "train_inputs": train_inputs,
                "train_targets": train_targets,
                "slow_value_eval_inputs": evaluation_inputs,
                "slow_value_eval_targets": evaluation_targets,
            }
        )
        data_manifest = {
            "generator": data["generator"],
            "target_renderer": data["target_renderer"],
            "system": system,
            "train_source_seed": data["train_source_seed"],
            "slow_value_eval_source_seed": data["slow_value_eval_source_seed"],
            "train_episodes": data["train_episodes"],
            "slow_value_eval_episodes": data["slow_value_eval_episodes"],
            "episode_samples": data["episode_samples"],
            "normalization": data["normalization"],
            "evaluation_role": "SLOW_VALUE_EVAL",
            "competence_internal_dev_reused": False,
            "training_data_sha256": training_data_digest,
            "slow_value_evaluation_data_sha256": evaluation_data_digest,
            "data_sha256": data_digest,
        }

        candidate_stability_failed = False
        for seed in SEEDS:
            run_id = candidate_evaluation_run_ids[candidate_index]
            candidate_index += 1
            run_dir = ROOT / "experiments/runs" / run_id
            try:
                _ensure_capacity(protocol, evaluation_maximum_hours)
                run_dir.mkdir(parents=True, exist_ok=False)
            except Exception as error:
                close_invalid(error)
                raise
            launched_ids.append(run_id)
            started_at = _now()
            started_clock = time.perf_counter()
            provenance: dict[str, Any] | None = None
            try:
                provenance = capture_provenance(
                    ROOT,
                    run_dir,
                    ["uv", "run", "python", "scripts/run_sota_v12_slow_value.py"],
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
                payload = _candidate_evaluation(
                    summary=candidate_summaries[system],
                    system=system,
                    seed=seed,
                    train_inputs=train_inputs,
                    train_targets=train_targets,
                    evaluation_inputs=evaluation_inputs,
                    evaluation_targets=evaluation_targets,
                    training_data_digest=training_data_digest,
                    evaluation_data_digest=evaluation_data_digest,
                    config_digest=config_digest,
                    protocol=protocol,
                    device=device,
                    evaluation_run_id=run_id,
                    prediction_dir=run_dir / "predictions",
                )
                if _directory_gib(run_dir) > float(resources["run_disk_gib_maximum"]):
                    raise RuntimeError(
                        "candidate reevaluation exceeded its disk allowance"
                    )
                elapsed = time.perf_counter() - started_clock
                if elapsed > evaluation_maximum_hours * 3600.0:
                    raise RuntimeError(
                        "candidate reevaluation exceeded its wall-clock limit"
                    )
                payload |= {
                    "config_sha256": config_digest,
                    "data_sha256": data_digest,
                    "elapsed_seconds": elapsed,
                }
                evaluation_elapsed_hours[(system, seed)] = elapsed / 3600.0
                write_json_once_or_equal(run_dir / "result.json", payload)
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
                        phase=PHASE_CANDIDATE_EVAL,
                        family=CANDIDATE_FAMILY,
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
                candidate_rows.append(payload)
            except SotaV12StabilityError as error:
                candidate_stability_failed = True
                elapsed = time.perf_counter() - started_clock
                reason = f"{type(error).__name__}: {error}"
                failure = {
                    "family": CANDIDATE_FAMILY,
                    "system": system,
                    "seed": seed,
                    "status": "stability_failed",
                    "failure_reason": reason,
                    "evaluation_run_id": run_id,
                    "training_data_sha256": training_data_digest,
                    "slow_value_evaluation_data_sha256": evaluation_data_digest,
                    "slow_value_evaluation_source_seed": data[
                        "slow_value_eval_source_seed"
                    ],
                    "post_training_fit_applied": False,
                }
                _mark_failed_run(
                    run_dir=run_dir,
                    payload=failure,
                    failure_class="scientific_candidate_stability",
                    started_at=started_at,
                    ledger_entry=_ledger_entry(
                        run_id=run_id,
                        phase=PHASE_CANDIDATE_EVAL,
                        family=CANDIDATE_FAMILY,
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
                candidate_rows.append(failure)
                torch.cuda.empty_cache()
            except Exception as error:
                elapsed = time.perf_counter() - started_clock
                reason = f"{type(error).__name__}: {error}"
                failure = {
                    "family": CANDIDATE_FAMILY,
                    "system": system,
                    "seed": seed,
                    "status": "operational_failure",
                    "failure_reason": reason,
                    "evaluation_run_id": run_id,
                }
                _mark_failed_run(
                    run_dir=run_dir,
                    payload=failure,
                    failure_class="operational_or_instrumentation",
                    started_at=started_at,
                    ledger_entry=_ledger_entry(
                        run_id=run_id,
                        phase=PHASE_CANDIDATE_EVAL,
                        family=CANDIDATE_FAMILY,
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
                close_invalid(error)
                raise
        if candidate_stability_failed:
            _write_terminal(
                verdict=protocol["slow_value"][
                    "nonfinite_candidate_reevaluation_verdict"
                ],
                status="no-go",
                candidate_rows=candidate_rows,
                control_rows=control_rows,
                run_ids=launched_ids,
                generated_samples=generated_samples,
                failure_reason=f"candidate stability failed for {system}",
            )
            return 1

        control_stability_failed = False
        for seed in SEEDS:
            run_id = control_run_ids[control_index]
            control_index += 1
            run_dir = ROOT / "experiments/runs" / run_id
            control_maximum_hours = (
                paired_maximum_hours - evaluation_elapsed_hours[(system, seed)]
            )
            try:
                if control_maximum_hours <= 0.0:
                    raise RuntimeError(
                        "candidate reevaluation exhausted the paired run budget"
                    )
                _ensure_capacity(protocol, control_maximum_hours)
                run_dir.mkdir(parents=True, exist_ok=False)
            except Exception as error:
                close_invalid(error)
                raise
            launched_ids.append(run_id)
            started_at = _now()
            started_clock = time.perf_counter()
            provenance = None
            try:
                provenance = capture_provenance(
                    ROOT,
                    run_dir,
                    ["uv", "run", "python", "scripts/run_sota_v12_slow_value.py"],
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
                    family=CONTROL_FAMILY,
                    system=system,
                    train_inputs=train_inputs,
                    train_targets=train_targets,
                    internal_dev_inputs=evaluation_inputs,
                    internal_dev_targets=evaluation_targets,
                    snapshot_updates=SNAPSHOT_UPDATES,
                    evaluation_updates=EVALUATION_UPDATES,
                    common_preroll_samples_after_alignment=optimization[
                        "common_preroll_samples_after_alignment"
                    ],
                    chunk_samples=optimization["training_chunk_samples"],
                    learning_rate=optimization["learning_rate"],
                    projection_gain_weight=optimization["projection_gain_weight"],
                    gradient_clip_norm=optimization["gradient_clip_norm"],
                    maximum_elapsed_seconds=(
                        control_maximum_hours * 3600.0
                        - resources["artifact_finalization_reserve_seconds"]
                    ),
                    device=device,
                    seed=seed,
                    checkpoint_callback=(
                        lambda update, _row, seed_value=seed, system_value=system: (
                            print(
                                f"v12_control_checkpoint:{system_value}:seed={seed_value}:"
                                f"update={update}",
                                flush=True,
                            )
                        )
                    ),
                )
                final_evaluation = next(
                    row["internal_dev"]
                    for row in result.checkpoints
                    if row["update"] == EVALUATION_UPDATES[-1]
                )
                payload = asdict(result) | {
                    "status": "completed",
                    "config_sha256": config_digest,
                    "data_sha256": data_digest,
                    "training_data_sha256": training_data_digest,
                    "slow_value_evaluation_data_sha256": evaluation_data_digest,
                    "slow_value_evaluation_source_seed": data[
                        "slow_value_eval_source_seed"
                    ],
                    "slow_value_evaluation": final_evaluation,
                    "post_training_fit_applied": False,
                }
                _save_states(run_dir, states)
                payload["prediction_paths"] = _save_predictions(
                    model,
                    evaluation_inputs,
                    evaluation_targets,
                    device=device,
                    output_dir=run_dir / "predictions",
                )
                if _directory_gib(run_dir) > float(resources["run_disk_gib_maximum"]):
                    raise RuntimeError("control run exceeded its disk allowance")
                elapsed = time.perf_counter() - started_clock
                payload["elapsed_seconds_outer"] = elapsed
                write_json_once_or_equal(run_dir / "result.json", payload)
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
                        phase=PHASE_CONTROL,
                        family=CONTROL_FAMILY,
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
                control_rows.append(payload)
                del model
                torch.cuda.empty_cache()
            except SotaV12StabilityError as error:
                control_stability_failed = True
                elapsed = time.perf_counter() - started_clock
                reason = f"{type(error).__name__}: {error}"
                failure = {
                    "family": CONTROL_FAMILY,
                    "system": system,
                    "seed": seed,
                    "status": "stability_failed",
                    "failure_reason": reason,
                    "training_data_sha256": training_data_digest,
                    "slow_value_evaluation_data_sha256": evaluation_data_digest,
                    "slow_value_evaluation_source_seed": data[
                        "slow_value_eval_source_seed"
                    ],
                    "post_training_fit_applied": False,
                }
                _mark_failed_run(
                    run_dir=run_dir,
                    payload=failure,
                    failure_class="scientific_control_stability",
                    started_at=started_at,
                    ledger_entry=_ledger_entry(
                        run_id=run_id,
                        phase=PHASE_CONTROL,
                        family=CONTROL_FAMILY,
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
                control_rows.append(failure)
                torch.cuda.empty_cache()
            except Exception as error:
                elapsed = time.perf_counter() - started_clock
                reason = f"{type(error).__name__}: {error}"
                failure = {
                    "family": CONTROL_FAMILY,
                    "system": system,
                    "seed": seed,
                    "status": "operational_failure",
                    "failure_reason": reason,
                }
                _mark_failed_run(
                    run_dir=run_dir,
                    payload=failure,
                    failure_class="operational_or_instrumentation",
                    started_at=started_at,
                    ledger_entry=_ledger_entry(
                        run_id=run_id,
                        phase=PHASE_CONTROL,
                        family=CONTROL_FAMILY,
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
                close_invalid(error)
                raise
        if control_stability_failed:
            _write_terminal(
                verdict=protocol["slow_value"]["nonfinite_control_verdict"],
                status="no-go",
                candidate_rows=candidate_rows,
                control_rows=control_rows,
                run_ids=launched_ids,
                generated_samples=generated_samples,
                failure_reason=f"control stability failed for {system}",
            )
            return 1

    try:
        _assert_prediction_pairing(candidate_rows, control_rows)
        gate = evaluate_slow_value_gate(candidate_rows, control_rows, protocol)
    except Exception as error:
        close_invalid(error)
        raise

    status = "passed" if gate["passed"] else "no-go"
    summary = {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "stage": "slow_value",
        "status": status,
        "verdict": gate["verdict"],
        "run_ids": planned_ids,
        "candidate_evaluation_run_ids": candidate_evaluation_run_ids,
        "control_run_ids": control_run_ids,
        "candidate_trajectories": candidate_rows,
        "control_trajectories": control_rows,
        "gate": gate,
        "selection_eligible_synthetic_samples_generated": generated_samples,
        "physical_audio_samples_read": 0,
        "confirmation_opened": False,
        "gpu_hours_used_after_stage": _used_gpu_hours(),
        "run_disk_gib_after_stage": _run_disk_gib(),
    }
    write_json_once_or_equal(SUMMARY, summary)
    maturity_path = CAMPAIGN_DIR / "MATURITY.json"
    maturity = json.loads(maturity_path.read_text(encoding="utf-8"))
    maturity.update(
        {
            "scientific_runs_launched": _v12_run_count(),
            "selection_eligible_synthetic_samples_generated": (
                _selection_sample_total()
            ),
        }
    )
    if gate["passed"]:
        maturity.update(
            {
                "current_stage": "physical_development_authorized",
                "synthetic_verdict": gate["verdict"],
            }
        )
        write_json_once_or_equal(
            CAMPAIGN_DIR / "SYNTHETIC_VERDICT.json",
            {
                "campaign_version": CAMPAIGN_VERSION,
                "verdict": gate["verdict"],
                "evidence_path": str(SUMMARY.relative_to(ROOT)),
                "physical_or_commercial_superiority": False,
            },
        )
    else:
        maturity.update(
            {
                "current_stage": "slow_value_no_go",
                "status": "terminal_no_go",
                "verdict": gate["verdict"],
            }
        )
        write_json_once_or_equal(
            CAMPAIGN_DIR / "VERDICT.json",
            {
                "campaign_version": CAMPAIGN_VERSION,
                "verdict": gate["verdict"],
                "terminal_stage": "slow_value",
                "valid_scientific_gate_result": True,
                "evidence_path": str(SUMMARY.relative_to(ROOT)),
                "confirmation_opened": False,
                "physical_audio_samples_read": 0,
            },
        )
    replace_json(maturity_path, maturity)
    append_gate_event(
        GATE_LEDGER,
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "slow_value",
            "status": status,
            "evidence_path": str(SUMMARY.relative_to(ROOT)),
        },
    )
    print(json.dumps(gate, allow_nan=False, indent=2, sort_keys=True))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"SOTA v1.2 slow-value failed closed: {error}", file=sys.stderr)
        raise
