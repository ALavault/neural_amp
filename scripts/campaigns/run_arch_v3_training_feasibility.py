#!/usr/bin/env python3
"""Measure the frozen v3 training workload without reading scientific data."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from fssr_nam.campaign.amp_arch_v3_registry import gate_decisions
from fssr_nam.campaign.amp_quality_arch_v3 import (
    CAMPAIGN_PATH,
    CAMPAIGN_VERSION,
    INITIAL_FAMILIES,
    validate_repository_state,
)
from fssr_nam.campaign.quality_aa_provenance import write_new_json
from fssr_nam.losses import esr_loss
from fssr_nam.models import build_arch_v3_candidate
from fssr_nam.training.arch_v3 import projection_gain_loss

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / CAMPAIGN_PATH
EVIDENCE = CAMPAIGN_DIR / "TRAINING_FEASIBILITY_V2.json"
PREFLIGHT = ROOT / "experiments/summaries/amp_quality_arch_v3/preflight.json"
GATE_LEDGER = CAMPAIGN_DIR / "GATE_LEDGER.jsonl"
WARMUP_ITERATIONS = 3
MEASURED_ITERATIONS = 12


def _synchronize() -> None:
    torch.cuda.synchronize()


def _elapsed(operation: Any) -> float:
    _synchronize()
    started = time.perf_counter()
    operation()
    _synchronize()
    return time.perf_counter() - started


def _training_iteration(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    source: torch.Tensor,
    target: torch.Tensor,
    projection_gain_weight: float,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    output_holder: list[torch.Tensor] = []
    loss_holder: list[torch.Tensor] = []

    def operation() -> None:
        output = model.stream(source)
        model.detach_stream_state()
        loss = esr_loss(output, target) + projection_gain_weight * projection_gain_loss(
            output, target
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        output_holder.append(output)
        loss_holder.append(loss)

    elapsed = _elapsed(operation)
    output = output_holder[0]
    loss = loss_holder[0]
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )
    return {
        "elapsed_seconds": elapsed,
        "finite_output": bool(torch.isfinite(output).all().item()),
        "finite_loss": bool(torch.isfinite(loss).item()),
        "finite_parameter_gradients": finite_gradients,
    }


def _forward_seconds(model: torch.nn.Module, signal: torch.Tensor) -> float:
    def operation() -> None:
        model.reset_state()
        with torch.inference_mode():
            output = model(signal)
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError("non-finite v3 feasibility forward output")

    return _elapsed(operation)


def _projection(
    *,
    median_update_seconds: float,
    preroll_seconds: float,
    evaluation_episode_seconds: float,
    trajectories: int,
    updates: int,
    checkpoints: int,
    chunk_samples: int,
    scored_samples: int,
    evaluation_episodes_per_checkpoint: int,
) -> dict[str, float | int]:
    training_seconds = trajectories * updates * median_update_seconds
    preroll_events = trajectories * (1 + (updates * chunk_samples) // scored_samples)
    total_preroll_seconds = preroll_events * preroll_seconds
    evaluation_events = trajectories * checkpoints * evaluation_episodes_per_checkpoint
    evaluation_seconds = evaluation_events * evaluation_episode_seconds
    total_seconds = training_seconds + total_preroll_seconds + evaluation_seconds
    return {
        "trajectories": trajectories,
        "updates_per_trajectory": updates,
        "checkpoints_per_trajectory": checkpoints,
        "preroll_events": preroll_events,
        "evaluation_episode_events": evaluation_events,
        "training_gpu_hours": training_seconds / 3600.0,
        "preroll_gpu_hours": total_preroll_seconds / 3600.0,
        "evaluation_gpu_hours": evaluation_seconds / 3600.0,
        "total_gpu_hours": total_seconds / 3600.0,
    }


def main() -> int:
    if EVIDENCE.exists():
        raise RuntimeError(
            f"v3 training feasibility evidence already exists: {EVIDENCE}"
        )
    protocol = validate_repository_state(ROOT, require_frozen=True)
    if gate_decisions(GATE_LEDGER).get("preflight") != "passed":
        raise RuntimeError("v3 representability preflight has not passed")
    if not torch.cuda.is_available():
        raise RuntimeError("the CUDA device required by the v3 budget is unavailable")
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    initial_scale = float(preflight["initial_residual_scale"])
    architecture = protocol["architectures"]
    round_one = protocol["round_1"]
    competence = protocol["competence"]
    data = protocol["synthetic_data"]
    device = torch.device("cuda")
    torch.manual_seed(20_260_829)
    torch.cuda.manual_seed_all(20_260_829)
    chunk_samples = int(round_one["training_chunk_samples"])
    preroll_samples = int(data["preroll_samples"])
    episode_samples = int(data["episode_samples"])
    scored_samples = int(data["scored_samples"])
    results: dict[str, dict[str, Any]] = {}

    for family in INITIAL_FAMILIES:
        print(f"v3_feasibility_started:{family}", flush=True)
        torch.cuda.empty_cache()
        model = build_arch_v3_candidate(
            family,
            profile=str(architecture["profile"]),
            initial_residual_scale=initial_scale,
        ).to(device)
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(round_one["learning_rate"]),
            weight_decay=float(round_one["weight_decay"]),
        )
        source = torch.randn(1, chunk_samples, device=device) * 0.25
        target = torch.tanh(2.2 * source)
        preroll = torch.randn(1, preroll_samples, device=device) * 0.25
        evaluation = torch.randn(1, episode_samples, device=device) * 0.25
        model.reset_state()
        with torch.no_grad():
            model.stream(preroll)
        for _ in range(WARMUP_ITERATIONS):
            measurement = _training_iteration(
                model,
                optimizer,
                source,
                target,
                float(round_one["projection_gain_weight"]),
            )
            if not all(
                value for key, value in measurement.items() if key != "elapsed_seconds"
            ):
                raise RuntimeError(f"non-finite v3 feasibility warmup: {family}")
        torch.cuda.reset_peak_memory_stats()
        measurements = [
            _training_iteration(
                model,
                optimizer,
                source,
                target,
                float(round_one["projection_gain_weight"]),
            )
            for _ in range(MEASURED_ITERATIONS)
        ]
        update_seconds = statistics.median(
            float(row["elapsed_seconds"]) for row in measurements
        )
        preroll_seconds = _forward_seconds(model, preroll)
        evaluation_seconds = _forward_seconds(model, evaluation)
        finite = all(
            all(value for key, value in row.items() if key != "elapsed_seconds")
            for row in measurements
        )
        round_projection = _projection(
            median_update_seconds=update_seconds,
            preroll_seconds=preroll_seconds,
            evaluation_episode_seconds=evaluation_seconds,
            trajectories=len(round_one["systems"]) * len(round_one["seeds"]),
            updates=max(round_one["checkpoint_updates"]),
            checkpoints=len(round_one["checkpoint_updates"]),
            chunk_samples=chunk_samples,
            scored_samples=scored_samples,
            evaluation_episodes_per_checkpoint=int(data["train_episodes"])
            + int(data["internal_dev_episodes"]),
        )
        competence_projection = _projection(
            median_update_seconds=update_seconds,
            preroll_seconds=preroll_seconds,
            evaluation_episode_seconds=evaluation_seconds,
            trajectories=len(competence["seeds"]) * len(round_one["systems"]),
            updates=max(competence["checkpoint_updates"]),
            checkpoints=len(competence["checkpoint_updates"]),
            chunk_samples=chunk_samples,
            scored_samples=scored_samples,
            evaluation_episodes_per_checkpoint=int(data["train_episodes"])
            + int(data["internal_dev_episodes"]),
        )
        results[family] = {
            "finite": finite,
            "median_seconds_per_training_update": update_seconds,
            "seconds_per_preroll_episode": preroll_seconds,
            "seconds_per_evaluation_episode": evaluation_seconds,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "round_1_projection": round_projection,
            "competence_projection": competence_projection,
            "measurements": measurements,
        }
        print(
            f"v3_feasibility_completed:{family}:finite={finite}:"
            f"median={update_seconds:.6f}",
            flush=True,
        )
        del model, optimizer, source, target, preroll, evaluation
        torch.cuda.empty_cache()

    round_one_hours = sum(
        float(row["round_1_projection"]["total_gpu_hours"]) for row in results.values()
    )
    worst_competence_hours = max(
        float(row["competence_projection"]["total_gpu_hours"])
        for row in results.values()
    )
    committed_projection = round_one_hours + worst_competence_hours
    budget = float(protocol["resource_budget"]["campaign_gpu_hours_maximum"])
    checks = {
        "all_families_finite": all(bool(row["finite"]) for row in results.values()),
        "round_1_plus_worst_competence_within_campaign_budget": committed_projection
        <= budget,
    }
    report = {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "stage": "training_feasibility_pre_round_1",
        "implementation_revision": "exact_chunk_samples_v2",
        "supersedes_evidence": str(
            (CAMPAIGN_DIR / "TRAINING_FEASIBILITY.json").relative_to(ROOT)
        ),
        "status": "passed" if all(checks.values()) else "failed",
        "scientific_run": False,
        "input_source": "deterministic_random_tensor_only",
        "train_fixture_samples_read": 0,
        "validation_source_accessed": False,
        "physical_audio_samples_read": 0,
        "sealed_outputs_accessed": {"blackstar": False, "ua1176": False},
        "cuda_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "warmup_iterations": WARMUP_ITERATIONS,
        "measured_iterations": MEASURED_ITERATIONS,
        "mean_full_chunks_per_scored_episode": scored_samples / chunk_samples,
        "every_update_uses_exact_chunk_samples": True,
        "checks": checks,
        "round_1_projected_gpu_hours": round_one_hours,
        "worst_case_competence_projected_gpu_hours": worst_competence_hours,
        "committed_projected_gpu_hours": committed_projection,
        "campaign_gpu_hours_maximum": budget,
        "uncommitted_gpu_hours_for_optional_rounds": budget - committed_projection,
        "results": results,
    }
    write_new_json(EVIDENCE, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
