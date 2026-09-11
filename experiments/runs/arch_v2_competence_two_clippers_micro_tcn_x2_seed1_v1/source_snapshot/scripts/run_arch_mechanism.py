#!/usr/bin/env python3
"""Run the frozen synthetic mechanism qualification for architecture ideas."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from fssr_nam.campaign.amp_arch_gates import evaluate_arch_mechanism_gate
from fssr_nam.campaign.amp_arch_registry import (
    append_gate_event,
    gate_decisions,
)
from fssr_nam.campaign.amp_quality_arch_v1 import (
    CAMPAIGN_PATH,
    CAMPAIGN_VERSION,
    TRAINING_CANDIDATES,
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
from fssr_nam.data.arch_fixtures import build_mechanism_episodes
from fssr_nam.reporting.ledger import append_run, read_runs
from fssr_nam.training.arch_v1 import train_mechanism_model

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / CAMPAIGN_PATH
GATE_LEDGER = CAMPAIGN_DIR / "GATE_LEDGER.jsonl"
GLOBAL_LEDGER = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
SUMMARY = ROOT / "experiments/summaries/amp_quality_arch_v1/mechanism.json"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _run_id(family: str) -> str:
    return make_run_id("mechanism", "synthetic", family, "none", 0)


def _trajectory_directory(run_dir: Path, system: str, weight: float) -> Path:
    weight_text = f"{weight:.2f}".replace(".", "p")
    return run_dir / "trajectories" / f"{system}_aux_{weight_text}"


def _train(
    *,
    family: str,
    system: str,
    auxiliary_weight: float,
    episodes: dict[str, tuple[torch.Tensor, torch.Tensor]],
    validation: dict[str, tuple[torch.Tensor, torch.Tensor]],
    protocol: dict[str, Any],
    run_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    config = protocol["mechanism_training"]
    print(
        f"arch_mechanism_started:{family}:{system}:aux={auxiliary_weight:.2f}",
        flush=True,
    )
    model, result = train_mechanism_model(
        family=family,
        system=system,
        train_inputs=episodes[system][0],
        train_targets=episodes[system][1],
        validation_inputs=validation[system][0],
        validation_targets=validation[system][1],
        profile=str(config["profile"]),
        updates=int(config["updates_per_trajectory"]),
        tail_samples=int(config["loss_tail_samples"]),
        learning_rate=float(config["learning_rate"]),
        auxiliary_weight=auxiliary_weight,
        device=device,
        seed=int(config["seed"]),
    )
    payload = asdict(result)
    destination = _trajectory_directory(run_dir, system, auxiliary_weight)
    destination.mkdir(parents=True, exist_ok=False)
    write_new_json(destination / "result.json", payload)
    checkpoint = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    torch.save(checkpoint, destination / "model_state.pt")
    print(
        f"arch_mechanism_completed:{family}:{system}:aux={auxiliary_weight:.2f}:"
        f"esr={payload['validation']['esr']:.9g}",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()
    return payload


def _family_trajectory_specs(family: str) -> tuple[tuple[str, float], ...]:
    if family in {"micro_tcn_x2", "phys_det_tcn_x2"}:
        return (("static_composite", 0.0), ("dynamic_composite", 0.0))
    if family == "phys_s6_tcn_x2":
        return tuple(("dynamic_composite", weight) for weight in (0.01, 0.05, 0.10))
    return (("two_clippers", 0.0),)


def _update_maturity(gate: dict[str, Any], run_count: int) -> None:
    path = CAMPAIGN_DIR / "MATURITY.json"
    maturity = json.loads(path.read_text(encoding="utf-8"))
    maturity["scientific_runs_launched"] = run_count
    maturity["current_stage"] = "mechanism_passed" if gate["passed"] else "terminal"
    maturity["status"] = "active" if gate["passed"] else "terminal_no_go"
    maturity["mechanism_eligible_candidates"] = gate["eligible_candidates"]
    maturity["mechanism_rejected_candidates"] = gate["rejected_candidates"]
    if not gate["passed"]:
        maturity["verdict"] = "NO-GO-ARCH"
    replace_json(path, maturity)


def main() -> int:
    decisions = gate_decisions(GATE_LEDGER)
    validate_stage_authorization("mechanism", decisions)
    if "mechanism" in decisions:
        raise RuntimeError("architecture mechanism gate is already recorded")
    protocol = validate_repository_state(ROOT, require_frozen=True)
    run_ids = [_run_id(family) for family in TRAINING_CANDIDATES]
    prior_ids = {entry["run_id"] for entry in read_runs(GLOBAL_LEDGER)}
    run_dirs = [ROOT / "experiments/runs" / run_id for run_id in run_ids]
    if SUMMARY.exists() or any(path.exists() for path in run_dirs):
        raise RuntimeError("unregistered or immutable mechanism evidence exists")
    if prior_ids.intersection(run_ids):
        raise RuntimeError("an architecture mechanism run ID is already ledgered")
    maturity = json.loads((CAMPAIGN_DIR / "MATURITY.json").read_text(encoding="utf-8"))
    if maturity.get("scientific_runs_launched") != 0:
        raise RuntimeError("mechanism requires zero prior architecture scientific runs")
    config = protocol["mechanism_training"]
    train_arrays = build_mechanism_episodes(
        split="train",
        episodes=int(config["train_episodes"]),
        samples=int(config["episode_samples"]),
    )
    validation_arrays = build_mechanism_episodes(
        split="validation",
        episodes=int(config["validation_episodes"]),
        samples=int(config["episode_samples"]),
    )
    train = {
        system: (torch.from_numpy(inputs), torch.from_numpy(targets))
        for system, (inputs, targets) in train_arrays.items()
    }
    validation = {
        system: (torch.from_numpy(inputs), torch.from_numpy(targets))
        for system, (inputs, targets) in validation_arrays.items()
    }
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("frozen CUDA mechanism device is unavailable")
    trajectories: list[dict[str, Any]] = []
    family_evidence: dict[str, Any] = {}
    config_digest = digest_text(strict_json(protocol))
    data_digest = digest_text(
        strict_json(
            {
                "input_source": "deterministic_rights_free_synthetic",
                "train_seed": config["source_seed_train"],
                "validation_seed": config["source_seed_validation"],
                "episode_samples": config["episode_samples"],
                "train_episodes": config["train_episodes"],
                "validation_episodes": config["validation_episodes"],
            }
        )
    )
    for family, run_id, run_dir in zip(
        TRAINING_CANDIDATES, run_ids, run_dirs, strict=True
    ):
        started_at = _now()
        run_dir.mkdir(parents=True, exist_ok=False)
        provenance = capture_provenance(
            ROOT,
            run_dir,
            ["uv", "run", "python", "scripts/run_arch_mechanism.py"],
        )
        write_new_json(
            run_dir / "status.json",
            {"status": "running", "started_at": started_at, "finished_at": None},
        )
        rows = [
            _train(
                family=family,
                system=system,
                auxiliary_weight=weight,
                episodes=train,
                validation=validation,
                protocol=protocol,
                run_dir=run_dir,
                device=device,
            )
            for system, weight in _family_trajectory_specs(family)
        ]
        if family == "phys_s6_tcn_x2":
            selected_weight = min(
                (row["validation"]["esr"], row["auxiliary_weight"]) for row in rows
            )[1]
            rows.append(
                _train(
                    family=family,
                    system="static_composite",
                    auxiliary_weight=float(selected_weight),
                    episodes=train,
                    validation=validation,
                    protocol=protocol,
                    run_dir=run_dir,
                    device=device,
                )
            )
        trajectories.extend(rows)
        finished_at = _now()
        family_payload = {
            "run_id": run_id,
            "family": family,
            "status": "completed",
            "started_at": started_at,
            "finished_at": finished_at,
            "trajectories": rows,
            "provenance": provenance,
            "physical_audio_samples_read": 0,
        }
        write_new_json(run_dir / "mechanism.json", family_payload)
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
            {
                "date": finished_at,
                "run_id": run_id,
                "phase": "AMP-QUALITY-ARCH-MECHANISM",
                "model": family,
                "device": "synthetic",
                "seed": int(config["seed"]),
                "commit": provenance["git_head"],
                "config_sha256": config_digest,
                "data_sha256": data_digest,
                "status": "completed",
                "failure_reason": "",
                "results_path": str(run_dir.relative_to(ROOT)),
            },
        )
        family_evidence[family] = family_payload
    gate = evaluate_arch_mechanism_gate(trajectories, protocol)
    summary = {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "stage": "mechanism",
        "status": "passed" if gate["passed"] else "failed",
        "run_ids": run_ids,
        "trajectory_count": len(trajectories),
        "family_evidence": family_evidence,
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
            "stage": "mechanism",
            "status": "passed" if gate["passed"] else "failed",
            "evidence_path": str(SUMMARY.relative_to(ROOT)),
        },
    )
    _update_maturity(gate, len(run_ids))
    if not gate["passed"]:
        write_new_json(
            CAMPAIGN_DIR / "VERDICT.json",
            {
                "campaign_version": CAMPAIGN_VERSION,
                "verdict": "NO-GO-ARCH",
                "terminal_stage": "mechanism",
                "valid_scientific_gate_result": True,
                "evidence_path": str(SUMMARY.relative_to(ROOT)),
            },
        )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"architecture mechanism failed closed: {error}", file=sys.stderr)
        raise
