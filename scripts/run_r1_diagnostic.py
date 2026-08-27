#!/usr/bin/env python3
"""Run one immutable factorial, horizon, or cascade R1 trajectory."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from fssr_nam.campaign.r1 import (
    R1Executor,
    RunSpec,
    load_gate_decisions,
    parse_run_id,
    validate_repository_configs,
)
from fssr_nam.reporting.r1_preflight import validate_lock_digest
from fssr_nam.training.r1_diagnostic import (
    M4_CONFIG_PATH,
    M4_MANIFEST_PATH,
    M4_SPLIT_PATH,
    combined_sha256,
    execution_profile,
    load_m4_dataset,
    load_promoted_residual,
    load_synthetic_cascade_dataset,
    make_model,
    run_training,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATES = ROOT / ".codex_campaign/r1/GATES.json"
MATURITY_PATH = ROOT / ".codex_campaign/r1/MATURITY.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True, choices=("factorial", "horizon", "cascade")
    )
    parser.add_argument(
        "--device", required=True, choices=("fulltone", "bigmuff", "synthetic")
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=("a2", "s3", "rf31", "rf2047", "mono", "cascade"),
    )
    parser.add_argument("--loss", choices=("m4", "wright"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id")
    parser.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    parser.add_argument("--training-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="two updates through the real path; no run directory or ledger write",
    )
    parser.add_argument("--plan", action="store_true")
    return parser


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_worktree() -> None:
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            ".",
            ":(exclude)experiments/runs/**",
            ":(exclude).codex_campaign/RUN_LEDGER.jsonl",
            ":(exclude).codex_campaign/RESULT_INDEX.csv",
            ":(exclude).codex_campaign/r1/RESULT_INDEX.csv",
            ":(exclude).codex_campaign/r1/STATE.md",
            ":(exclude).codex_campaign/r1/MATURITY.json",
            ":(exclude).codex_campaign/r1/CLAIMS.md",
            ":(exclude).codex_campaign/r1/FAILURES.md",
            ":(exclude).codex_campaign/r1/HANDOFF.md",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("counted R1 diagnostics require a clean committed worktree")


def _decisions(path: Path) -> dict[str, object]:
    decisions: dict[str, object] = {}
    if MATURITY_PATH.is_file():
        maturity = json.loads(MATURITY_PATH.read_text(encoding="utf-8"))
        gates = maturity.get("gates", {})
        if isinstance(gates, dict):
            decisions.update(gates)
    if path.is_file():
        decisions.update(load_gate_decisions(path))
    return decisions


def _passed(value: object) -> bool:
    if value is True or (isinstance(value, str) and value in {"passed", "promoted"}):
        return True
    return isinstance(value, dict) and (
        value.get("passed") is True or value.get("decision") in {"passed", "promoted"}
    )


def _selected_loss(
    stage: str, requested: str | None, decisions: dict[str, object]
) -> str:
    if stage == "factorial":
        if requested is None:
            raise ValueError(
                "factorial trajectories require --loss m4 or --loss wright"
            )
        return requested
    factorial = decisions.get("factorial")
    if not isinstance(factorial, dict) or not _passed(factorial):
        raise RuntimeError("horizon/cascade requires a computed passing factorial gate")
    selected = factorial.get("selected_loss")
    if selected not in {"m4", "wright"}:
        raise RuntimeError("factorial gate does not contain a resolved selected_loss")
    if requested is not None and requested != selected:
        raise RuntimeError("requested loss differs from the factorial gate selection")
    return str(selected)


def _validate_condition(spec: RunSpec) -> None:
    allowed = {
        "factorial": {
            (device, model)
            for device in ("fulltone", "bigmuff")
            for model in ("a2", "s3")
        },
        "horizon": {
            (device, model)
            for device in ("fulltone", "bigmuff")
            for model in ("rf31", "rf2047")
        },
        "cascade": {
            ("synthetic", "mono"),
            ("synthetic", "cascade"),
            ("bigmuff", "cascade"),
            ("fulltone", "cascade"),
        },
    }
    if (spec.device, spec.model) not in allowed[spec.stage] or spec.seed != 0:
        raise ValueError("trajectory is outside the locked R1 diagnostic matrix")


def _promoted_residual_reference(
    spec: RunSpec, decisions: dict[str, object]
) -> tuple[dict[str, torch.Tensor] | None, dict[str, str] | None]:
    if spec.stage != "cascade" or spec.device == "synthetic":
        return None, None
    horizon = decisions.get("horizon")
    if not isinstance(horizon, dict) or not _passed(horizon):
        raise RuntimeError("physical cascade requires the passing horizon gate")
    references = horizon.get("promoted_residuals")
    if not isinstance(references, dict) or not isinstance(
        references.get(spec.device), dict
    ):
        raise RuntimeError(f"horizon gate has no promoted RF2047 for {spec.device}")
    reference = references[spec.device]
    run_id = reference.get("run_id")
    relative_path = reference.get("path")
    digest = reference.get("sha256")
    if not all(
        isinstance(value, str) and value for value in (run_id, relative_path, digest)
    ):
        raise RuntimeError("promoted residual reference is incomplete")
    source_spec = parse_run_id(run_id)
    if (
        source_spec.stage != "horizon"
        or source_spec.device != spec.device
        or source_spec.model != "rf2047"
        or source_spec.loss != spec.loss
        or source_spec.seed != spec.seed
    ):
        raise RuntimeError("promoted residual provenance does not match this cascade")
    expected = (
        ROOT / "experiments/runs" / run_id / "checkpoints/promoted-residual-state.pt"
    )
    checkpoint = (ROOT / relative_path).resolve()
    if checkpoint != expected.resolve():
        raise RuntimeError("promoted residual path is outside its exact horizon run")
    status = json.loads(
        (expected.parents[1] / "status.json").read_text(encoding="utf-8")
    )
    if status.get("status") != "completed":
        raise RuntimeError("promoted RF2047 source run is not completed")
    state = load_promoted_residual(expected, expected_sha256=digest)
    return state, {"run_id": run_id, "path": relative_path, "sha256": digest}


def _environment(device: torch.device) -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "training_device": str(device),
        "precision": "float32",
    }


def _checkpoint_index(run_dir: Path, selected_step: int) -> dict[str, Any]:
    snapshots = []
    for step in (200, 1_000, 5_000):
        path = run_dir / "checkpoints" / f"step-{step}.pt"
        snapshots.append(
            {
                "step": step,
                "path": str(path.relative_to(run_dir)),
                "sha256": sha256_file(path),
            }
        )
    selected = run_dir / "checkpoints/selected-model-state.pt"
    return {
        "policy": "lowest_validation_esr_at_common_checkpoints",
        "checkpoint_steps": [200, 1_000, 5_000],
        "snapshots": snapshots,
        "selected_step": selected_step,
        "selected_path": str(selected.relative_to(run_dir)),
        "selected_sha256": sha256_file(selected),
    }


def _preflight(
    spec: RunSpec,
    stage_config: dict[str, Any],
    m4_config: dict[str, Any],
) -> dict[str, Any]:
    profile = execution_profile(stage_config, m4_config, preflight=True)
    dataset = (
        load_synthetic_cascade_dataset(ROOT, profile)
        if spec.device == "synthetic"
        else load_m4_dataset(ROOT, spec.device, profile)
    )
    # Counted physical cascade runs must load the gate-selected state.  Preflight
    # deliberately uses a fresh RF2047 because it cannot depend on future results.
    model, kind = make_model(ROOT, stage=spec.stage, model_name=spec.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = run_training(
        model=model,
        model_kind=kind,
        stage=spec.stage,
        model_name=spec.model,
        loss_mode=spec.loss,
        seed=spec.seed,
        dataset=dataset,
        stage_config=stage_config,
        m4_config=m4_config,
        profile=profile,
        device=device,
        checkpoint_directory=None,
    )
    return {
        "status": "passed",
        "preflight": True,
        "counts_toward_r1": False,
        "writes_performed": False,
        "run_id": spec.run_id,
        "optimizer_steps": profile.optimizer_steps,
        "checkpoint_steps": list(profile.checkpoint_steps),
        "selected_checkpoint": result.metrics["selected_checkpoint"],
        "checks": result.metrics["checks"],
        "counted_physical_cascade_requires_promoted_rf2047": True,
    }


def main() -> None:
    args = _parser().parse_args()
    configs = validate_repository_configs(ROOT)
    stage_config = configs[args.stage]
    m4_config = yaml.safe_load((ROOT / M4_CONFIG_PATH).read_text(encoding="utf-8"))
    decisions = _decisions(args.gates)
    loss = _selected_loss(args.stage, args.loss, decisions)
    spec = RunSpec(args.stage, args.device, args.model, loss, args.seed)
    _validate_condition(spec)
    expected_id = spec.run_id
    if args.run_id is not None and args.run_id != expected_id:
        raise ValueError(f"run-id must be exactly {expected_id}")

    if args.plan:
        print(
            json.dumps(
                {
                    "run": spec.as_dict(),
                    "preflight": args.preflight,
                    "writes_performed": False,
                },
                indent=2,
            )
        )
        return
    if args.preflight:
        if args.run_id is not None:
            raise ValueError("preflight does not accept --run-id")
        print(json.dumps(_preflight(spec, stage_config, m4_config), indent=2))
        return
    if args.run_id is None:
        raise ValueError(f"a counted run requires --run-id {expected_id}")
    if args.training_device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("counted R1 diagnostics require an available CUDA device")

    validate_lock_digest(ROOT)
    _require_clean_worktree()
    promoted_state, promoted_reference = _promoted_residual_reference(spec, decisions)
    manifest_bytes = (ROOT / M4_MANIFEST_PATH).read_bytes()
    split_bytes = (ROOT / M4_SPLIT_PATH).read_bytes()
    data_hash = combined_sha256(manifest_bytes, split_bytes)
    synthetic_dataset = None
    profile = execution_profile(stage_config, m4_config, preflight=False)
    if spec.device == "synthetic":
        synthetic_dataset = load_synthetic_cascade_dataset(ROOT, profile)
        data_hash = synthetic_dataset.data_sha256
    commit = _git_commit()
    command = " ".join(["uv", "run", "python", *sys.argv])
    executor = R1Executor(ROOT, stage_configs=configs, gate_decisions=decisions)
    executor.prepare_run(
        spec,
        data_sha256=data_hash,
        commit=commit,
        command=command,
    )
    run_dir = ROOT / "experiments/runs" / spec.run_id
    for name in ("checkpoints", "predictions", "figures"):
        (run_dir / name).mkdir()
    started = time.perf_counter()
    failure_reason = ""
    try:
        dataset = synthetic_dataset or load_m4_dataset(ROOT, spec.device, profile)
        (run_dir / "dataset-manifest.json").write_bytes(dataset.manifest_bytes)
        (run_dir / "split-manifest.json").write_bytes(dataset.split_bytes)
        (run_dir / "seed.txt").write_text(f"{spec.seed}\n", encoding="utf-8")
        (run_dir / "git-commit.txt").write_text(commit + "\n", encoding="utf-8")
        (run_dir / "execution-resolved.json").write_text(
            json.dumps(
                {
                    "profile": {
                        "optimizer_steps": profile.optimizer_steps,
                        "checkpoint_steps": list(profile.checkpoint_steps),
                        "context_samples": profile.context_samples,
                        "output_samples": profile.output_samples,
                        "batch_size": profile.batch_size,
                        "validation_samples": profile.validation_samples,
                        "evaluation_block_samples": profile.evaluation_block_samples,
                    },
                    "m4_config": str(M4_CONFIG_PATH),
                    "m4_manifest": str(M4_MANIFEST_PATH),
                    "m4_split": str(M4_SPLIT_PATH),
                    "promoted_residual": promoted_reference,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        device = torch.device("cuda")
        (run_dir / "environment.json").write_text(
            json.dumps(_environment(device), indent=2) + "\n", encoding="utf-8"
        )
        model, model_kind = make_model(
            ROOT,
            stage=spec.stage,
            model_name=spec.model,
            promoted_residual_state=promoted_state,
        )
        result = run_training(
            model=model,
            model_kind=model_kind,
            stage=spec.stage,
            model_name=spec.model,
            loss_mode=spec.loss,
            seed=spec.seed,
            dataset=dataset,
            stage_config=stage_config,
            m4_config=m4_config,
            profile=profile,
            device=device,
            checkpoint_directory=run_dir / "checkpoints",
        )
        metrics = dict(result.metrics)
        metrics["run_id"] = spec.run_id
        metrics["device"] = spec.device
        metrics["data_sha256"] = data_hash
        metrics["promoted_residual"] = promoted_reference
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        (run_dir / "training-history.json").write_text(
            json.dumps(result.history, indent=2) + "\n", encoding="utf-8"
        )
        result.selected_validation_prediction.tofile(
            run_dir / "predictions/selected_validation_prediction.f32"
        )
        checkpoint_index = _checkpoint_index(
            run_dir, int(result.metrics["selected_checkpoint"]["step"])
        )
        if spec.stage == "horizon" and spec.model == "rf2047":
            if result.selected_residual_state is None:
                raise RuntimeError("RF2047 run did not expose its residual state")
            residual_path = run_dir / "checkpoints/promoted-residual-state.pt"
            torch.save(result.selected_residual_state, residual_path)
            checkpoint_index["promoted_residual"] = {
                "path": str(residual_path.relative_to(run_dir)),
                "sha256": sha256_file(residual_path),
                "source_step": result.metrics["selected_checkpoint"]["step"],
                "receptive_field": 2_047,
            }
        (run_dir / "checkpoints/index.json").write_text(
            json.dumps(checkpoint_index, indent=2) + "\n", encoding="utf-8"
        )
        (run_dir / "model-export.json").write_text(
            json.dumps(
                {
                    "model": spec.model,
                    "state_dict": "checkpoints/selected-model-state.pt",
                    "latency_samples": result.metrics["latency_samples"],
                    "promoted_residual_source": promoted_reference,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        executor.finalize_run(spec, status="completed")
    except BaseException as error:
        failure_reason = f"{type(error).__name__}: {error}"
        (run_dir / "failure.json").write_text(
            json.dumps(
                {"reason": failure_reason, "traceback": traceback.format_exc()},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        executor.finalize_run(
            spec,
            status="failed",
            failure_reason=failure_reason,
        )
        raise
    finally:
        (run_dir / "timings.json").write_text(
            json.dumps(
                {
                    "wall_seconds": time.perf_counter() - started,
                    "completed": not failure_reason,
                    "gpu_peak_memory_bytes": (
                        torch.cuda.max_memory_allocated()
                        if torch.cuda.is_available()
                        else None
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"status": "completed", "run_id": spec.run_id}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
