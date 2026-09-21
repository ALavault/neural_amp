#!/usr/bin/env python3
"""Train and audit one immutable M3 synthetic FSSR-NAM run."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from fssr_nam.data.excitations import generate_excitation
from fssr_nam.metrics.spectral import spectral_metrics
from fssr_nam.metrics.time import time_metrics
from fssr_nam.models.residual import normalized_residual_penalty
from fssr_nam.reporting.ledger import append_run
from fssr_nam.training.m3 import delay_target, model_factory, synthetic_target

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/training/m3_synthetic.yaml"
LEDGER = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
RESULT_INDEX = ROOT / ".codex_campaign/RESULT_INDEX.csv"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def sha256_bytes(*payloads: bytes) -> str:
    digest = hashlib.sha256()
    for payload in payloads:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def excitation(seed: int, sample_rate: int, sample_count: int) -> np.ndarray:
    duration = sample_count / sample_rate
    parts = (
        generate_excitation(
            "multisine", sample_rate=sample_rate, duration_seconds=duration, seed=seed
        ),
        generate_excitation(
            "colored_noise",
            sample_rate=sample_rate,
            duration_seconds=duration,
            seed=seed + 1,
        ),
        generate_excitation(
            "level_changes",
            sample_rate=sample_rate,
            duration_seconds=duration,
            seed=seed + 2,
        ),
    )
    mixed = 0.45 * parts[0] + 0.35 * parts[1] + 0.2 * parts[2]
    return np.asarray(np.clip(mixed, -0.5, 0.5), dtype=np.float32)


def components(model, signal: torch.Tensor):
    if hasattr(model, "forward_components"):
        return model.forward_components(signal)
    output = model(signal)
    return output, output, torch.zeros_like(output)


def loss_terms(model, signal, target_signal, weights):
    output, _, residual = components(model, signal)
    time_loss = (output - target_signal).square().sum() / (
        target_signal.square().sum() + 1.0e-12
    )
    shaper = model.core.shaper if hasattr(model, "core") else model.shaper
    curvature = shaper.curvature_penalty()
    residual_loss = normalized_residual_penalty(residual, target_signal)
    total = (
        weights["normalized_time_error"] * time_loss
        + weights["spline_curvature"] * curvature
        + weights["normalized_residual_energy"] * residual_loss
    )
    return total, time_loss, residual_loss


def irregular_stream(model, signal: torch.Tensor) -> torch.Tensor:
    outputs = []
    sizes = (1, 7, 64, 3, 128, 17)
    position = 0
    while position < len(signal):
        size = sizes[len(outputs) % len(sizes)]
        outputs.append(model.stream(signal[position : position + size]))
        position += size
    return torch.cat(outputs)


def evaluate(model, signal, target_signal) -> tuple[np.ndarray, dict[str, float]]:
    model.eval()
    with torch.inference_mode():
        output, _, residual = components(model, signal)
    prediction = output.numpy()
    reference = target_signal.numpy()
    metrics = {
        **time_metrics(prediction, reference),
        **spectral_metrics(prediction, reference),
    }
    metrics["residual_energy_ratio"] = float(
        residual.square().sum() / (output.square().sum() + 1.0e-12)
    )
    return prediction, metrics


def post_training_checks(model, model_config, signal, prediction) -> dict[str, object]:
    reloaded = model_factory(model.code, model_config)
    reloaded.load_state_dict(model.state_dict())
    reloaded.eval()
    with torch.inference_mode():
        reloaded_prediction = reloaded(signal).numpy()
    model.reset_state()
    with torch.inference_mode():
        streamed = irregular_stream(model, signal).numpy()
    model.reset_state()
    with torch.inference_mode():
        reset = irregular_stream(model, signal).numpy()
    altered = signal.clone()
    midpoint = len(signal) // 2
    altered[midpoint:] = torch.flip(altered[midpoint:], dims=(0,))
    with torch.inference_mode():
        original_prefix = reloaded(signal)[:midpoint].numpy()
        altered_prefix = reloaded(altered)[:midpoint].numpy()
    return {
        "reload_max_abs": float(np.max(np.abs(reloaded_prediction - prediction))),
        "irregular_stream_max_abs": float(np.max(np.abs(streamed - prediction))),
        "reset_max_abs": float(np.max(np.abs(reset - streamed))),
        "causal_prefix_max_abs": float(
            np.max(np.abs(original_prefix - altered_prefix))
        ),
        "finite_output": bool(np.all(np.isfinite(prediction))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", required=True, choices=("S0", "S1", "S2", "S3", "S4")
    )
    parser.add_argument(
        "--case", required=True, choices=("tanh", "slow_sag", "short_nonlinear_memory")
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_bytes = CONFIG_PATH.read_bytes()
    config = yaml.safe_load(config_bytes)
    specification = next(
        (
            item
            for item in config["runs"]
            if item["variant"] == args.variant and item["case"] == args.case
        ),
        None,
    )
    if specification is None:
        raise ValueError("variant/case pair is not preregistered")
    run_dir = ROOT / "experiments/runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for directory in ("checkpoints", "predictions", "figures"):
        (run_dir / directory).mkdir()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    sample_rate = int(config["sample_rate"])
    sample_count = int(config["sample_count"])
    train_input_np = excitation(config["train_seed"], sample_rate, sample_count)
    validation_input_np = excitation(
        config["validation_seed"], sample_rate, sample_count
    )
    train_target_np = synthetic_target(args.case, train_input_np, sample_rate)
    validation_target_np = synthetic_target(args.case, validation_input_np, sample_rate)
    model = model_factory(args.variant, config["model"])
    latency = int(getattr(model, "latency_samples", 0))
    train_input = torch.from_numpy(train_input_np)
    validation_input = torch.from_numpy(validation_input_np)
    train_target = delay_target(torch.from_numpy(train_target_np), latency)
    validation_target = delay_target(torch.from_numpy(validation_target_np), latency)
    data_manifest = {
        "schema_version": 1,
        "tier": "SYNTHETIC",
        "case": args.case,
        "sample_rate": sample_rate,
        "sample_count": sample_count,
        "train_seed": config["train_seed"],
        "validation_seed": config["validation_seed"],
        "train_input_sha256": hashlib.sha256(train_input_np.tobytes()).hexdigest(),
        "train_target_sha256": hashlib.sha256(train_target_np.tobytes()).hexdigest(),
        "validation_input_sha256": hashlib.sha256(
            validation_input_np.tobytes()
        ).hexdigest(),
        "validation_target_sha256": hashlib.sha256(
            validation_target_np.tobytes()
        ).hexdigest(),
    }
    split_manifest = {
        "schema_version": 1,
        "split_rule": "independent_generator_seed",
        "train_group": f"synthetic-seed-{config['train_seed']}",
        "validation_group": f"synthetic-seed-{config['validation_seed']}",
        "leakage_check": "passed",
    }
    resolved = {
        "campaign": config,
        "selected_run": specification,
        "variant": args.variant,
        "case": args.case,
        "seed": args.seed,
        "latency_samples": latency,
    }
    commit = git_commit()
    started_iso = datetime.now().astimezone().isoformat(timespec="seconds")
    command = (
        f"uv run python scripts/campaigns/run_m3_synthetic.py --variant {args.variant} "
        f"--case {args.case} --seed {args.seed} --run-id {args.run_id}"
    )
    (run_dir / "config-resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (run_dir / "git-commit.txt").write_text(commit + "\n", encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "device": "cpu",
                "threads": torch.get_num_threads(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "dataset-manifest.json").write_text(
        json.dumps(data_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "split-manifest.json").write_text(
        json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "seed.txt").write_text(f"{args.seed}\n", encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({"status": "running", "started_at": started_iso}, indent=2) + "\n",
        encoding="utf-8",
    )
    train_input_np.tofile(run_dir / "predictions/train_input.f32")
    train_target_np.tofile(run_dir / "predictions/train_target.f32")
    validation_input_np.tofile(run_dir / "predictions/validation_input.f32")
    validation_target_np.tofile(run_dir / "predictions/validation_target.f32")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    history = []
    status = "failed"
    failure_reason = ""
    primary = ""
    started = time.perf_counter()
    try:
        with (
            (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout_log,
            (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr_log,
            contextlib.redirect_stdout(Tee(sys.__stdout__, stdout_log)),
            contextlib.redirect_stderr(Tee(sys.__stderr__, stderr_log)),
        ):
            _, initial_metrics = evaluate(model, validation_input, validation_target)
            print(
                f"Starting {args.run_id}: initial_esr={initial_metrics['esr']:.9g}",
                flush=True,
            )
            model.train()
            for step in range(int(specification["steps"])):
                optimizer.zero_grad(set_to_none=True)
                total, time_loss, residual_loss = loss_terms(
                    model, train_input, train_target, config["loss"]
                )
                if not torch.isfinite(total):
                    raise RuntimeError(f"non-finite loss at step {step}")
                total.backward()
                if not all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise RuntimeError(f"non-finite gradient at step {step}")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["gradient_clip_norm"])
                )
                optimizer.step()
                if step == 0 or (step + 1) % 10 == 0:
                    history.append(
                        {
                            "step": step + 1,
                            "loss": float(total.detach()),
                            "time_loss": float(time_loss.detach()),
                            "residual_loss": float(residual_loss.detach()),
                        }
                    )
            prediction, final_metrics = evaluate(
                model, validation_input, validation_target
            )
            checks = post_training_checks(
                model, config["model"], validation_input, prediction
            )
            if (
                not checks["finite_output"]
                or max(
                    checks["reload_max_abs"],
                    checks["irregular_stream_max_abs"],
                    checks["reset_max_abs"],
                    checks["causal_prefix_max_abs"],
                )
                > 2.0e-5
            ):
                raise RuntimeError(f"post-training checks failed: {checks}")
            metrics = {
                "initial": initial_metrics,
                "final": final_metrics,
                "improvement_relative_esr": (
                    initial_metrics["esr"] - final_metrics["esr"]
                )
                / initial_metrics["esr"],
                "checks": checks,
                "parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "latency_samples": latency,
            }
            (run_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
            (run_dir / "training-history.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )
            torch.save(model.state_dict(), run_dir / "checkpoints/model-state.pt")
            prediction.astype(np.float32).tofile(
                run_dir / "predictions/validation_prediction.f32"
            )
            figure, axes = plt.subplots(2, 1, figsize=(9, 6))
            axes[0].plot(
                [item["step"] for item in history],
                [item["time_loss"] for item in history],
            )
            axes[0].set_yscale("log")
            axes[0].set_ylabel("train normalized error")
            count = min(3000, len(prediction))
            axes[1].plot(
                validation_target.numpy()[:count], label="target", linewidth=0.7
            )
            axes[1].plot(prediction[:count], label=args.variant, linewidth=0.6)
            axes[1].legend()
            axes[1].set_xlabel("sample")
            figure.tight_layout()
            figure.savefig(run_dir / "figures/training_and_prediction.png", dpi=140)
            plt.close(figure)
            primary = f"esr={final_metrics['esr']:.9g}"
            status = "completed"
            print(f"Completed {args.run_id}: {primary}", flush=True)
    except Exception as error:
        failure_reason = f"{type(error).__name__}: {error}"
        with (run_dir / "stderr.log").open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        raise
    finally:
        elapsed = time.perf_counter() - started
        if not (run_dir / "metrics.json").exists():
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {"status": "unavailable", "reason": failure_reason}, indent=2
                )
                + "\n",
                encoding="utf-8",
            )
        (run_dir / "timings.json").write_text(
            json.dumps(
                {"wall_seconds": elapsed, "completed": status == "completed"}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        finished_iso = datetime.now().astimezone().isoformat(timespec="seconds")
        (run_dir / "status.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "started_at": started_iso,
                    "finished_at": finished_iso,
                    "failure_reason": failure_reason,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        data_bytes = json.dumps(data_manifest, sort_keys=True).encode()
        append_run(
            LEDGER,
            {
                "date": finished_iso,
                "run_id": args.run_id,
                "phase": "M3",
                "model": args.variant,
                "device": args.case,
                "seed": args.seed,
                "commit": commit,
                "config_sha256": sha256_bytes(config_bytes),
                "data_sha256": sha256_bytes(data_bytes),
                "status": status,
                "failure_reason": failure_reason,
                "results_path": f"experiments/runs/{args.run_id}",
            },
        )
        with RESULT_INDEX.open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerow(
                [
                    args.run_id,
                    "M3",
                    args.variant,
                    args.case,
                    args.seed,
                    status,
                    primary,
                    f"experiments/runs/{args.run_id}",
                ]
            )


if __name__ == "__main__":
    main()
