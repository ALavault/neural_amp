#!/usr/bin/env python3
"""Train and audit one immutable M4 physical-device smoke run."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
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
import soundfile as sf
import torch
import yaml
from nam._dependencies.auraloss.freq import MultiResolutionSTFTLoss

from fssr_nam.metrics.spectral import spectral_metrics
from fssr_nam.metrics.time import time_metrics
from fssr_nam.models.residual import normalized_residual_penalty
from fssr_nam.reporting.ledger import append_run
from fssr_nam.training.m4 import (
    causal_predict,
    delay_target,
    latency_samples,
    model_factory,
    training_prediction,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/training/m4_smoke.yaml"
RECOVERY_CONFIG_PATH = ROOT / "configs/training/m4_recovery.yaml"
MEMORY_CONFIG_PATH = ROOT / "configs/training/m4_memory.yaml"
GRID_CONFIG_PATH = ROOT / "configs/training/m4_grid.yaml"
MODEL_CONFIG_PATH = ROOT / "configs/training/m3_synthetic.yaml"
MANIFEST_PATH = ROOT / "datasets/manifests/m4_internal.json"
SPLIT_PATH = ROOT / "datasets/splits/m4_internal.json"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("B0", "B2", "S3", "S4"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--recovery-wide", action="store_true")
    parser.add_argument("--memory-taps", action="store_true")
    parser.add_argument("--grid-range", action="store_true")
    return parser.parse_args()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combined_hash(*payloads: bytes) -> str:
    digest = hashlib.sha256()
    for payload in payloads:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def environment(device: torch.device) -> dict[str, object]:
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
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
    }


def load_device_audio(
    manifest: dict, device_name: str
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    files = {
        item["split"]: item
        for item in manifest["files"]
        if item["device"] == device_name
    }
    if set(files) != {"train", "validation", "test"}:
        raise RuntimeError(f"incomplete manifest for {device_name}")
    result = {}
    for split, entry in files.items():
        input_path = ROOT / entry["input_path"]
        target_path = ROOT / entry["target_path"]
        if sha256(input_path) != entry["input_sha256"]:
            raise RuntimeError(f"input checksum mismatch for {device_name}/{split}")
        if sha256(target_path) != entry["target_sha256"]:
            raise RuntimeError(f"target checksum mismatch for {device_name}/{split}")
        x, input_rate = sf.read(input_path, dtype="float32")
        y, target_rate = sf.read(target_path, dtype="float32")
        if input_rate != 48_000 or target_rate != 48_000 or x.shape != y.shape:
            raise RuntimeError(f"invalid prepared pair for {device_name}/{split}")
        result[split] = (x, y)
    return result


def esr(prediction: np.ndarray, target: np.ndarray) -> float:
    numerator = np.sum(np.square(prediction - target), dtype=np.float64)
    denominator = np.sum(np.square(target), dtype=np.float64)
    return float(numerator / max(denominator, np.finfo(float).eps))


def forward_components(model, code: str, windows: torch.Tensor, output_samples: int):
    if code not in {"S3", "S4"}:
        prediction = training_prediction(model, code, windows, output_samples)
        return prediction, prediction.new_zeros(())
    output, _, residual = model.forward_components(windows)
    return output[..., -output_samples:], residual[..., -output_samples:]


def residual_ratio(
    model, code: str, signal: np.ndarray, block_size: int, device
) -> float:
    if code not in {"S3", "S4"}:
        return 0.0
    model.reset_state()
    output_energy = 0.0
    residual_energy = 0.0
    with torch.inference_mode():
        for start in range(0, signal.size, block_size):
            chunk = torch.from_numpy(signal[start : start + block_size]).to(device)
            output, _, residual = model.stream_components(chunk)
            output_energy += float(output.double().square().sum().cpu())
            residual_energy += float(residual.double().square().sum().cpu())
    return residual_energy / max(output_energy, np.finfo(float).eps)


def save_figure(history, target, prediction, path: Path, code: str) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(9, 8))
    axes[0].plot([item["step"] for item in history], [item["loss"] for item in history])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("training loss")
    count = min(4000, target.size)
    axes[1].plot(target[:count], label="target", linewidth=0.7)
    axes[1].plot(prediction[:count], label=code, linewidth=0.6)
    axes[1].legend()
    axes[1].set_ylabel("amplitude")
    axes[2].specgram(target - prediction, NFFT=1024, Fs=48_000, noverlap=768)
    axes[2].set_ylabel("error frequency (Hz)")
    axes[2].set_xlabel("time (s)")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def append_result_index(entry: list[object]) -> None:
    with RESULT_INDEX.open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerow(entry)


def main() -> None:
    args = parse_args()
    config_bytes = CONFIG_PATH.read_bytes()
    manifest_bytes = MANIFEST_PATH.read_bytes()
    split_bytes = SPLIT_PATH.read_bytes()
    config = yaml.safe_load(config_bytes)
    manifest = json.loads(manifest_bytes)
    split_manifest = json.loads(split_bytes)
    if args.device not in config["devices"] or args.seed not in config["seeds"]:
        raise ValueError("device or seed is outside the preregistered M4 matrix")
    recovery = None
    if args.recovery_wide:
        recovery = yaml.safe_load(RECOVERY_CONFIG_PATH.read_text(encoding="utf-8"))
        if (
            args.preflight
            or args.model != recovery["model"]
            or args.seed != recovery["seed"]
            or args.device not in recovery["devices"]
        ):
            raise ValueError("run is outside the preregistered M4 recovery")
    memory = None
    if args.memory_taps:
        memory = yaml.safe_load(MEMORY_CONFIG_PATH.read_text(encoding="utf-8"))
        if (
            args.recovery_wide
            or args.model not in memory["models"]
            or args.seed not in memory["seeds"]
            or args.device not in memory["devices"]
        ):
            raise ValueError("run is outside the preregistered M4 memory diagnostic")
    grid = None
    if args.grid_range:
        grid = yaml.safe_load(GRID_CONFIG_PATH.read_text(encoding="utf-8"))
        if (
            args.recovery_wide
            or args.memory_taps
            or args.model not in grid["models"]
            or args.seed not in grid["seeds"]
            or args.device not in grid["devices"]
        ):
            raise ValueError("run is outside the preregistered M4 grid diagnostic")
    run_dir = ROOT / "experiments/runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for directory in ("checkpoints", "predictions", "figures"):
        (run_dir / directory).mkdir()

    execution = dict(config["preflight"] if args.preflight else {})
    steps = int(execution.get("optimizer_steps", config["optimizer_steps"]))
    validation_interval = int(
        execution.get("validation_interval_steps", config["validation_interval_steps"])
    )
    validation_samples = int(
        execution.get("validation_samples", config["validation_samples"])
    )
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))[
        "model"
    ]
    if recovery is not None:
        model_config["residual_channels"] = int(recovery["residual_channels"])
    if memory is not None:
        model_config["taps"] = int(memory["taps"])
    if grid is not None:
        model_config["taps"] = int(grid["taps"])
        model_config["spline_range"] = float(grid["spline_range"])
    resolved = {
        "campaign": config,
        "model": args.model,
        "device": args.device,
        "seed": args.seed,
        "preflight": args.preflight,
        "recovery": recovery,
        "memory": memory,
        "grid": grid,
        "execution": {
            "optimizer_steps": steps,
            "validation_interval_steps": validation_interval,
            "validation_samples": validation_samples,
        },
        "fssr_model": model_config if args.model in {"S3", "S4"} else None,
    }
    resolved_bytes = yaml.safe_dump(resolved, sort_keys=False).encode()
    config_hash = hashlib.sha256(resolved_bytes).hexdigest()
    data_hash = combined_hash(manifest_bytes, split_bytes)
    commit = git_commit()
    started_iso = datetime.now().astimezone().isoformat(timespec="seconds")
    command = (
        f"uv run python scripts/run_m4_smoke.py --model {args.model} "
        f"--device {args.device} --seed {args.seed} --run-id {args.run_id}"
        + (" --preflight" if args.preflight else "")
        + (" --recovery-wide" if args.recovery_wide else "")
        + (" --memory-taps" if args.memory_taps else "")
        + (" --grid-range" if args.grid_range else "")
    )
    (run_dir / "config-resolved.yaml").write_bytes(resolved_bytes)
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (run_dir / "git-commit.txt").write_text(commit + "\n", encoding="utf-8")
    (run_dir / "dataset-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "split-manifest.json").write_text(
        json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "seed.txt").write_text(f"{args.seed}\n", encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({"status": "running", "started_at": started_iso}, indent=2) + "\n",
        encoding="utf-8",
    )

    seed_everything(args.seed)
    training_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (run_dir / "environment.json").write_text(
        json.dumps(environment(training_device), indent=2) + "\n", encoding="utf-8"
    )
    audio = load_device_audio(manifest, args.device)
    train_x, train_y = audio["train"]
    validation_x, validation_y = audio["validation"]
    test_x, test_y = audio["test"]
    model = model_factory(args.model, root=ROOT, model_config=model_config).to(
        training_device
    )
    latency = latency_samples(model)
    train_target = delay_target(train_y, latency)
    validation_target = delay_target(validation_y, latency)
    test_target = delay_target(test_y, latency)
    output_samples = int(config["output_samples"])
    context_samples = int(config["context_samples"])
    batch_size = int(config["batch_size"])
    block_samples = int(config["evaluation_block_samples"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    mrstft = MultiResolutionSTFTLoss().to(training_device)
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float | int]] = []
    best_esr = float("inf")
    best_step = 0
    status = "failed"
    failure_reason = ""
    primary = ""
    if grid is not None:
        phase = "M4_GRID_PREFLIGHT" if args.preflight else "M4_GRID"
    elif memory is not None:
        phase = "M4_MEMORY_PREFLIGHT" if args.preflight else "M4_MEMORY"
    elif args.preflight:
        phase = "M4_PREFLIGHT"
    elif args.recovery_wide:
        phase = "M4_RECOVERY"
    else:
        phase = "M4"
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        with (
            (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout_log,
            (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr_log,
            contextlib.redirect_stdout(Tee(sys.__stdout__, stdout_log)),
            contextlib.redirect_stderr(Tee(sys.__stderr__, stderr_log)),
        ):
            initial_prediction = causal_predict(
                model,
                args.model,
                validation_x[:validation_samples],
                device=training_device,
                context_samples=context_samples,
                block_samples=block_samples,
            )
            initial_esr = esr(
                initial_prediction, validation_target[:validation_samples]
            )
            best_esr = initial_esr
            best_step = 0
            torch.save(
                model.state_dict(),
                run_dir / "checkpoints/model-state.pt",
            )
            print(f"Starting {args.run_id}: initial_esr={initial_esr:.9g}")
            for step in range(1, steps + 1):
                starts = rng.integers(
                    context_samples,
                    train_x.size - output_samples,
                    size=batch_size,
                )
                windows_np = np.stack(
                    [
                        train_x[start - context_samples : start + output_samples]
                        for start in starts
                    ]
                )
                targets_np = np.stack(
                    [train_target[start : start + output_samples] for start in starts]
                )
                windows = torch.from_numpy(windows_np).to(training_device)
                targets = torch.from_numpy(targets_np).to(training_device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                prediction, residual_signal = forward_components(
                    model, args.model, windows, output_samples
                )
                mse_loss = torch.nn.functional.mse_loss(prediction, targets)
                spectral_loss = mrstft(prediction[:, None], targets[:, None])
                curvature = prediction.new_zeros(())
                if hasattr(model, "core"):
                    curvature = model.core.regularization()
                residual_loss = prediction.new_zeros(())
                if args.model in {"S3", "S4"}:
                    residual_loss = normalized_residual_penalty(
                        residual_signal, targets
                    )
                loss_config = config["loss"]
                total = (
                    float(loss_config["mse"]) * mse_loss
                    + float(loss_config["mrstft"]) * spectral_loss
                    + float(loss_config["spline_curvature"]) * curvature
                    + float(loss_config["normalized_residual_energy"]) * residual_loss
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
                record = {
                    "step": step,
                    "loss": float(total.detach()),
                    "mse": float(mse_loss.detach()),
                    "mrstft": float(spectral_loss.detach()),
                    "residual": float(residual_loss.detach()),
                }
                if step % validation_interval == 0 or step == steps:
                    validation_prediction = causal_predict(
                        model,
                        args.model,
                        validation_x[:validation_samples],
                        device=training_device,
                        context_samples=context_samples,
                        block_samples=block_samples,
                    )
                    validation_esr = esr(
                        validation_prediction,
                        validation_target[:validation_samples],
                    )
                    record["validation_esr"] = validation_esr
                    print(
                        f"step={step} loss={record['loss']:.9g} "
                        f"validation_esr={validation_esr:.9g}"
                    )
                    if validation_esr < best_esr:
                        best_esr = validation_esr
                        best_step = step
                        torch.save(
                            model.state_dict(),
                            run_dir / "checkpoints/model-state.pt",
                        )
                history.append(record)

            model.load_state_dict(
                torch.load(
                    run_dir / "checkpoints/model-state.pt",
                    map_location=training_device,
                    weights_only=True,
                )
            )
            evaluation_count = validation_samples if args.preflight else test_x.size
            final_input = test_x[:evaluation_count]
            final_target = test_target[:evaluation_count]
            final_prediction = causal_predict(
                model,
                args.model,
                final_input,
                device=training_device,
                context_samples=context_samples,
                block_samples=block_samples,
            )
            alternate_prediction = causal_predict(
                model,
                args.model,
                final_input[: min(final_input.size, 32768)],
                device=training_device,
                context_samples=context_samples,
                block_samples=4093,
            )
            normal_prefix = final_prediction[: alternate_prediction.size]
            block_max_abs = float(
                np.max(np.abs(normal_prefix - alternate_prediction), initial=0.0)
            )
            if block_max_abs > 2.0e-5:
                raise RuntimeError(
                    f"post-training block parity failed: {block_max_abs:.9g}"
                )
            test_metrics = {
                **time_metrics(final_prediction, final_target),
                **spectral_metrics(final_prediction, final_target),
            }
            test_metrics["residual_energy_ratio"] = residual_ratio(
                model, args.model, final_input, block_samples, training_device
            )
            metrics = {
                "initial_validation_esr": initial_esr,
                "best_validation_esr": best_esr,
                "best_step": best_step,
                "test": test_metrics,
                "checks": {
                    "finite_prediction": bool(np.all(np.isfinite(final_prediction))),
                    "alternate_block_max_abs": block_max_abs,
                },
                "parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "latency_samples": latency,
                "training_output_samples_seen": (steps * batch_size * output_samples),
                "preflight": args.preflight,
            }
            (run_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
            (run_dir / "training-history.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )
            final_prediction.tofile(run_dir / "predictions/test_prediction.f32")
            save_figure(
                history,
                final_target,
                final_prediction,
                run_dir / "figures/training_prediction_error.png",
                args.model,
            )
            checkpoint_path = run_dir / "checkpoints/model-state.pt"
            (run_dir / "checkpoints/index.json").write_text(
                json.dumps(
                    {
                        "path": "model-state.pt",
                        "sha256": sha256(checkpoint_path),
                        "selection": "lowest_validation_esr",
                        "step": best_step,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if args.model == "B0":
                model.sample_rate = 48_000
                model.export(run_dir, basename="model")
            else:
                (run_dir / "model-export.json").write_text(
                    json.dumps(
                        {
                            "model": args.model,
                            "latency_samples": latency,
                            "state_dict": "checkpoints/model-state.pt",
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            primary = f"esr={test_metrics['esr']:.9g}"
            status = "completed"
            print(f"Completed {args.run_id}: {primary}")
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
        if not (run_dir / "checkpoints/index.json").exists():
            (run_dir / "checkpoints/index.json").write_text(
                json.dumps({"status": "unavailable"}, indent=2) + "\n",
                encoding="utf-8",
            )
        (run_dir / "timings.json").write_text(
            json.dumps(
                {
                    "wall_seconds": elapsed,
                    "completed": status == "completed",
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
        relative_path = str(run_dir.relative_to(ROOT))
        append_run(
            LEDGER,
            {
                "date": finished_iso,
                "run_id": args.run_id,
                "phase": phase,
                "model": args.model,
                "device": args.device,
                "seed": args.seed,
                "commit": commit,
                "config_sha256": config_hash,
                "data_sha256": data_hash,
                "status": status,
                "failure_reason": failure_reason,
                "results_path": relative_path,
            },
        )
        append_result_index(
            [
                args.run_id,
                phase,
                args.model,
                args.device,
                args.seed,
                status,
                primary,
                relative_path,
            ]
        )


if __name__ == "__main__":
    main()
