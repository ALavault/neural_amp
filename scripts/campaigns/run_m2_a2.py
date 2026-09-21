#!/usr/bin/env python3
"""Run one immutable official packed A2 reproduction training."""

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
import pytorch_lightning as pl
import soundfile as sf
import torch
import yaml
from nam.models import init_from_nam
from nam.train import full
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from fssr_nam.metrics.spectral import spectral_metrics
from fssr_nam.metrics.time import time_metrics
from fssr_nam.reporting.ledger import append_run
from fssr_nam.training.nam_a2 import make_configs

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_CONFIG = ROOT / "configs/training/m2_a2_smoke.yaml"
MODEL_CONFIG = (
    ROOT
    / "third_party/neural-amp-modeler/nam/train/_resources/config_model_packed.json"
)
DATASET_MANIFEST = ROOT / "datasets/manifests/m2_synthetic_tanh.json"
SPLIT_MANIFEST = ROOT / "datasets/splits/m2_synthetic_tanh.json"
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


def seed_everything(seed: int, deterministic_mode: str) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True, verbose=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if deterministic_mode not in {"strict", "warn_only", "off"}:
        raise ValueError(f"unknown deterministic mode: {deterministic_mode}")
    torch.use_deterministic_algorithms(
        deterministic_mode != "off", warn_only=deterministic_mode == "warn_only"
    )


def environment() -> dict[str, object]:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pytorch_lightning": pl.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
    }


def extract_history(run_dir: Path) -> dict[str, list[dict[str, float | int]]]:
    event_files = sorted(run_dir.glob("lightning_logs/**/events.out.tfevents.*"))
    if not event_files:
        raise RuntimeError("official trainer produced no TensorBoard event file")
    accumulator = EventAccumulator(str(event_files[-1]))
    accumulator.Reload()
    history = {}
    for tag in accumulator.Tags().get("scalars", []):
        history[tag] = [
            {"step": event.step, "value": float(event.value)}
            for event in accumulator.Scalars(tag)
        ]
    return history


def evaluate_exports(run_dir: Path, manifest: dict) -> dict[str, object]:
    container = json.loads((run_dir / "model.nam").read_text(encoding="utf-8"))
    labels = ("lite", "full")
    validation = next(
        item for item in manifest["files"] if item["name"] == "validation"
    )
    x, rate = sf.read(ROOT / validation["input_path"], dtype="float32")
    target, target_rate = sf.read(ROOT / validation["output_path"], dtype="float32")
    if rate != target_rate or rate != 48_000:
        raise RuntimeError("unexpected validation sample rate")
    predictions = {}
    metrics = {}
    for label, submodel in zip(labels, container["config"]["submodels"], strict=True):
        model_dict = submodel["model"]
        (run_dir / f"model_{label}.nam").write_text(
            json.dumps(model_dict) + "\n", encoding="utf-8"
        )
        model = init_from_nam(model_dict).eval()
        with torch.inference_mode():
            prediction = model(torch.from_numpy(x), pad_start=True).cpu().numpy()
        if prediction.shape != target.shape or not np.all(np.isfinite(prediction)):
            raise RuntimeError(f"invalid {label} exported prediction")
        predictions[label] = prediction
        np.save(run_dir / "predictions" / f"validation_{label}.npy", prediction)
        metrics[label] = {
            **time_metrics(prediction, target),
            **spectral_metrics(prediction, target),
            "parameters": len(model_dict["weights"]) - 1,
            "exported_weights": len(model_dict["weights"]),
        }
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    stop = min(len(target), 24_000)
    axes[0].plot(target[:stop], label="target", linewidth=0.8)
    axes[0].plot(predictions["lite"][:stop], label="A2 Lite", linewidth=0.6)
    axes[0].plot(predictions["full"][:stop], label="A2 Full", linewidth=0.6)
    axes[0].legend()
    axes[0].set_ylabel("amplitude")
    axes[1].plot(target[:stop] - predictions["lite"][:stop], label="Lite error")
    axes[1].plot(target[:stop] - predictions["full"][:stop], label="Full error")
    axes[1].legend()
    axes[1].set_xlabel("sample")
    axes[1].set_ylabel("error")
    figure.tight_layout()
    figure.savefig(run_dir / "figures/validation_prediction.png", dpi=140)
    plt.close(figure)
    return metrics


def plot_history(history: dict, run_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    for tag in ("val_loss_packed_0", "val_loss_packed_1"):
        events = history.get(tag, [])
        if events:
            axis.plot(
                [event["step"] for event in events],
                [event["value"] for event in events],
                marker="o",
                label=tag,
            )
    axis.set_yscale("log")
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("validation ESR")
    axis.legend()
    figure.tight_layout()
    figure.savefig(run_dir / "figures/validation_curve.png", dpi=140)
    plt.close(figure)


def checkpoint_index(run_dir: Path) -> None:
    paths = sorted(run_dir.glob("*.ckpt")) + sorted(
        run_dir.glob("lightning_logs/**/*.ckpt")
    )
    payload = {
        "deduplication": "checkpoints remain at official trainer paths; no copies made",
        "files": [
            {"path": str(path.relative_to(run_dir)), "bytes": path.stat().st_size}
            for path in paths
        ],
    }
    (run_dir / "checkpoints/index.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def append_result(run_id: str, seed: int, status: str, primary: str) -> None:
    with RESULT_INDEX.open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerow(
            [
                run_id,
                "M2",
                "NAM_A2_packed",
                "synthetic_tanh",
                seed,
                status,
                primary,
                f"experiments/runs/{run_id}",
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    campaign = yaml.safe_load(CAMPAIGN_CONFIG.read_text(encoding="utf-8"))
    if args.seed not in campaign["seeds"]:
        raise ValueError(f"seed {args.seed} is not preregistered in {CAMPAIGN_CONFIG}")
    run_id = args.run_id or f"m2_a2_tanh_seed{args.seed}_v1"
    run_dir = ROOT / "experiments/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for directory in ("checkpoints", "predictions", "figures"):
        (run_dir / directory).mkdir()

    manifest_bytes = DATASET_MANIFEST.read_bytes()
    split_bytes = SPLIT_MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    split = json.loads(split_bytes)
    data, model, learning = make_configs(
        campaign, manifest, root=ROOT, model_config_path=MODEL_CONFIG
    )
    resolved = {
        "campaign": campaign,
        "data": data,
        "model": model,
        "learning": learning,
        "seed": args.seed,
    }
    config_bytes = json.dumps(resolved, sort_keys=True).encode()
    config_hash = sha256_bytes(config_bytes)
    data_hash = sha256_bytes(manifest_bytes, split_bytes)
    commit = git_commit()
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    command = (
        "uv run python scripts/campaigns/run_m2_a2.py"
        f" --seed {args.seed} --run-id {run_id}"
    )
    (run_dir / "config-resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (run_dir / "git-commit.txt").write_text(commit + "\n", encoding="utf-8")
    seed_everything(args.seed, campaign["deterministic_mode"])
    (run_dir / "environment.json").write_text(
        json.dumps(environment(), indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "dataset-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "split-manifest.json").write_text(
        json.dumps(split, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "seed.txt").write_text(f"{args.seed}\n", encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({"status": "running", "started_at": timestamp}, indent=2) + "\n",
        encoding="utf-8",
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    status = "failed"
    failure_reason = ""
    primary = ""
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_log,
            stderr_path.open("w", encoding="utf-8") as stderr_log,
            contextlib.redirect_stdout(Tee(sys.__stdout__, stdout_log)),
            contextlib.redirect_stderr(Tee(sys.__stderr__, stderr_log)),
        ):
            print(f"Starting {run_id} at commit {commit}")
            full.main(data, model, learning, run_dir, no_show=True, make_plots=False)
            history = extract_history(run_dir)
            (run_dir / "training-history.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )
            plot_history(history, run_dir)
            metrics = evaluate_exports(run_dir, manifest)
            (run_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
            checkpoint_index(run_dir)
            primary = f"full_esr={metrics['full']['esr']:.9g}"
            status = "completed"
            print(f"Completed {run_id}: {primary}")
    except Exception as error:
        failure_reason = f"{type(error).__name__}: {error}"
        with stderr_path.open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        raise
    finally:
        elapsed = time.perf_counter() - started
        if not (run_dir / "metrics.json").exists():
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "status": "unavailable",
                        "reason": failure_reason or "run did not reach evaluation",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        if not (run_dir / "checkpoints/index.json").exists():
            checkpoint_index(run_dir)
        timings = {
            "wall_seconds": elapsed,
            "scope": "training_and_evaluation",
            "completed": status == "completed",
            "gpu_peak_memory_bytes": (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
            ),
        }
        (run_dir / "timings.json").write_text(
            json.dumps(timings, indent=2) + "\n", encoding="utf-8"
        )
        finished = datetime.now().astimezone().isoformat(timespec="seconds")
        (run_dir / "status.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "started_at": timestamp,
                    "finished_at": finished,
                    "failure_reason": failure_reason,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        append_run(
            LEDGER,
            {
                "date": finished,
                "run_id": run_id,
                "phase": "M2",
                "model": "NAM_A2_packed",
                "device": "synthetic_tanh",
                "seed": args.seed,
                "commit": commit,
                "config_sha256": config_hash,
                "data_sha256": data_hash,
                "status": status,
                "failure_reason": failure_reason,
                "results_path": f"experiments/runs/{run_id}",
            },
        )
        append_result(run_id, args.seed, status, primary)


if __name__ == "__main__":
    main()
