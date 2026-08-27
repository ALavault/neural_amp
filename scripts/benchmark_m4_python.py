#!/usr/bin/env python3
"""Benchmark trained M4 alternatives in their Python streaming paths."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import platform
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

from fssr_nam.reporting.ledger import append_run
from fssr_nam.training.m4 import model_factory

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/training/m4_smoke.yaml"
MODEL_CONFIG_PATH = ROOT / "configs/training/m3_synthetic.yaml"
MANIFEST_PATH = ROOT / "datasets/manifests/m4_internal.json"
SPLIT_PATH = ROOT / "datasets/splits/m4_internal.json"
M2_CPU_PATH = ROOT / "experiments/runs/m2_a2_cpu_seed0_v2/metrics.json"
LEDGER = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
RESULT_INDEX = ROOT / ".codex_campaign/RESULT_INDEX.csv"
CHECKPOINT_RUNS = {
    "B2": "m4_fulltone_b2_seed0_v1",
    "S3": "m4_fulltone_s3_seed0_v1",
    "S4": "m4_fulltone_s4_seed0_v1",
}


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
    parser.add_argument("--run-id", required=True)
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


def benchmark_model(
    model, signal: torch.Tensor, block_size: int
) -> dict[str, float | int]:
    timed_blocks = {1: 2000, 16: 1500, 64: 1000, 128: 1000}[block_size]
    warmup_blocks = min(100, timed_blocks // 4)
    required = (warmup_blocks + timed_blocks) * block_size
    if required > signal.numel():
        raise RuntimeError("benchmark signal is too short")
    model.reset_state()
    position = 0
    with torch.inference_mode():
        for _ in range(warmup_blocks):
            model.stream(signal[position : position + block_size])
            position += block_size
        timings = []
        for _ in range(timed_blocks):
            started = time.perf_counter_ns()
            model.stream(signal[position : position + block_size])
            timings.append(time.perf_counter_ns() - started)
            position += block_size
    values = np.asarray(timings, dtype=np.float64)
    median_block = float(np.median(values))
    p95_block = float(np.percentile(values, 95))
    return {
        "block_size": block_size,
        "warmup_blocks": warmup_blocks,
        "timed_blocks": timed_blocks,
        "median_block_ns": median_block,
        "p95_block_ns": p95_block,
        "median_ns_per_sample": median_block / block_size,
        "p95_ns_per_sample": p95_block / block_size,
        "median_realtime_factor": 1.0e9 / (48_000 * median_block / block_size),
        "p95_realtime_factor": 1.0e9 / (48_000 * p95_block / block_size),
    }


def runtime_state_bytes(model) -> int:
    """Count materialized streaming tensors, excluding fixed coefficients."""
    names = ("_state", "_stream_state", "_hidden", "_accumulator")
    tensors = []
    seen = set()
    integer_fields = 0
    for module in model.modules():
        for name in names:
            value = getattr(module, name, None)
            if (
                isinstance(value, torch.Tensor)
                and value.numel()
                and id(value) not in seen
            ):
                seen.add(id(value))
                tensors.append(value)
        if hasattr(module, "_count") and isinstance(module._count, int):
            integer_fields += 1
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors) + (
        8 * integer_fields
    )


def append_result_index(entry: list[object]) -> None:
    with RESULT_INDEX.open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerow(entry)


def main() -> None:
    args = parse_args()
    run_dir = ROOT / "experiments/runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for directory in ("checkpoints", "predictions", "figures"):
        (run_dir / directory).mkdir()
    config_bytes = CONFIG_PATH.read_bytes()
    manifest_bytes = MANIFEST_PATH.read_bytes()
    split_bytes = SPLIT_PATH.read_bytes()
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))[
        "model"
    ]
    resolved = {
        "schema_version": 1,
        "campaign_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "models": CHECKPOINT_RUNS,
        "block_sizes": [1, 16, 64, 128],
        "threads": 1,
        "affinity": [0],
        "precision": "float32",
        "sample_rate": 48_000,
        "timing_scope": "model.stream only",
    }
    resolved_bytes = yaml.safe_dump(resolved, sort_keys=False).encode()
    config_hash = hashlib.sha256(resolved_bytes).hexdigest()
    data_hash = hashlib.sha256(manifest_bytes + split_bytes).hexdigest()
    commit = git_commit()
    started_iso = datetime.now().astimezone().isoformat(timespec="seconds")
    command = f"uv run python scripts/benchmark_m4_python.py --run-id {args.run_id}"
    (run_dir / "config-resolved.yaml").write_bytes(resolved_bytes)
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (run_dir / "git-commit.txt").write_text(commit + "\n", encoding="utf-8")
    (run_dir / "dataset-manifest.json").write_bytes(manifest_bytes)
    (run_dir / "split-manifest.json").write_bytes(split_bytes)
    (run_dir / "seed.txt").write_text("0\n", encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({"status": "running", "started_at": started_iso}, indent=2) + "\n",
        encoding="utf-8",
    )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    affinity_applied = False
    try:
        os.sched_setaffinity(0, {0})
        affinity_applied = True
    except OSError:
        pass
    (run_dir / "environment.json").write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "cpu": platform.processor(),
                "threads": torch.get_num_threads(),
                "affinity": sorted(os.sched_getaffinity(0)),
                "affinity_applied": affinity_applied,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    input_path = ROOT / "datasets/raw/internal_m4/fulltone_full_drive_2/test_input.wav"
    signal, sample_rate = sf.read(input_path, dtype="float32")
    if sample_rate != 48_000:
        raise RuntimeError("unexpected benchmark sample rate")
    signal_tensor = torch.from_numpy(signal)
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
            results = {}
            for code, source_run in CHECKPOINT_RUNS.items():
                model = model_factory(code, root=ROOT, model_config=model_config).eval()
                checkpoint = (
                    ROOT
                    / "experiments/runs"
                    / source_run
                    / "checkpoints/model-state.pt"
                )
                model.load_state_dict(
                    torch.load(checkpoint, map_location="cpu", weights_only=True)
                )
                results[code] = {
                    "source_run": source_run,
                    "checkpoint_sha256": sha256(checkpoint),
                    "parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "blocks": [
                        benchmark_model(model, signal_tensor, block_size)
                        for block_size in resolved["block_sizes"]
                    ],
                }
                results[code]["state_bytes"] = runtime_state_bytes(model)
                block64 = next(
                    item for item in results[code]["blocks"] if item["block_size"] == 64
                )
                print(
                    f"{code} block64 median={block64['median_ns_per_sample']:.3f} "
                    f"ns/sample"
                )
            a2_metrics = json.loads(M2_CPU_PATH.read_text(encoding="utf-8"))
            a2_reference = a2_metrics["benchmark"]["custom"]["full"]
            metrics = {
                "python_streaming": results,
                "official_a2_core_cpp_reference": a2_reference,
                "comparability": {
                    "python_alternatives_are_mutually_comparable": True,
                    "python_vs_cpp_is_final_h2_evidence": False,
                    "reason": (
                        "FSSR C++ inference is not implemented at M4; cross-engine "
                        "timings are diagnostic only."
                    ),
                },
            }
            (run_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
            figure, axis = plt.subplots(figsize=(7, 4))
            for code, result in results.items():
                axis.plot(
                    [item["block_size"] for item in result["blocks"]],
                    [item["median_ns_per_sample"] for item in result["blocks"]],
                    marker="o",
                    label=f"{code} Python",
                )
            axis.plot(
                [item["block_size"] for item in a2_reference],
                [item["median_ns_per_sample"] for item in a2_reference],
                marker="x",
                linestyle="--",
                label="B0 official C++ reference",
            )
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_xlabel("block size")
            axis.set_ylabel("median ns/sample")
            axis.legend()
            figure.tight_layout()
            figure.savefig(run_dir / "figures/cpu_by_block.png", dpi=140)
            plt.close(figure)
            s3_block64 = next(
                item for item in results["S3"]["blocks"] if item["block_size"] == 64
            )
            primary = f"s3_block64_ns={s3_block64['median_ns_per_sample']:.9g}"
            status = "completed"
    except Exception as error:
        failure_reason = f"{type(error).__name__}: {error}"
        with (run_dir / "stderr.log").open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        raise
    finally:
        elapsed = time.perf_counter() - started
        if not (run_dir / "metrics.json").exists():
            (run_dir / "metrics.json").write_text(
                json.dumps({"status": "unavailable", "reason": failure_reason}) + "\n",
                encoding="utf-8",
            )
        (run_dir / "timings.json").write_text(
            json.dumps({"wall_seconds": elapsed}, indent=2) + "\n", encoding="utf-8"
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
                "phase": "M4",
                "model": "B2_S3_S4_python_benchmark",
                "device": "host_cpu",
                "seed": 0,
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
                "M4",
                "B2_S3_S4_python_benchmark",
                "host_cpu",
                0,
                status,
                primary,
                relative_path,
            ]
        )


if __name__ == "__main__":
    main()
