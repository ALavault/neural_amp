#!/usr/bin/env python3
"""Audit A2 Python/C++ parity and real CPU inference cost."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import yaml
from nam.models import init_from_nam

from fssr_nam.reporting.ledger import append_run

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/evaluation/m2_cpu.yaml"
RUN_ID = "m2_a2_cpu_seed0_v1"
RUN_DIR = ROOT / "experiments/runs" / RUN_ID
LEDGER = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
RESULT_INDEX = ROOT / ".codex_campaign/RESULT_INDEX.csv"
BLOCK_RUNNER = ROOT / "build/fssr_cpp/nam_block_runner"
BENCHMARK = ROOT / "build/fssr_cpp/nam_benchmark"
OFFICIAL_RENDER = ROOT / "build/nam_core/tools/render"
OFFICIAL_BENCHMARK = ROOT / "build/nam_core/tools/bench_a2_fast"


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


def run_command(command: list[str], stdout_log, stderr_log) -> str:
    printable = " ".join(command)
    print(f"$ {printable}", flush=True)
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    stdout_log.write(f"$ {printable}\n{result.stdout}")
    stderr_log.write(f"$ {printable}\n{result.stderr}")
    stdout_log.flush()
    stderr_log.flush()
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit {result.returncode}: {printable}")
    return result.stdout


def read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def cpu_environment(core: int) -> dict[str, object]:
    lscpu = subprocess.run(
        ["lscpu", "--json"], check=True, capture_output=True, text=True
    ).stdout
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_affinity_before_pin": sorted(os.sched_getaffinity(0)),
        "selected_cpu_core": core,
        "threads": 1,
        "build_type": "Release",
        "compiler_flags": "-Ofast, interprocedural optimization",
        "core_host_interface": "double",
        "a2_fast_internal_precision": "float32",
        "scaling_governor": read_optional(
            Path(f"/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_governor")
        ),
        "intel_no_turbo": read_optional(
            Path("/sys/devices/system/cpu/intel_pstate/no_turbo")
        ),
        "lscpu": json.loads(lscpu),
    }


def load_python_prediction(model_path: Path, samples: np.ndarray) -> np.ndarray:
    model_dict = json.loads(model_path.read_text(encoding="utf-8"))
    model = init_from_nam(model_dict).eval()
    with torch.inference_mode():
        return model(torch.from_numpy(samples), pad_start=True).numpy()


def max_abs(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise RuntimeError(f"shape mismatch: {first.shape} != {second.shape}")
    return float(np.max(np.abs(first.astype(np.float64) - second.astype(np.float64))))


def run_block(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    pattern: list[int],
    core: int,
    stdout_log,
    stderr_log,
    reset_path: Path | None = None,
) -> np.ndarray:
    command = [
        "taskset",
        "-c",
        str(core),
        str(BLOCK_RUNNER),
        str(model_path),
        str(input_path),
        str(output_path),
        ",".join(str(value) for value in pattern),
    ]
    if reset_path is not None:
        command.append(str(reset_path))
    run_command(command, stdout_log, stderr_log)
    return np.fromfile(output_path, dtype=np.float32)


def parity_audit(config: dict, core: int, stdout_log, stderr_log) -> dict:
    source_run = ROOT / "experiments/runs" / config["training_run"]
    manifest = json.loads((source_run / "dataset-manifest.json").read_text())
    validation = next(
        item for item in manifest["files"] if item["name"] == "validation"
    )
    samples, sample_rate = sf.read(ROOT / validation["input_path"], dtype="float32")
    if sample_rate != config["sample_rate"]:
        raise RuntimeError("validation sample-rate mismatch")
    input_raw = RUN_DIR / "predictions/parity_input.f32"
    samples.tofile(input_raw)
    sf.write(
        RUN_DIR / "predictions/parity_input.wav",
        samples,
        sample_rate,
        subtype="FLOAT",
    )
    results = {}
    reference_outputs = {}
    for label in config["models"]:
        model_path = source_run / f"model_{label}.nam"
        python_output = load_python_prediction(model_path, samples)
        python_output.tofile(RUN_DIR / f"predictions/python_{label}.f32")
        regular_outputs = {}
        reset_difference = None
        for block_size in config["parity_patterns"]["regular"]:
            output_path = RUN_DIR / f"predictions/cpp_{label}_b{block_size}.f32"
            reset_path = (
                RUN_DIR / f"predictions/cpp_{label}_b{block_size}_reset.f32"
                if block_size == 64
                else None
            )
            output = run_block(
                model_path,
                input_raw,
                output_path,
                [block_size],
                core,
                stdout_log,
                stderr_log,
                reset_path,
            )
            regular_outputs[str(block_size)] = output
            if reset_path is not None:
                reset_difference = max_abs(
                    output, np.fromfile(reset_path, dtype=np.float32)
                )
        irregular_path = RUN_DIR / f"predictions/cpp_{label}_irregular.f32"
        irregular = run_block(
            model_path,
            input_raw,
            irregular_path,
            config["parity_patterns"]["irregular"],
            core,
            stdout_log,
            stderr_log,
        )
        reference = regular_outputs["64"]
        official_path = RUN_DIR / f"predictions/official_render_{label}.wav"
        run_command(
            [
                "taskset",
                "-c",
                str(core),
                str(OFFICIAL_RENDER),
                str(model_path),
                str(RUN_DIR / "predictions/parity_input.wav"),
                str(official_path),
            ],
            stdout_log,
            stderr_log,
        )
        official, official_rate = sf.read(official_path, dtype="float32")
        if official_rate != sample_rate:
            raise RuntimeError("official render sample-rate mismatch")
        results[label] = {
            "num_samples": len(samples),
            "python_cpp_max_abs": max_abs(python_output, reference),
            "regular_block_max_abs_vs_64": {
                size: max_abs(output, reference)
                for size, output in regular_outputs.items()
            },
            "irregular_block_max_abs_vs_64": max_abs(irregular, reference),
            "reset_max_abs": reset_difference,
            "official_render_max_abs_vs_runner_64": max_abs(official, reference),
        }
        reference_outputs[label] = (python_output, reference)
    tolerance = config["tolerances"]
    checks = {}
    for label, result in results.items():
        checks[f"{label}_python_cpp"] = (
            result["python_cpp_max_abs"] <= tolerance["python_cpp_max_abs"]
        )
        checks[f"{label}_regular_blocks"] = (
            max(result["regular_block_max_abs_vs_64"].values())
            <= tolerance["block_cpp_max_abs"]
        )
        checks[f"{label}_irregular_blocks"] = (
            result["irregular_block_max_abs_vs_64"] <= tolerance["block_cpp_max_abs"]
        )
        checks[f"{label}_reset"] = (
            result["reset_max_abs"] <= tolerance["reset_cpp_max_abs"]
        )
        checks[f"{label}_official_render"] = (
            result["official_render_max_abs_vs_runner_64"]
            <= tolerance["block_cpp_max_abs"]
        )
    if not all(checks.values()):
        raise RuntimeError(f"parity checks failed: {checks}")
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    count = 12_000
    for axis, label in zip(axes, config["models"], strict=True):
        python_output, cpp_output = reference_outputs[label]
        axis.plot((python_output - cpp_output)[:count], linewidth=0.6)
        axis.set_ylabel(f"{label} error")
    axes[-1].set_xlabel("sample")
    figure.tight_layout()
    figure.savefig(RUN_DIR / "figures/python_cpp_error.png", dpi=140)
    plt.close(figure)
    return {"results": results, "checks": checks, "tolerances": tolerance}


def benchmark_audit(config: dict, core: int, stdout_log, stderr_log) -> dict:
    source_run = ROOT / "experiments/runs" / config["training_run"]
    results = {}
    for label in config["models"]:
        model_path = source_run / f"model_{label}.nam"
        model_results = []
        for block_size in config["block_sizes"]:
            output = run_command(
                [
                    "taskset",
                    "-c",
                    str(core),
                    str(BENCHMARK),
                    str(model_path),
                    str(block_size),
                    str(config["seconds_per_iteration"]),
                    str(config["iterations"]),
                ],
                stdout_log,
                stderr_log,
            )
            model_results.append(json.loads(output))
        results[label] = model_results
    official_output = run_command(
        [
            "taskset",
            "-c",
            str(core),
            str(OFFICIAL_BENCHMARK),
            "--buffer",
            "64",
            "--seconds",
            str(config["seconds_per_iteration"]),
            "--iters",
            str(config["iterations"]),
            str(source_run / "model_lite.nam"),
            str(source_run / "model_full.nam"),
        ],
        stdout_log,
        stderr_log,
    )
    (RUN_DIR / "official-bench-a2-fast.txt").write_text(
        official_output, encoding="utf-8"
    )
    figure, axis = plt.subplots(figsize=(7, 4))
    for label, values in results.items():
        axis.plot(
            [item["block_size"] for item in values],
            [item["median_ns_per_sample"] for item in values],
            marker="o",
            label=f"{label} median",
        )
        axis.plot(
            [item["block_size"] for item in values],
            [item["p95_ns_per_sample"] for item in values],
            marker="x",
            linestyle="--",
            label=f"{label} p95",
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("block size")
    axis.set_ylabel("nanoseconds per sample")
    axis.legend()
    figure.tight_layout()
    figure.savefig(RUN_DIR / "figures/cpu_block_size.png", dpi=140)
    plt.close(figure)
    return {"custom": results, "official_fast_vs_generic_raw": official_output}


def main() -> None:
    config_bytes = CONFIG_PATH.read_bytes()
    config = yaml.safe_load(config_bytes)
    source_run = ROOT / "experiments/runs" / config["training_run"]
    manifest_bytes = (source_run / "dataset-manifest.json").read_bytes()
    split_bytes = (source_run / "split-manifest.json").read_bytes()
    for executable in (BLOCK_RUNNER, BENCHMARK, OFFICIAL_RENDER, OFFICIAL_BENCHMARK):
        if not executable.is_file():
            raise FileNotFoundError(executable)
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    for directory in ("checkpoints", "predictions", "figures"):
        (RUN_DIR / directory).mkdir()
    commit = git_commit()
    start_iso = datetime.now().astimezone().isoformat(timespec="seconds")
    core = min(os.sched_getaffinity(0))
    (RUN_DIR / "config-resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    command = "uv run python scripts/audit_m2_a2_inference.py"
    (RUN_DIR / "command.txt").write_text(command + "\n", encoding="utf-8")
    (RUN_DIR / "git-commit.txt").write_text(commit + "\n", encoding="utf-8")
    (RUN_DIR / "environment.json").write_text(
        json.dumps(cpu_environment(core), indent=2) + "\n", encoding="utf-8"
    )
    (RUN_DIR / "dataset-manifest.json").write_bytes(manifest_bytes)
    (RUN_DIR / "split-manifest.json").write_bytes(split_bytes)
    (RUN_DIR / "seed.txt").write_text("0\n", encoding="utf-8")
    (RUN_DIR / "status.json").write_text(
        json.dumps({"status": "running", "started_at": start_iso}, indent=2) + "\n",
        encoding="utf-8",
    )
    started = time.perf_counter()
    status = "failed"
    failure_reason = ""
    primary = ""
    try:
        with (
            (RUN_DIR / "stdout.log").open("w", encoding="utf-8") as stdout_log,
            (RUN_DIR / "stderr.log").open("w", encoding="utf-8") as stderr_log,
        ):
            parity = parity_audit(config, core, stdout_log, stderr_log)
            benchmark = benchmark_audit(config, core, stdout_log, stderr_log)
        metrics = {"parity": parity, "benchmark": benchmark}
        (RUN_DIR / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        full64 = next(
            item for item in benchmark["custom"]["full"] if item["block_size"] == 64
        )
        primary = f"full_b64_median_ns_per_sample={full64['median_ns_per_sample']:.6g}"
        status = "completed"
    except Exception as error:
        failure_reason = f"{type(error).__name__}: {error}"
        raise
    finally:
        elapsed = time.perf_counter() - started
        if not (RUN_DIR / "metrics.json").exists():
            (RUN_DIR / "metrics.json").write_text(
                json.dumps(
                    {"status": "unavailable", "reason": failure_reason}, indent=2
                )
                + "\n",
                encoding="utf-8",
            )
        (RUN_DIR / "timings.json").write_text(
            json.dumps(
                {"wall_seconds": elapsed, "completed": status == "completed"}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        finish_iso = datetime.now().astimezone().isoformat(timespec="seconds")
        (RUN_DIR / "status.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "started_at": start_iso,
                    "finished_at": finish_iso,
                    "failure_reason": failure_reason,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        config_hash = sha256_bytes(config_bytes)
        data_hash = sha256_bytes(manifest_bytes, split_bytes)
        append_run(
            LEDGER,
            {
                "date": finish_iso,
                "run_id": RUN_ID,
                "phase": "M2",
                "model": "NAM_A2_Full_Lite",
                "device": platform.processor() or "cpu",
                "seed": 0,
                "commit": commit,
                "config_sha256": config_hash,
                "data_sha256": data_hash,
                "status": status,
                "failure_reason": failure_reason,
                "results_path": f"experiments/runs/{RUN_ID}",
            },
        )
        with RESULT_INDEX.open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerow(
                [
                    RUN_ID,
                    "M2",
                    "NAM_A2_Full_Lite",
                    "cpu",
                    0,
                    status,
                    primary,
                    f"experiments/runs/{RUN_ID}",
                ]
            )


if __name__ == "__main__":
    main()
