"""Python CLI implementation for the prospective SOTA prototype."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml
from numpy.typing import ArrayLike, NDArray
from torch import nn

from fssr_nam.campaign.amp_sota_prototype_v1 import (
    AMENDMENT_PATH,
    CAMPAIGN_PATH,
    CAMPAIGN_VERSION,
    DEVELOPMENT_DEVICES,
    validate_repository_state,
    validate_stage_authorization,
)
from fssr_nam.campaign.amp_sota_registry import gate_decisions
from fssr_nam.campaign.quality_aa_provenance import (
    capture_provenance,
    digest_text,
    replace_json,
    strict_json,
    write_new_json,
)
from fssr_nam.metrics.spectral import log_mel_error, multi_resolution_stft_error
from fssr_nam.metrics.time import error_to_signal_ratio, mean_absolute_error
from fssr_nam.models.prototype import (
    PROTOTYPE_ACTIVATIONS,
    PROTOTYPE_RESAMPLERS,
    PROTOTYPE_SLOW_CONTROLS,
)
from fssr_nam.reporting.ledger import append_run, read_runs
from fssr_nam.training.prototype import (
    REGISTERED_SEEDS,
    TARGET_SAMPLE_RATE,
    DevelopmentSource,
    PrototypeDataBoundaryError,
    PrototypeModelConfig,
    build_checkpoint_payload,
    load_development_sources,
    load_prototype_checkpoint,
    train_development_model,
    write_checkpoint_once,
)

MANIFEST_PATH = Path("datasets/manifests/r1_physical.json")
SPLIT_MANIFEST_PATH = Path("datasets/splits/r1_physical.json")
GLOBAL_LEDGER_PATH = Path(".codex_campaign/RUN_LEDGER.jsonl")
GATE_LEDGER_PATH = CAMPAIGN_PATH / "GATE_LEDGER.jsonl"
RUN_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")
PROVENANCE_SOURCES = (
    "configs/amp_sota_prototype_v1/protocol.yaml",
    "configs/amp_sota_prototype_v1_1/protocol.yaml",
    ".codex_campaign/amp_sota_prototype_v1/PROTOCOL_LOCK.yaml",
    ".codex_campaign/amp_sota_prototype_v1_1/PROTOCOL_LOCK.yaml",
    "datasets/manifests/r1_physical.json",
    "datasets/splits/r1_physical.json",
    "src/fssr_nam/models/prototype.py",
    "src/fssr_nam/training/prototype.py",
    "src/fssr_nam/inference/prototype_cli.py",
    "scripts/run_amp_sota_prototype.py",
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _validate_run_id(run_id: str) -> None:
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run-id must contain lowercase letters, digits, '_' or '-'")


def _compute_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("prototype compute-device must be cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    return device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _as_mono_float32(signal: ArrayLike) -> NDArray[np.float32]:
    samples = np.asarray(signal, dtype=np.float32)
    if samples.ndim != 1 or samples.size < 1:
        raise ValueError("prototype audio must be nonempty and mono")
    if not np.all(np.isfinite(samples)):
        raise ValueError("prototype audio must contain only finite samples")
    return samples


def _reset_model(model: nn.Module) -> None:
    reset = getattr(model, "reset", None)
    if not callable(reset):
        reset = getattr(model, "reset_state", None)
    if not callable(reset):
        raise TypeError("prototype model must expose reset() or reset_state()")
    reset()


def render_streaming(
    model: nn.Module,
    signal: ArrayLike,
    *,
    compute_device: torch.device,
    block_size: int = 128,
) -> NDArray[np.float32]:
    """Render one source with exactly one reset before its first block."""
    samples = _as_mono_float32(signal)
    if block_size < 1:
        raise ValueError("render block size must be positive")
    stream = getattr(model, "stream", None)
    if not callable(stream):
        raise TypeError("prototype model must expose stream()")
    model.to(compute_device)
    model.eval()
    _reset_model(model)
    outputs: list[NDArray[np.float32]] = []
    with torch.inference_mode():
        for start in range(0, samples.size, block_size):
            block = torch.from_numpy(samples[start : start + block_size]).to(
                compute_device
            )[None]
            prediction = model.stream(block)
            if prediction.shape != block.shape or not torch.isfinite(prediction).all():
                raise RuntimeError("prototype render produced invalid audio")
            outputs.append(
                np.asarray(prediction[0].detach().cpu().numpy(), dtype=np.float32)
            )
    rendered = np.concatenate(outputs)
    if rendered.shape != samples.shape:
        raise RuntimeError("prototype render length changed")
    return rendered


def align_declared_latency(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    latency_samples: int,
    common_preroll_samples: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Apply only declared model latency, then the common scoring preroll."""
    predicted = _as_mono_float32(prediction)
    reference = _as_mono_float32(target)
    if predicted.shape != reference.shape:
        raise ValueError("prediction and target lengths differ")
    if latency_samples < 0 or common_preroll_samples < 0:
        raise ValueError("latency and preroll must be non-negative")
    scored = predicted.size - latency_samples - common_preroll_samples
    if scored < 4_096:
        raise ValueError("source is too short for frozen aligned metrics")
    prediction_start = latency_samples + common_preroll_samples
    target_start = common_preroll_samples
    prediction_aligned = predicted[prediction_start:]
    target_aligned = reference[target_start : target_start + scored]
    if prediction_aligned.shape != target_aligned.shape:
        raise RuntimeError("declared-latency alignment changed paired lengths")
    return prediction_aligned, target_aligned


def evaluate_development_sources(
    model: nn.Module,
    sources: tuple[DevelopmentSource, ...],
    *,
    compute_device: torch.device,
    block_size: int,
    common_preroll_samples: int,
) -> dict[str, Any]:
    """Evaluate source files independently; windows are never observations."""
    if not sources:
        raise ValueError("at least one development source is required")
    latency = getattr(model, "latency_samples", None)
    if isinstance(latency, bool) or not isinstance(latency, int) or latency < 0:
        raise TypeError("prototype model must expose integer latency_samples")
    rows: list[dict[str, Any]] = []
    for source in sources:
        prediction = render_streaming(
            model,
            source.input,
            compute_device=compute_device,
            block_size=block_size,
        )
        aligned_prediction, aligned_target = align_declared_latency(
            prediction,
            source.target,
            latency_samples=latency,
            common_preroll_samples=common_preroll_samples,
        )
        metrics = {
            "esr": error_to_signal_ratio(aligned_prediction, aligned_target),
            "mae": mean_absolute_error(aligned_prediction, aligned_target),
            "log_mel": log_mel_error(aligned_prediction, aligned_target),
            "mrstft": multi_resolution_stft_error(aligned_prediction, aligned_target),
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError("prototype evaluation produced non-finite metrics")
        rows.append(
            {
                "physical_device": source.physical_device,
                "split": source.split,
                "source_id": source.source_id,
                "samples_scored": int(aligned_prediction.size),
                "metrics": metrics,
            }
        )
    medians = {
        name: float(np.median([row["metrics"][name] for row in rows]))
        for name in ("esr", "mae", "log_mel", "mrstft")
    }
    return {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "evidence_tier": "INTERNAL_DEV",
        "observation_unit": ["device", "seed", "source_file"],
        "windows_are_independent_observations": False,
        "alignment": "declared_model_latency_only",
        "declared_latency_samples": latency,
        "common_preroll_samples": common_preroll_samples,
        "source_rows": rows,
        "median_source_metrics": medians,
        "sealed_outputs_accessed": {"blackstar": False, "ua1176": False},
    }


def render_wav_once(
    model: nn.Module,
    input_path: Path,
    output_path: Path,
    *,
    compute_device: torch.device,
    block_size: int = 128,
) -> None:
    """Render one 48 kHz mono WAV without replacing an existing output."""
    signal, sample_rate = sf.read(input_path, dtype="float32")
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError("render input must be 48 kHz")
    rendered = render_streaming(
        model,
        signal,
        compute_device=compute_device,
        block_size=block_size,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        sf.write(stream, rendered, TARGET_SAMPLE_RATE, subtype="FLOAT", format="WAV")


def benchmark_python_diagnostic(
    model: nn.Module,
    *,
    compute_device: torch.device,
    block_size: int = 128,
    repetitions: int = 30,
    audio_seconds_per_repetition: float = 1.0,
    warmup_blocks: int = 16,
    seed: int = 20_260_830,
) -> dict[str, Any]:
    """Measure Python streaming cost; this result is never gate-eligible."""
    if block_size < 1 or repetitions < 1 or warmup_blocks < 0:
        raise ValueError("benchmark dimensions must be positive")
    if audio_seconds_per_repetition <= 0.0:
        raise ValueError("benchmark audio duration must be positive")
    timed_blocks = math.ceil(
        audio_seconds_per_repetition * TARGET_SAMPLE_RATE / block_size
    )
    audio_duration = timed_blocks * block_size / TARGET_SAMPLE_RATE
    generator = torch.Generator(device="cpu").manual_seed(seed)
    block = 0.1 * torch.randn((1, block_size), generator=generator)
    block = block.to(compute_device)
    model.to(compute_device)
    model.eval()
    rtfs: list[float] = []
    with torch.inference_mode():
        for _ in range(repetitions):
            _reset_model(model)
            for _ in range(warmup_blocks):
                output = model.stream(block)
                if output.shape != block.shape:
                    raise RuntimeError("prototype benchmark output shape changed")
            _synchronize(compute_device)
            started = time.perf_counter_ns()
            for _ in range(timed_blocks):
                output = model.stream(block)
            _synchronize(compute_device)
            elapsed_seconds = (time.perf_counter_ns() - started) / 1.0e9
            if not torch.isfinite(output).all():
                raise RuntimeError("prototype benchmark produced non-finite audio")
            rtfs.append(elapsed_seconds / audio_duration)
    return {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "kind": "python_streaming_diagnostic",
        "gate_eligible": False,
        "scientific_eligible": False,
        "gate_decision": None,
        "native_required_for_gate": True,
        "non_eligibility_reason": "python_diagnostic_not_common_cpp_gate_engine",
        "rtf_definition": "compute_time_over_audio_duration",
        "sample_rate_hz": TARGET_SAMPLE_RATE,
        "block_size": block_size,
        "repetitions": repetitions,
        "timed_blocks_per_repetition": timed_blocks,
        "audio_seconds_per_repetition": audio_duration,
        "warmup_blocks": warmup_blocks,
        "compute_device": str(compute_device),
        "torch_threads": torch.get_num_threads(),
        "median_rtf": float(np.median(rtfs)),
        "p95_rtf": float(np.percentile(rtfs, 95)),
        "rtf_by_repetition": rtfs,
    }


def _write_json_once(path: Path, payload: Any) -> None:
    write_new_json(path, payload)


def _manifest_data_digest(root: Path, physical_device: str) -> str:
    manifest = json.loads((root / MANIFEST_PATH).read_text(encoding="utf-8"))
    rows = []
    for row in manifest.get("files", []):
        if row.get("device") == physical_device and row.get("split") == "train":
            rows.append(
                {
                    "source_id": row.get("source_id"),
                    "input_sha256": row.get("input_sha256"),
                    "target_sha256": row.get("target_sha256"),
                }
            )
    if not rows or any(None in row.values() for row in rows):
        raise PrototypeDataBoundaryError("development data identity is incomplete")
    return digest_text(strict_json({"device": physical_device, "sources": rows}))


def _ledger_entry(
    *,
    run_id: str,
    phase: str,
    model: str,
    device: str,
    seed: int,
    status: str,
    failure_reason: str,
    run_dir: Path,
    root: Path,
    git_head: str,
    config_digest: str,
    data_digest: str,
) -> dict[str, Any]:
    return {
        "date": _now(),
        "run_id": run_id,
        "phase": phase,
        "model": model,
        "device": device,
        "seed": seed,
        "commit": git_head,
        "config_sha256": config_digest,
        "data_sha256": data_digest,
        "status": status,
        "failure_reason": failure_reason,
        "results_path": str(run_dir.relative_to(root)),
    }


def _create_run(
    root: Path, run_id: str, command: list[str]
) -> tuple[Path, dict[str, Any], str]:
    _validate_run_id(run_id)
    ledger_path = root / GLOBAL_LEDGER_PATH
    if any(row.get("run_id") == run_id for row in read_runs(ledger_path)):
        raise RuntimeError("run-id already exists in the global ledger")
    run_dir = root / "experiments/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    provenance = capture_provenance(
        root,
        run_dir,
        command,
        source_files=PROVENANCE_SOURCES,
    )
    started_at = _now()
    write_new_json(
        run_dir / "status.json",
        {"status": "running", "started_at": started_at, "finished_at": None},
    )
    return run_dir, provenance, started_at


def _development_authorized(root: Path) -> dict[str, Any]:
    protocol = validate_repository_state(root, require_frozen=True)
    decisions = gate_decisions(root / GATE_LEDGER_PATH)
    validate_stage_authorization("development", decisions)
    amendment = yaml.safe_load((root / AMENDMENT_PATH).read_text(encoding="utf-8"))
    if not isinstance(amendment, dict):
        raise RuntimeError("v1.1 protocol amendment must be a mapping")
    return {"protocol": protocol, "amendment": amendment}


def _train(args: argparse.Namespace, root: Path, command: list[str]) -> int:
    declarations = _development_authorized(root)
    training = declarations["amendment"]["training"]
    if args.seed not in training["seeds"]:
        raise ValueError("seed is outside the frozen v1.1 schedule")
    model_config = PrototypeModelConfig(
        activation=args.activation,
        resampler=args.resampler,
        slow_control=args.slow_control,
        profile=training["profile"],
        initial_residual_scale=0.5,
    )
    resolved = {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "operation": "train",
        "run_id": args.run_id,
        "physical_device": args.physical_device,
        "seed": args.seed,
        "compute_device": args.compute_device,
        "model_config": asdict(model_config),
        "training": training,
    }
    config_digest = digest_text(strict_json(resolved))
    data_digest = _manifest_data_digest(root, args.physical_device)
    run_dir, provenance, started_at = _create_run(root, args.run_id, command)
    write_new_json(run_dir / "config-resolved.json", resolved)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_paths: list[str] = []
    status = "failed"
    failure_reason = ""
    result_payload: dict[str, Any] | None = None
    try:
        compute_device = _compute_device(args.compute_device)
        torch.manual_seed(args.seed)
        if compute_device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        torch.use_deterministic_algorithms(True)
        sources = load_development_sources(
            root,
            root / MANIFEST_PATH,
            root / SPLIT_MANIFEST_PATH,
            physical_device=args.physical_device,
            split="train",
        )
        model = model_config.build().to(compute_device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )

        def save_checkpoint(update: int, current_model: nn.Module) -> None:
            path = checkpoint_dir / f"update_{update:08d}.pt"
            payload = build_checkpoint_payload(
                current_model,
                model_config,
                physical_device=args.physical_device,
                seed=args.seed,
                update=update,
                source_ids=[source.source_id for source in sources],
            )
            write_checkpoint_once(path, payload)
            checkpoint_paths.append(str(path.relative_to(root)))

        result = train_development_model(
            model,
            sources,
            optimizer=optimizer,
            compute_device=compute_device,
            seed=args.seed,
            updates=int(training["selection_checkpoint"]),
            checkpoint_updates=tuple(training["checkpoint_updates"]),
            chunk_samples=int(training["training_chunk_samples"]),
            common_preroll_samples=int(training["common_preroll_samples"]),
            projection_gain_weight=float(training["projection_gain_weight"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            checkpoint_callback=save_checkpoint,
        )
        result_payload = {
            "schema_version": 1,
            "campaign_version": CAMPAIGN_VERSION,
            "status": "completed",
            "evidence_tier": "INTERNAL_DEV",
            "training": asdict(result),
            "checkpoints": checkpoint_paths,
            "sealed_outputs_accessed": {"blackstar": False, "ua1176": False},
        }
        write_new_json(run_dir / "result.json", result_payload)
        status = "completed"
    except Exception as error:
        failure_reason = str(error)
        raise
    finally:
        replace_json(
            run_dir / "status.json",
            {
                "status": status,
                "started_at": started_at,
                "finished_at": _now(),
                "failure_reason": failure_reason,
            },
        )
        append_run(
            root / GLOBAL_LEDGER_PATH,
            _ledger_entry(
                run_id=args.run_id,
                phase="AMP-SOTA-PROTOTYPE-v1.1-DEVELOPMENT",
                model="slow_long_tcn_x2",
                device=args.physical_device,
                seed=args.seed,
                status=status,
                failure_reason=failure_reason,
                run_dir=run_dir,
                root=root,
                git_head=provenance["git_head"],
                config_digest=config_digest,
                data_digest=data_digest,
            ),
        )
    if result_payload is None:
        raise RuntimeError("prototype training ended without a result")
    return 0


def _evaluate(args: argparse.Namespace, root: Path) -> int:
    declarations = _development_authorized(root)
    training = declarations["amendment"]["training"]
    compute_device = _compute_device(args.compute_device)
    loaded = load_prototype_checkpoint(args.checkpoint, map_location=compute_device)
    if loaded.training["physical_device"] != args.physical_device:
        raise PrototypeDataBoundaryError(
            "checkpoint and evaluation physical devices differ"
        )
    sources = load_development_sources(
        root,
        root / MANIFEST_PATH,
        root / SPLIT_MANIFEST_PATH,
        physical_device=args.physical_device,
        split=args.split,
    )
    result = evaluate_development_sources(
        loaded.model,
        sources,
        compute_device=compute_device,
        block_size=128,
        common_preroll_samples=int(training["common_preroll_samples"]),
    )
    result["checkpoint"] = str(args.checkpoint)
    result["checkpoint_seed"] = loaded.training["seed"]
    _write_json_once(args.output, result)
    return 0


def _render(args: argparse.Namespace) -> int:
    compute_device = _compute_device(args.compute_device)
    loaded = load_prototype_checkpoint(args.checkpoint, map_location=compute_device)
    render_wav_once(
        loaded.model,
        args.input,
        args.output,
        compute_device=compute_device,
        block_size=args.block_size,
    )
    return 0


def _benchmark(args: argparse.Namespace, root: Path, command: list[str]) -> int:
    validate_repository_state(root, require_frozen=True)
    run_dir, provenance, started_at = _create_run(root, args.run_id, command)
    status = "failed"
    failure_reason = ""
    config_digest = ""
    data_digest = ""
    checkpoint_seed = 0
    try:
        compute_device = _compute_device(args.compute_device)
        if compute_device.type != "cpu":
            raise ValueError("Python diagnostic benchmark is CPU-only")
        torch.set_num_threads(1)
        loaded = load_prototype_checkpoint(args.checkpoint, map_location=compute_device)
        checkpoint_seed = int(loaded.training["seed"])
        resolved = {
            "schema_version": 1,
            "campaign_version": CAMPAIGN_VERSION,
            "operation": "benchmark_python_diagnostic",
            "run_id": args.run_id,
            "checkpoint": str(args.checkpoint),
            "compute_device": args.compute_device,
            "block_size": 128,
            "repetitions": 30,
            "audio_seconds_per_repetition": 1,
        }
        config_digest = digest_text(strict_json(resolved))
        data_digest = digest_text(
            strict_json(
                {
                    "model_config": asdict(loaded.model_config),
                    "training": loaded.training,
                }
            )
        )
        write_new_json(run_dir / "config-resolved.json", resolved)
        result = benchmark_python_diagnostic(
            loaded.model,
            compute_device=compute_device,
            block_size=128,
            repetitions=30,
            audio_seconds_per_repetition=1.0,
        )
        result["checkpoint"] = str(args.checkpoint)
        write_new_json(run_dir / "result.json", result)
        status = "completed"
    except Exception as error:
        failure_reason = str(error)
        if not config_digest:
            config_digest = digest_text(
                strict_json({"operation": "benchmark_python_diagnostic"})
            )
        if not data_digest:
            data_digest = digest_text(strict_json({"checkpoint": str(args.checkpoint)}))
        raise
    finally:
        replace_json(
            run_dir / "status.json",
            {
                "status": status,
                "started_at": started_at,
                "finished_at": _now(),
                "failure_reason": failure_reason,
            },
        )
        append_run(
            root / GLOBAL_LEDGER_PATH,
            _ledger_entry(
                run_id=args.run_id,
                phase="AMP-SOTA-PROTOTYPE-v1.1-PYTHON-DIAGNOSTIC",
                model="slow_long_tcn_x2",
                device=args.compute_device,
                seed=checkpoint_seed,
                status=status,
                failure_reason=failure_reason,
                run_dir=run_dir,
                root=root,
                git_head=provenance["git_head"],
                config_digest=config_digest,
                data_digest=data_digest,
            ),
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict Python interface for AMP-SOTA-PROTOTYPE-v1.1"
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--run-id", required=True)
    train.add_argument("--physical-device", required=True, choices=DEVELOPMENT_DEVICES)
    train.add_argument("--seed", required=True, type=int, choices=REGISTERED_SEEDS)
    train.add_argument("--activation", default="tanh", choices=PROTOTYPE_ACTIVATIONS)
    train.add_argument(
        "--resampler", default="kaiser_windowed_sinc", choices=PROTOTYPE_RESAMPLERS
    )
    train.add_argument(
        "--slow-control",
        default="causal_zero_order_hold",
        choices=PROTOTYPE_SLOW_CONTROLS,
    )
    train.add_argument("--compute-device", default="cuda")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    evaluate.add_argument(
        "--physical-device", required=True, choices=DEVELOPMENT_DEVICES
    )
    evaluate.add_argument("--split", required=True, choices=("validation", "test"))
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--compute-device", default="cuda")

    render = subparsers.add_parser("render")
    render.add_argument("--checkpoint", required=True, type=Path)
    render.add_argument("--input", required=True, type=Path)
    render.add_argument("--output", required=True, type=Path)
    render.add_argument("--block-size", default=128, type=int)
    render.add_argument("--compute-device", default="cpu")

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--run-id", required=True)
    benchmark.add_argument("--checkpoint", required=True, type=Path)
    benchmark.add_argument("--compute-device", default="cpu", choices=("cpu",))
    return parser


def main(root: Path, argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(arguments)
    command = [
        "uv",
        "run",
        "python",
        "scripts/run_amp_sota_prototype.py",
        *arguments,
    ]
    if args.operation == "train":
        return _train(args, root, command)
    if args.operation == "evaluate":
        return _evaluate(args, root)
    if args.operation == "render":
        return _render(args)
    if args.operation == "benchmark":
        return _benchmark(args, root, command)
    raise AssertionError("unreachable prototype operation")
