#!/usr/bin/env python3
"""Execute the immutable, conditional Wright LSTM-64 competence gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml

from fssr_nam.campaign.r1 import R1Executor, RunSpec
from fssr_nam.losses import WrightLoss
from fssr_nam.metrics.time import time_metrics
from fssr_nam.models import WrightLSTM
from fssr_nam.reporting.ledger import read_runs
from fssr_nam.reporting.r1_preflight import validate_lock_digest
from fssr_nam.training.r1_competence import (
    CompetenceGateDecision,
    evaluate_competence_gate,
    seed_zero_allows_additional_runs,
    validate_competence_protocol,
)
from fssr_nam.training.wright import (
    evaluate_prediction,
    frame_audio,
    predict_streaming,
    train_epoch,
)

ROOT = Path(__file__).resolve().parents[1]
TRAINING_CONFIG_PATH = ROOT / "configs/training/r1_competence.yaml"
DATA_CONFIG_PATH = ROOT / "configs/data/r1_wright_bigmuff_native.yaml"
MODEL_CONFIG_PATH = ROOT / "configs/models/r1/wright_lstm64.yaml"
MANIFEST_PATH = ROOT / "datasets/manifests/r1_wright_bigmuff_native.json"
SPLIT_PATH = ROOT / "datasets/splits/r1_wright_bigmuff_native.json"
LEDGER_PATH = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
SUMMARY_PATH = ROOT / "experiments/summaries/r1_competence_gate.json"
GATES_PATH = ROOT / ".codex_campaign/r1/GATES.json"
LOCK_PATH = ROOT / ".codex_campaign/r1/DIAGNOSTIC_LOCK.yaml"
LOCK_DIGEST_PATH = ROOT / ".codex_campaign/r1/DIAGNOSTIC_LOCK.sha256"
ACTIVE_LOCK_PATH = ROOT / ".codex_campaign/r1/DIAGNOSTIC_LOCK_ACTIVE"

EVALUATION_CHUNK_SAMPLES = 100_000
PARITY_CHUNK_SAMPLES = 4_093
PARITY_PREFIX_SAMPLES = 262_144
MINIMUM_GPU_MEMORY_BYTES = 23_000_000_000
SOURCE_PATHS = (
    TRAINING_CONFIG_PATH,
    DATA_CONFIG_PATH,
    MODEL_CONFIG_PATH,
    MANIFEST_PATH,
    SPLIT_PATH,
)


class TerminationRequested(RuntimeError):
    """A counted trajectory received a catchable process termination request."""


def _raise_termination(signum: int, _frame: object) -> None:
    raise TerminationRequested(f"received signal {signum}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    preflight = parser.add_mutually_exclusive_group()
    preflight.add_argument(
        "--preflight",
        action="store_true",
        help="run one non-scientific synthetic optimizer update (legacy alias)",
    )
    preflight.add_argument(
        "--preflight-steps",
        type=int,
        metavar="N",
        help="run N non-scientific synthetic optimizer updates without writing state",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_digest(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        name = str(path.relative_to(ROOT)).encode("utf-8")
        payload = path.read_bytes()
        for value in (name, payload):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _validate_data_declarations(
    data_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> None:
    expected_header = {
        "schema_version": 1,
        "campaign_version": "FSSR-R1-v1",
        "name": "r1_wright_bigmuff_native",
        "tier": "INTERNAL_DEV",
        "sample_rate": 44_100,
        "normalization": "none",
        "transformation": "none",
    }
    for field, expected in expected_header.items():
        if data_config.get(field) != expected:
            raise ValueError(
                f"data configuration {field} must be {expected!r}, "
                f"got {data_config.get(field)!r}"
            )
    if manifest.get("name") != data_config["name"]:
        raise ValueError("Wright manifest names the wrong dataset")
    if manifest.get("campaign_version") != data_config["campaign_version"]:
        raise ValueError("Wright manifest campaign version mismatch")
    if manifest.get("tier") != "INTERNAL_DEV":
        raise ValueError("Wright competence data must remain INTERNAL_DEV")
    if manifest.get("configuration") != str(DATA_CONFIG_PATH.relative_to(ROOT)):
        raise ValueError("Wright manifest points to the wrong data configuration")
    if manifest.get("configuration_sha256") != _sha256(DATA_CONFIG_PATH):
        raise ValueError("Wright manifest does not bind the current data configuration")
    if split_manifest.get("dataset") != data_config["name"]:
        raise ValueError("Wright split manifest names the wrong dataset")
    if split_manifest.get("campaign_version") != data_config["campaign_version"]:
        raise ValueError("Wright split manifest campaign version mismatch")
    if split_manifest.get("path_overlap") != []:
        raise ValueError("Wright split manifest reports path overlap")
    if split_manifest.get("leakage_check") != "passed_at_published_file_level":
        raise ValueError("Wright split leakage check has not passed")

    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("Wright manifest files must be a list")
    by_split = {entry.get("split"): entry for entry in files if isinstance(entry, dict)}
    expected_splits = {"train", "validation", "test"}
    if len(files) != 3 or set(by_split) != expected_splits:
        raise ValueError(
            "Wright manifest must contain train/validation/test exactly once"
        )
    declared_splits = data_config.get("splits")
    if not isinstance(declared_splits, dict) or set(declared_splits) != expected_splits:
        raise ValueError("Wright data configuration has an invalid split declaration")
    groups = split_manifest.get("groups")
    if not isinstance(groups, dict) or set(groups) != expected_splits:
        raise ValueError("Wright split groups must contain train/validation/test")

    seen_groups: set[str] = set()
    for split_name in ("train", "validation", "test"):
        declaration = declared_splits[split_name]
        entry = by_split[split_name]
        for field in (
            "source_group",
            "input_path",
            "target_path",
            "input_sha256",
            "target_sha256",
        ):
            if entry.get(field) != declaration.get(field):
                raise ValueError(f"Wright {split_name} {field} provenance mismatch")
        if entry.get("sample_rate") != 44_100 or entry.get("tier") != "INTERNAL_DEV":
            raise ValueError(f"Wright {split_name} rate or tier mismatch")
        group = declaration["source_group"]
        if groups.get(split_name) != [group] or group in seen_groups:
            raise ValueError("Wright source groups are not disjoint by complete file")
        seen_groups.add(group)

    reference = data_config.get("reference_model")
    manifest_reference = manifest.get("reference_model")
    if not isinstance(reference, dict) or manifest_reference != reference:
        raise ValueError("Wright released-model provenance mismatch")
    reference_path = ROOT / reference["path"]
    if _sha256(reference_path) != reference["sha256"]:
        raise ValueError("Wright released-model checksum mismatch")


def _validate_model_declaration(
    model_config: Mapping[str, Any], data_config: Mapping[str, Any]
) -> None:
    expected = {
        "schema_version": 1,
        "campaign_version": "FSSR-R1-v1",
        "model": "wright_lstm",
        "sample_rate": 44_100,
        "input_channels": 1,
        "output_channels": 1,
        "unit_type": "LSTM",
        "layers": 1,
        "hidden_size": 64,
        "direct_skip": True,
        "head": "linear",
        "latency_samples": 0,
        "reference_model": data_config["reference_model"]["path"],
        "reference_commit": data_config["source_commit"],
    }
    for field, value in expected.items():
        if model_config.get(field) != value:
            raise ValueError(
                f"Wright model declaration {field} must be {value!r}, "
                f"got {model_config.get(field)!r}"
            )


def _load_declarations() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    for path in SOURCE_PATHS:
        if not path.is_file():
            raise FileNotFoundError(
                f"required Wright competence input is absent: {path}"
            )
    training_config = _load_yaml(TRAINING_CONFIG_PATH)
    data_config = _load_yaml(DATA_CONFIG_PATH)
    model_config = _load_yaml(MODEL_CONFIG_PATH)
    manifest = _load_json(MANIFEST_PATH)
    split_manifest = _load_json(SPLIT_PATH)
    validate_competence_protocol(training_config)
    _validate_data_declarations(data_config, manifest, split_manifest)
    _validate_model_declaration(model_config, data_config)
    return training_config, data_config, manifest, split_manifest


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_worktree() -> None:
    ignored = (
        "experiments/runs/**",
        "experiments/summaries/r1_competence_gate.json",
        ".codex_campaign/RUN_LEDGER.jsonl",
    )
    command = ["git", "status", "--porcelain", "--untracked-files=all", "--", "."]
    command.extend(f":(exclude){path}" for path in ignored)
    status = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("counted R1 competence runs require a clean committed tree")


def _seed_everything(seed: int) -> torch.Generator:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return torch.Generator(device="cpu").manual_seed(seed)


def _wright_loss(config: Mapping[str, Any]) -> WrightLoss:
    loss = config["loss"]
    return WrightLoss(
        preemphasis=-float(loss["preemphasis"][0]),
        esr_weight=float(loss["esr_preemphasized_weight"]),
        dc_weight=float(loss["dc_weight"]),
        epsilon=float(loss["epsilon"]),
    )


def _optimizer(model: WrightLSTM, config: Mapping[str, Any]) -> torch.optim.Adam:
    declaration = config["optimizer"]
    return torch.optim.Adam(
        model.parameters(),
        lr=float(declaration["learning_rate"]),
        weight_decay=float(declaration["weight_decay"]),
    )


def _scheduler(
    optimizer: torch.optim.Optimizer, config: Mapping[str, Any]
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    declaration = config["scheduler"]
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(declaration["mode"]),
        factor=float(declaration["factor"]),
        patience=int(declaration["patience_validations"]),
    )


def _load_audio(
    split_name: str,
    data_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    test_unsealed: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if split_name == "test" and not test_unsealed:
        raise RuntimeError(
            "sealed test audio cannot be opened before checkpoint selection"
        )
    declaration = data_config["splits"][split_name]
    entry = next(item for item in manifest["files"] if item["split"] == split_name)
    input_path = ROOT / declaration["input_path"]
    target_path = ROOT / declaration["target_path"]
    for path, field in (
        (input_path, "input_sha256"),
        (target_path, "target_sha256"),
    ):
        expected = declaration[field]
        if entry[field] != expected or _sha256(path) != expected:
            raise RuntimeError(f"{split_name} {field} checksum mismatch")
    signal, input_rate = sf.read(input_path, dtype="float32")
    target, target_rate = sf.read(target_path, dtype="float32")
    expected_rate = int(data_config["sample_rate"])
    if input_rate != expected_rate or target_rate != expected_rate:
        raise RuntimeError(f"{split_name} sample-rate mismatch")
    if signal.ndim != 1 or target.ndim != 1 or signal.shape != target.shape:
        raise RuntimeError(f"{split_name} must be a paired mono file")
    if not np.isfinite(signal).all() or not np.isfinite(target).all():
        raise RuntimeError(f"{split_name} contains a non-finite sample")
    expected_samples = int(entry["input_summary"]["samples"])
    if signal.size != expected_samples or target.size != expected_samples:
        raise RuntimeError(f"{split_name} sample count differs from the manifest")
    return signal, target


def _device_environment(device: torch.device | None) -> dict[str, object]:
    packages = {}
    for name in ("fssr-nam", "numpy", "soundfile", "torch", "pyyaml"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = None
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "index": device.index or 0,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": packages,
        "torch_cuda_runtime": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
        "cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
        "training_device": str(device) if device is not None else None,
        "gpu": gpu,
        "precision": "float32",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "tf32": False,
    }


def _training_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("the counted Wright competence protocol requires CUDA")
    device = torch.device("cuda", 0)
    properties = torch.cuda.get_device_properties(device)
    if properties.total_memory < MINIMUM_GPU_MEMORY_BYTES:
        raise RuntimeError(
            "the counted Wright competence protocol requires a 24 GB-class GPU; "
            f"found {properties.total_memory} bytes"
        )
    return device


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _copy_provenance(run_dir: Path, data_sha256: str, protocol_sha256: str) -> None:
    provenance_dir = run_dir / "provenance"
    provenance_dir.mkdir()
    names = {
        TRAINING_CONFIG_PATH: "training-config.yaml",
        DATA_CONFIG_PATH: "data-config.yaml",
        MODEL_CONFIG_PATH: "model-config.yaml",
        MANIFEST_PATH: "dataset-manifest.json",
        SPLIT_PATH: "split-manifest.json",
        LOCK_PATH: "diagnostic-lock.yaml",
        LOCK_DIGEST_PATH: "diagnostic-lock.sha256",
    }
    if ACTIVE_LOCK_PATH.is_file():
        active_relative = ACTIVE_LOCK_PATH.read_text(encoding="utf-8").strip()
        active_lock = ROOT / active_relative
        names[ACTIVE_LOCK_PATH] = "diagnostic-lock-active.txt"
        names[active_lock] = "diagnostic-lock-amendment.yaml"
        names[active_lock.with_suffix(".sha256")] = "diagnostic-lock-amendment.sha256"
    digests = {}
    for source, name in names.items():
        shutil.copy2(source, provenance_dir / name)
        digests[str(source.relative_to(ROOT))] = _sha256(source)
    _write_json(
        provenance_dir / "source-digests.json",
        {
            "combined_data_sha256": data_sha256,
            "diagnostic_protocol_sha256": protocol_sha256,
            "files": digests,
        },
    )
    for name in ("checkpoints", "predictions", "figures"):
        (run_dir / name).mkdir()


def _log(run_dir: Path, message: str) -> None:
    print(message, flush=True)
    with (run_dir / "stdout.log").open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def _train_seed(
    spec: RunSpec,
    run_dir: Path,
    config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    generator = _seed_everything(spec.seed)
    device = _training_device()
    _write_json(run_dir / "environment.json", _device_environment(device))
    torch.cuda.reset_peak_memory_stats(device)

    # The sealed test pair is deliberately neither hashed nor decoded here.
    train_input, train_target = _load_audio("train", data_config, manifest)
    validation_input, validation_target = _load_audio(
        "validation", data_config, manifest
    )
    segment_samples = int(config["segment_samples"])
    input_frames = frame_audio(train_input, segment_samples)
    target_frames = frame_audio(train_target, segment_samples)
    if input_frames.shape != target_frames.shape:
        raise RuntimeError("training frame pairing failed")

    model = WrightLSTM(hidden_size=64, sample_rate=44_100).to(
        device=device, dtype=torch.float32
    )
    loss_function = _wright_loss(config)
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, config)
    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    best_validation_esr = float("inf")
    best_epoch = 0
    stale_validations = 0
    optimizer_updates = 0
    stopped_early = False
    validation_frequency = int(config["validation_frequency_epochs"])
    early_patience = int(config["early_stopping"]["patience_validations"])

    for epoch in range(1, int(config["maximum_epochs"]) + 1):
        epoch_started = time.perf_counter()
        training_loss, updates = train_epoch(
            model,
            input_frames,
            target_frames,
            loss_function,
            optimizer,
            batch_size=int(config["batch_size"]),
            warmup_samples=int(config["warmup_samples"]),
            tbptt_samples=int(config["tbptt_samples"]),
            device=device,
            generator=generator,
        )
        optimizer_updates += updates
        record: dict[str, Any] = {
            "epoch": epoch,
            "training_wright_loss": training_loss,
            "optimizer_updates": optimizer_updates,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "wall_seconds": time.perf_counter() - epoch_started,
        }
        if epoch % validation_frequency == 0:
            prediction = predict_streaming(
                model,
                validation_input,
                device=device,
                chunk_samples=EVALUATION_CHUNK_SAMPLES,
            )
            validation_loss, validation_esr = evaluate_prediction(
                prediction, validation_target, loss_function
            )
            scheduler.step(validation_loss)
            record.update(
                {
                    "validation_wright_loss": validation_loss,
                    "validation_esr": validation_esr,
                    "learning_rate_after_scheduler": float(
                        optimizer.param_groups[0]["lr"]
                    ),
                }
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_validation_esr = validation_esr
                best_epoch = epoch
                stale_validations = 0
                torch.save(model.state_dict(), run_dir / "checkpoints/best-model.pt")
            else:
                stale_validations += 1
            record["stale_validations"] = stale_validations
            _log(
                run_dir,
                f"seed={spec.seed} epoch={epoch} train={training_loss:.9g} "
                f"val={validation_loss:.9g} val_esr={validation_esr:.9g} "
                f"stale={stale_validations} lr={optimizer.param_groups[0]['lr']:.9g}",
            )
        history.append(record)
        _write_json(run_dir / "training-history.json", history)
        # Wright's pinned script stops only once the counter exceeds 25.
        if stale_validations > early_patience:
            stopped_early = True
            _log(run_dir, f"official early stopping at epoch={epoch}")
            break

    if best_epoch < 1:
        raise RuntimeError("training produced no validation-selected checkpoint")
    checkpoint_path = run_dir / "checkpoints/best-model.pt"
    checkpoint_sha256 = _sha256(checkpoint_path)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True),
        strict=True,
    )
    _write_json(
        run_dir / "checkpoints/index.json",
        {
            "path": "best-model.pt",
            "sha256": checkpoint_sha256,
            "selection": "lowest_validation_wright_loss",
            "comparison": "strict_improvement",
            "epoch": best_epoch,
            "validation_wright_loss": best_validation_loss,
        },
    )

    # This is the sole unseal point: training has ended and the selected weights
    # have already been reloaded and bound by checksum.
    unsealed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    _write_json(
        run_dir / "test-seal.json",
        {
            "status": "unsealed_after_checkpoint_selection",
            "test_opened_during_training": False,
            "checkpoint_path": "checkpoints/best-model.pt",
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_epoch": best_epoch,
            "unsealed_at": unsealed_at,
        },
    )
    test_input, test_target = _load_audio(
        "test", data_config, manifest, test_unsealed=True
    )
    test_prediction = predict_streaming(
        model,
        test_input,
        device=device,
        chunk_samples=EVALUATION_CHUNK_SAMPLES,
    )
    parity_samples = min(test_input.size, PARITY_PREFIX_SAMPLES)
    alternate = predict_streaming(
        model,
        test_input[:parity_samples],
        device=device,
        chunk_samples=PARITY_CHUNK_SAMPLES,
    )
    block_difference = float(
        np.max(np.abs(alternate - test_prediction[:parity_samples]), initial=0.0)
    )
    if block_difference > 2.0e-5:
        raise RuntimeError(f"post-training block parity failed: {block_difference:.9g}")
    test_loss, test_esr = evaluate_prediction(
        test_prediction, test_target, loss_function
    )
    if not np.isfinite(test_esr):
        raise RuntimeError("sealed test ESR is non-finite")
    sf.write(
        run_dir / "predictions/test-output.wav",
        test_prediction,
        int(config["sample_rate"]),
        subtype="FLOAT",
    )
    metrics = {
        "schema_version": 1,
        "run_id": spec.run_id,
        "seed": spec.seed,
        "best_epoch": best_epoch,
        "best_validation_wright_loss": best_validation_loss,
        "best_validation_esr": best_validation_esr,
        "epochs_completed": int(history[-1]["epoch"]),
        "stopped_by_official_early_stopping": stopped_early,
        "optimizer_updates": optimizer_updates,
        "training_segments": int(input_frames.shape[0]),
        "segment_samples": segment_samples,
        "dropped_training_tail_samples": int(
            train_input.size - input_frames.shape[0] * segment_samples
        ),
        "test_wright_loss": test_loss,
        "test_esr": test_esr,
        "test_time_metrics": time_metrics(test_prediction, test_target),
        "test_samples": int(test_input.size),
        "test_evaluated_after_checkpoint_selection": True,
        "test_unsealed_at": unsealed_at,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "latency_samples": 0,
        "sample_rate": 44_100,
        "checks": {
            "finite_prediction": bool(np.isfinite(test_prediction).all()),
            "block_parity_max_abs": block_difference,
            "block_parity_prefix_samples": parity_samples,
        },
    }
    _write_json(run_dir / "metrics.json", metrics)
    return metrics


def _validate_completed_artifacts(
    run_dir: Path, expected_metrics: Mapping[str, Any]
) -> None:
    required = (
        "config-resolved.yaml",
        "run-spec.json",
        "command.txt",
        "registration.json",
        "environment.json",
        "training-history.json",
        "test-seal.json",
        "metrics.json",
        "checkpoints/best-model.pt",
        "checkpoints/index.json",
        "predictions/test-output.wav",
        "provenance/training-config.yaml",
        "provenance/data-config.yaml",
        "provenance/model-config.yaml",
        "provenance/dataset-manifest.json",
        "provenance/split-manifest.json",
        "provenance/diagnostic-lock.yaml",
        "provenance/diagnostic-lock.sha256",
        "provenance/source-digests.json",
    )
    missing = [path for path in required if not (run_dir / path).is_file()]
    if missing:
        raise RuntimeError(
            "completed competence trajectory lacks required artifacts: "
            + ", ".join(missing)
        )
    persisted = _load_json(run_dir / "metrics.json")
    if persisted != dict(expected_metrics):
        raise RuntimeError("returned and persisted competence metrics differ")


def _run_seed(
    spec: RunSpec,
    executor: R1Executor,
    config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    data_sha256: str,
    protocol_sha256: str,
    commit: str,
    command: str,
) -> dict[str, Any]:
    reservation = executor.prepare_run(
        spec,
        data_sha256=data_sha256,
        commit=commit,
        command=command,
    )
    run_dir = Path(str(reservation["run_directory"]))
    started = time.perf_counter()
    status = "failed"
    failure_reason = ""
    metrics: dict[str, Any] | None = None
    caught: BaseException | None = None
    try:
        _copy_provenance(run_dir, data_sha256, protocol_sha256)
        metrics = _train_seed(spec, run_dir, config, data_config, manifest)
        _validate_completed_artifacts(run_dir, metrics)
        status = "completed"
    except BaseException as error:
        caught = error
        failure_reason = f"{type(error).__name__}: {error}"
        (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        if not (run_dir / "metrics.json").exists():
            _write_json(
                run_dir / "metrics.json",
                {"status": "unavailable", "reason": failure_reason},
            )
        if not (run_dir / "checkpoints/index.json").exists():
            _write_json(
                run_dir / "checkpoints/index.json",
                {"status": "unavailable", "reason": failure_reason},
            )
        if not (run_dir / "environment.json").exists():
            _write_json(run_dir / "environment.json", _device_environment(None))
    finally:
        _write_json(
            run_dir / "timings.json",
            {
                "wall_seconds": time.perf_counter() - started,
                "gpu_peak_memory_bytes": (
                    torch.cuda.max_memory_allocated()
                    if torch.cuda.is_available()
                    else None
                ),
            },
        )
        executor.finalize_run(spec, status=status, failure_reason=failure_reason)
    if caught is not None:
        raise caught
    if metrics is None:  # pragma: no cover - defensive after terminal branches
        raise RuntimeError("completed competence run has no metrics")
    return metrics


def _registered_result(
    spec: RunSpec, *, expected_commit: str, expected_data_sha256: str
) -> tuple[str, dict[str, Any] | None, str] | None:
    matches = [
        entry for entry in read_runs(LEDGER_PATH) if entry.get("run_id") == spec.run_id
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"duplicate ledger entries for {spec.run_id}")
    entry = matches[0]
    status = str(entry["status"])
    reason = str(entry.get("failure_reason", ""))
    if entry.get("commit") != expected_commit:
        raise RuntimeError(
            f"registered trajectory commit differs from this gate: {spec.run_id}"
        )
    if entry.get("data_sha256") != expected_data_sha256:
        raise RuntimeError(
            f"registered trajectory data provenance differs: {spec.run_id}"
        )
    if status != "completed":
        return status, None, reason
    run_dir = ROOT / "experiments/runs" / spec.run_id
    metrics = _load_json(run_dir / "metrics.json")
    if (
        metrics.get("run_id") != spec.run_id
        or int(metrics.get("seed", -1)) != spec.seed
    ):
        raise RuntimeError(f"registered metrics mismatch for {spec.run_id}")
    test_esr = metrics.get("test_esr")
    if isinstance(test_esr, bool) or not isinstance(test_esr, (int, float)):
        raise RuntimeError(f"registered test ESR is invalid for {spec.run_id}")
    return status, metrics, reason


def _gate_payload(
    decision: CompetenceGateDecision,
    evidence: list[dict[str, object]],
    *,
    config_sha256: str,
    data_sha256: str,
    protocol_sha256: str,
    commit: str,
    noncanonical_attempts: tuple[dict[str, Any], ...],
) -> dict[str, object]:
    seed_zero = dict(decision.seed_esr)[0]
    seed_zero_decision = "passed" if decision.seed_zero_continuation else "failed"
    return {
        "schema_version": 1,
        "campaign_version": "FSSR-R1-v1",
        "stage": "competence",
        "status": decision.status,
        "scientific_result": True,
        "decision": decision.as_dict(),
        "gates": {
            "competence_seed0": {
                "decision": seed_zero_decision,
                "passed": decision.seed_zero_continuation,
                "test_esr": seed_zero,
                "limit": 0.15,
            },
            "competence": decision.as_dict(),
        },
        "evidence": evidence,
        "audit": {
            "quarantined_noncanonical_attempts": [
                {
                    "run_id": entry["run_id"],
                    "status": entry["status"],
                    "failure_reason": entry["failure_reason"],
                    "counted_toward_gate": False,
                }
                for entry in noncanonical_attempts
            ]
        },
        "training_config_sha256": config_sha256,
        "data_sha256": data_sha256,
        "protocol_sha256": protocol_sha256,
        "commit": commit,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def _write_summary_once(payload: Mapping[str, object]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(payload), indent=2) + "\n")


def _existing_summary() -> dict[str, Any] | None:
    if not SUMMARY_PATH.exists():
        return None
    summary = _load_json(SUMMARY_PATH)
    if (
        summary.get("campaign_version") != "FSSR-R1-v1"
        or summary.get("stage") != "competence"
        or summary.get("status") not in {"passed", "failed"}
    ):
        raise RuntimeError("existing competence summary is malformed")
    return summary


def _publish_gate_decisions(summary: Mapping[str, Any]) -> None:
    protocol_sha256 = summary.get("protocol_sha256")
    if not isinstance(protocol_sha256, str):
        raise RuntimeError("competence summary lacks the diagnostic protocol digest")
    created_at = summary.get("created_at")
    if not isinstance(created_at, str):
        raise RuntimeError("competence summary lacks its evaluation timestamp")
    summary_gates = summary.get("gates")
    if not isinstance(summary_gates, dict) or set(summary_gates) != {
        "competence_seed0",
        "competence",
    }:
        raise RuntimeError("competence summary has an invalid gate mapping")
    summary_reference = str(SUMMARY_PATH.relative_to(ROOT))
    summary_sha256 = _sha256(SUMMARY_PATH)
    proposed = {}
    for name, value in summary_gates.items():
        if not isinstance(value, dict):
            raise RuntimeError(f"competence gate {name} must be a mapping")
        decision = dict(value)
        decision.update(
            {
                "evaluated_at": created_at,
                "protocol_sha256": protocol_sha256,
                "evidence": summary_reference,
                "evidence_sha256": summary_sha256,
            }
        )
        proposed[name] = decision

    if GATES_PATH.exists():
        document = _load_json(GATES_PATH)
        if (
            document.get("schema_version") != 1
            or document.get("campaign_version") != "FSSR-R1-v1"
        ):
            raise RuntimeError("existing R1 gate registry has an invalid header")
    else:
        document = {
            "schema_version": 1,
            "campaign_version": "FSSR-R1-v1",
            "gates": {},
        }
    gates = document.get("gates")
    if not isinstance(gates, dict):
        raise RuntimeError("existing R1 gate registry has no gate mapping")
    changed = False
    for name, decision in proposed.items():
        if name in gates:
            if gates[name] != decision:
                raise RuntimeError(f"refusing divergent immutable gate: {name}")
        else:
            gates[name] = decision
            changed = True
    if not changed:
        return
    document["updated_at"] = created_at
    GATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = GATES_PATH.with_name(f".{GATES_PATH.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, indent=2) + "\n")
    os.replace(temporary, GATES_PATH)


def _run_gate(
    config: dict[str, Any],
    data_config: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, object]:
    protocol_sha256 = validate_lock_digest(ROOT)
    existing_summary = _existing_summary()
    if existing_summary is not None:
        if existing_summary.get("protocol_sha256") != protocol_sha256:
            raise RuntimeError("competence summary belongs to a different protocol")
        _publish_gate_decisions(existing_summary)
        return existing_summary
    _require_clean_worktree()
    commit = _git_commit()
    data_sha256 = _combined_digest((DATA_CONFIG_PATH, MANIFEST_PATH, SPLIT_PATH))
    config_sha256 = _sha256(TRAINING_CONFIG_PATH)
    command = shlex.join([sys.executable, *sys.argv])
    evidence: list[dict[str, object]] = []
    observed: dict[int, float] = {}
    audit_executor = R1Executor(ROOT, stage_configs={"competence": config})
    noncanonical_attempts = audit_executor.quarantined_noncanonical_attempts()

    for declared_seed in config["seeds"]:
        seed = int(declared_seed)
        if seed > 0 and not seed_zero_allows_additional_runs(config, observed[0]):
            break
        decisions: dict[str, object] = {}
        if seed > 0:
            decisions["competence_seed0"] = {
                "decision": "passed",
                "test_esr": observed[0],
            }
        executor = R1Executor(
            ROOT,
            stage_configs={"competence": config},
            gate_decisions=decisions,
        )
        spec = RunSpec("competence", "bigmuff", "lstm64", "wright", seed)
        registered = _registered_result(
            spec,
            expected_commit=commit,
            expected_data_sha256=data_sha256,
        )
        if registered is None:
            metrics = _run_seed(
                spec,
                executor,
                config,
                data_config,
                manifest,
                data_sha256=data_sha256,
                protocol_sha256=protocol_sha256,
                commit=commit,
                command=command,
            )
            status = "completed"
            reason = ""
        else:
            status, metrics, reason = registered
        if status != "completed" or metrics is None:
            raise RuntimeError(
                f"competence trajectory {spec.run_id} is terminal as {status}: {reason}"
            )
        esr = float(metrics["test_esr"])
        observed[seed] = esr
        evidence.append(
            {
                "run_id": spec.run_id,
                "seed": seed,
                "ledger_status": status,
                "test_esr": esr,
                "commit": commit,
                "data_sha256": data_sha256,
                "metrics": f"experiments/runs/{spec.run_id}/metrics.json",
            }
        )
        if seed == 0 and not seed_zero_allows_additional_runs(config, esr):
            break

    decision = evaluate_competence_gate(config, observed)
    payload = _gate_payload(
        decision,
        evidence,
        config_sha256=config_sha256,
        data_sha256=data_sha256,
        protocol_sha256=protocol_sha256,
        commit=commit,
        noncanonical_attempts=noncanonical_attempts,
    )
    _write_summary_once(payload)
    _publish_gate_decisions(payload)
    return payload


def _run_preflight(steps: int, config: Mapping[str, Any]) -> dict[str, object]:
    if isinstance(steps, bool) or not 1 <= steps <= 16:
        raise ValueError("--preflight-steps must be between 1 and 16")
    generator = _seed_everything(0)
    device = torch.device("cpu")
    sample_count = int(config["warmup_samples"]) + steps * int(config["tbptt_samples"])
    signal = np.linspace(-0.1, 0.1, sample_count, dtype=np.float32)
    target = np.tanh(2.0 * signal).astype(np.float32)
    input_frames = frame_audio(signal, sample_count)
    target_frames = frame_audio(target, sample_count)
    model = WrightLSTM(hidden_size=64, sample_rate=44_100).to(device)
    optimizer = _optimizer(model, config)
    loss, updates = train_epoch(
        model,
        input_frames,
        target_frames,
        _wright_loss(config),
        optimizer,
        batch_size=1,
        warmup_samples=int(config["warmup_samples"]),
        tbptt_samples=int(config["tbptt_samples"]),
        device=device,
        generator=generator,
    )
    if updates != steps:
        raise RuntimeError(
            f"preflight requested {steps} updates but observed {updates}"
        )
    scheduler = _scheduler(optimizer, config)
    scheduler.step(loss)
    probe = np.linspace(-0.05, 0.05, 2_049, dtype=np.float32)
    expected = predict_streaming(model, probe, device=device, chunk_samples=257)
    alternate = predict_streaming(model, probe, device=device, chunk_samples=509)
    parity = float(np.max(np.abs(expected - alternate), initial=0.0))
    if not np.isfinite(loss) or parity > 2.0e-5:
        raise RuntimeError("non-scientific Wright preflight failed")
    return {
        "schema_version": 1,
        "mode": "non_scientific_synthetic_preflight",
        "scientific_result": False,
        "writes_performed": False,
        "real_audio_opened": False,
        "test_audio_opened": False,
        "optimizer_steps_requested": steps,
        "optimizer_steps_observed": updates,
        "finite_synthetic_loss": True,
        "block_parity_max_abs": parity,
        "canonical_run_ids": [
            RunSpec("competence", "bigmuff", "lstm64", "wright", seed).run_id
            for seed in config["seeds"]
        ],
    }


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args = _parser().parse_args()
    config, data_config, manifest, _ = _load_declarations()
    if args.preflight or args.preflight_steps is not None:
        steps = 1 if args.preflight else args.preflight_steps
        result = _run_preflight(int(steps), config)
    else:
        signal.signal(signal.SIGTERM, _raise_termination)
        result = _run_gate(config, data_config, manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
