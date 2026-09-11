"""Locked training primitives for the non-competence R1 diagnostics.

The public runner owns run reservation and provenance.  This module only loads
the frozen M4 data, constructs the declared models, and executes one numerical
trajectory.  Keeping those responsibilities separate lets preflight exercise
the real optimization path without creating a counted campaign run.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml
from numpy.typing import NDArray
from scipy.signal import lfilter
from torch import Tensor, nn

from fssr_nam.data.excitations import generate_excitation
from fssr_nam.losses import WrightLoss
from fssr_nam.metrics.spectral import spectral_metrics
from fssr_nam.metrics.time import time_metrics
from fssr_nam.models.r1 import R1Cascade, R1Mono, RF2047Residual
from fssr_nam.models.residual import normalized_residual_penalty
from fssr_nam.training.m4 import causal_predict
from fssr_nam.training.m4 import model_factory as m4_model_factory
from fssr_nam.training.r1 import PhaseScheduler, ResidualPenaltyRamp

FloatArray = NDArray[np.float32]

M4_CONFIG_PATH = Path("configs/training/m4_smoke.yaml")
M4_MODEL_CONFIG_PATH = Path("configs/training/m3_synthetic.yaml")
M4_MANIFEST_PATH = Path("datasets/manifests/m4_internal.json")
M4_SPLIT_PATH = Path("datasets/splits/m4_internal.json")
SYNTHETIC_CONFIG_PATH = Path("configs/data/synthetic.yaml")

DEVICE_TO_M4 = {
    "fulltone": "fulltone_full_drive_2",
    "bigmuff": "electro_harmonix_big_muff",
}
M4_LOCKED_DIMENSIONS = {
    "context_samples": 6_346,
    "output_samples": 8_192,
    "batch_size": 2,
    "validation_samples": 240_000,
    "evaluation_block_samples": 8_192,
}
LOCKED_CHECKPOINT_STEPS = (200, 1_000, 5_000)
A2_LINEAR_MACS_PER_SAMPLE = 11_777.0


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    """Resolved numerical budget for one trajectory."""

    optimizer_steps: int
    checkpoint_steps: tuple[int, ...]
    context_samples: int
    output_samples: int
    batch_size: int
    validation_samples: int
    evaluation_block_samples: int
    preflight: bool = False


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """In-memory train/validation pairs; the sealed test split is not opened."""

    train_input: FloatArray
    train_target: FloatArray
    validation_input: FloatArray
    validation_target: FloatArray
    manifest: dict[str, Any]
    split_manifest: dict[str, Any]
    manifest_bytes: bytes
    split_bytes: bytes
    data_sha256: str


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Completed trajectory artifacts returned to the immutable runner."""

    metrics: dict[str, Any]
    history: list[dict[str, Any]]
    selected_state: dict[str, Tensor]
    selected_residual_state: dict[str, Tensor] | None
    selected_validation_prediction: FloatArray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combined_sha256(*payloads: bytes) -> str:
    digest = hashlib.sha256()
    for payload in payloads:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _mapping_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def validate_locked_training_reuse(
    stage_config: Mapping[str, Any], m4_config: Mapping[str, Any]
) -> None:
    """Reject drift from the frozen M4 data dimensions and optimizer recipe."""
    if stage_config.get("optimizer_steps") != 5_000:
        raise ValueError("R1 diagnostics require exactly 5000 optimizer updates")
    if tuple(stage_config.get("checkpoint_steps", ())) != LOCKED_CHECKPOINT_STEPS:
        raise ValueError("R1 checkpoints must be exactly 200/1000/5000")
    for name, expected in M4_LOCKED_DIMENSIONS.items():
        if m4_config.get(name) != expected:
            raise ValueError(f"frozen M4 {name} changed from {expected}")
    optimizer = m4_config.get("optimizer")
    if optimizer != {
        "name": "Adam",
        "learning_rate": 0.004,
        "weight_decay": 3.17e-7,
    }:
        raise ValueError("frozen M4 optimizer declaration changed")
    if m4_config.get("gradient_clip_norm") != 1.0:
        raise ValueError("frozen M4 gradient clip changed")
    m4_loss = m4_config.get("loss")
    expected_loss = {
        "mse": 1.0,
        "mrstft": 0.0005,
        "spline_curvature": 1.0e-5,
        "normalized_residual_energy": 1.0e-3,
    }
    if m4_loss != expected_loss:
        raise ValueError("frozen M4 loss/regularization declaration changed")

    if stage_config.get("stage") == "factorial":
        for name, expected in M4_LOCKED_DIMENSIONS.items():
            if stage_config.get(name) != expected:
                raise ValueError(f"factorial {name} must reuse M4 exactly")
        if stage_config.get("optimizer") != optimizer:
            raise ValueError("factorial optimizer must reuse M4 exactly")
        if stage_config.get("gradient_clip_norm") != 1.0:
            raise ValueError("factorial gradient clip must reuse M4 exactly")
        if stage_config.get("losses", {}).get("m4") != {
            "mse": 1.0,
            "mrstft": 0.0005,
        }:
            raise ValueError("factorial M4 loss differs from M4")
        if stage_config.get("regularization") != {
            "spline_curvature": 1.0e-5,
            "normalized_residual_energy": 1.0e-3,
        }:
            raise ValueError("factorial regularizations differ from M4")


def execution_profile(
    stage_config: Mapping[str, Any],
    m4_config: Mapping[str, Any],
    *,
    preflight: bool,
) -> ExecutionProfile:
    validate_locked_training_reuse(stage_config, m4_config)
    if preflight:
        return ExecutionProfile(
            optimizer_steps=2,
            checkpoint_steps=(1, 2),
            context_samples=M4_LOCKED_DIMENSIONS["context_samples"],
            output_samples=4_096,
            batch_size=1,
            validation_samples=8_192,
            evaluation_block_samples=1_024,
            preflight=True,
        )
    return ExecutionProfile(
        optimizer_steps=5_000,
        checkpoint_steps=LOCKED_CHECKPOINT_STEPS,
        context_samples=M4_LOCKED_DIMENSIONS["context_samples"],
        output_samples=M4_LOCKED_DIMENSIONS["output_samples"],
        batch_size=M4_LOCKED_DIMENSIONS["batch_size"],
        validation_samples=M4_LOCKED_DIMENSIONS["validation_samples"],
        evaluation_block_samples=M4_LOCKED_DIMENSIONS["evaluation_block_samples"],
    )


def _read_pair(
    root: Path,
    input_entry: Mapping[str, Any],
    *,
    frame_limit: int | None,
) -> tuple[FloatArray, FloatArray]:
    input_path = root / str(input_entry["input_path"])
    target_path = root / str(input_entry["target_path"])
    if sha256_file(input_path) != input_entry["input_sha256"]:
        raise RuntimeError(f"input checksum mismatch: {input_path}")
    if sha256_file(target_path) != input_entry["target_sha256"]:
        raise RuntimeError(f"target checksum mismatch: {target_path}")
    read_frames = -1 if frame_limit is None else frame_limit
    source, source_rate = sf.read(input_path, frames=read_frames, dtype="float32")
    target, target_rate = sf.read(target_path, frames=read_frames, dtype="float32")
    if source_rate != 48_000 or target_rate != 48_000:
        raise RuntimeError("R1 physical diagnostics require the M4 48 kHz audio")
    if source.ndim != 1 or target.ndim != 1 or source.shape != target.shape:
        raise RuntimeError("R1 physical diagnostics require paired mono audio")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise RuntimeError("R1 physical audio contains non-finite samples")
    return np.asarray(source, dtype=np.float32), np.asarray(target, dtype=np.float32)


def load_m4_dataset(
    root: Path,
    device: str,
    profile: ExecutionProfile,
) -> DatasetBundle:
    """Load exactly the frozen M4 Fulltone/Big Muff manifest and split."""
    try:
        m4_device = DEVICE_TO_M4[device]
    except KeyError as error:
        raise ValueError(f"R1 diagnostic device is not in M4: {device}") from error
    manifest_bytes = (root / M4_MANIFEST_PATH).read_bytes()
    split_bytes = (root / M4_SPLIT_PATH).read_bytes()
    manifest = json.loads(manifest_bytes)
    split_manifest = json.loads(split_bytes)
    if (
        manifest.get("name") != "m4_internal"
        or manifest.get("tier") != "INTERNAL_DEV"
        or split_manifest.get("dataset") != "m4_internal"
        or split_manifest.get("path_overlap") != []
    ):
        raise RuntimeError("the frozen M4 manifest/split contract is invalid")
    entries = {
        entry["split"]: entry
        for entry in manifest.get("files", [])
        if entry.get("device") == m4_device
    }
    if set(entries) != {"train", "validation", "test"}:
        raise RuntimeError(f"incomplete M4 source-disjoint split for {device}")
    groups = split_manifest.get("groups", {}).get(m4_device)
    if not isinstance(groups, dict) or set(groups) != {"train", "validation", "test"}:
        raise RuntimeError(f"missing M4 split groups for {device}")
    minimum = profile.context_samples + profile.output_samples + 1
    frame_limit = (
        max(minimum, profile.validation_samples) if profile.preflight else None
    )
    train_input, train_target = _read_pair(
        root, entries["train"], frame_limit=frame_limit
    )
    validation_input, validation_target = _read_pair(
        root, entries["validation"], frame_limit=frame_limit
    )
    if train_input.size < minimum:
        raise RuntimeError("M4 training file is too short for the locked windows")
    return DatasetBundle(
        train_input=train_input,
        train_target=train_target,
        validation_input=validation_input,
        validation_target=validation_target,
        manifest=manifest,
        split_manifest=split_manifest,
        manifest_bytes=manifest_bytes,
        split_bytes=split_bytes,
        data_sha256=combined_sha256(manifest_bytes, split_bytes),
    )


def two_clippers_with_interstage_filter(signal: NDArray[np.floating]) -> FloatArray:
    """Deterministic causal two-clipper topology used by the R1 cascade gate."""
    source = np.asarray(signal, dtype=np.float64)
    if source.ndim != 1 or not np.isfinite(source).all():
        raise ValueError("synthetic input must be finite mono audio")
    first = np.tanh(3.2 * source) / np.tanh(3.2)
    interstage = lfilter(
        np.array([0.18, 0.34, 0.29, 0.14, 0.05], dtype=np.float64),
        [1.0],
        first,
    )
    positive = np.tanh(2.4 * (1.15 * interstage + 0.025))
    negative = 0.82 * np.tanh(4.0 * (1.15 * interstage + 0.025))
    second = np.where(interstage >= 0.0, positive, negative)
    output = lfilter(
        np.array([0.12, 0.31, 0.34, 0.17, 0.06], dtype=np.float64),
        [1.0],
        second,
    )
    if not np.isfinite(output).all():
        raise RuntimeError("two-clipper system produced non-finite audio")
    return np.asarray(output, dtype=np.float32)


def load_synthetic_cascade_dataset(
    root: Path, profile: ExecutionProfile
) -> DatasetBundle:
    """Generate three seed-disjoint realizations from the frozen synthetic config."""
    config_bytes = (root / SYNTHETIC_CONFIG_PATH).read_bytes()
    config = yaml.safe_load(config_bytes)
    if (
        config.get("tier") != "SYNTHETIC"
        or config.get("derived_sample_rates") != [96_000, 48_000]
        or config.get("seed") != 0
    ):
        raise RuntimeError("frozen synthetic configuration changed")
    sample_rate = 48_000
    duration = float(config["duration_seconds"])
    seeds = {"train": 0, "validation": 1, "test": 2}
    pairs: dict[str, tuple[FloatArray, FloatArray]] = {}
    hashes: dict[str, dict[str, str]] = {}
    for split in ("train", "validation"):
        seed = seeds[split]
        source = generate_excitation(
            "colored_noise",
            sample_rate=sample_rate,
            duration_seconds=duration,
            seed=seed,
        )
        target = two_clippers_with_interstage_filter(source)
        pairs[split] = (source, target)
        hashes[split] = {
            "input_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
            "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
        }
    minimum = profile.context_samples + profile.output_samples + 1
    if pairs["train"][0].size < minimum:
        raise RuntimeError("synthetic fixture is too short for the locked windows")
    manifest = {
        "schema_version": 1,
        "campaign_version": "FSSR-R1-v1",
        "tier": "SYNTHETIC",
        "system": "two_clippers_with_interstage_filter",
        "sample_rate": sample_rate,
        "source_configuration": str(SYNTHETIC_CONFIG_PATH),
        "splits": {
            "train": {
                "seed": seeds["train"],
                "samples": len(pairs["train"][0]),
                **hashes["train"],
            },
            "validation": {
                "seed": seeds["validation"],
                "samples": len(pairs["validation"][0]),
                **hashes["validation"],
            },
            "test": {"seed": seeds["test"], "status": "sealed_not_materialized"},
        },
    }
    split_manifest = {
        "schema_version": 1,
        "rule": "independent deterministic generator seed per split",
        "groups": {split: [f"synthetic-seed-{seed}"] for split, seed in seeds.items()},
        "path_overlap": [],
        "leakage_check": "passed",
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    split_bytes = (json.dumps(split_manifest, sort_keys=True) + "\n").encode()
    return DatasetBundle(
        train_input=pairs["train"][0],
        train_target=pairs["train"][1],
        validation_input=pairs["validation"][0],
        validation_target=pairs["validation"][1],
        manifest=manifest,
        split_manifest=split_manifest,
        manifest_bytes=manifest_bytes,
        split_bytes=split_bytes,
        data_sha256=combined_sha256(manifest_bytes, split_bytes),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_model(
    root: Path,
    *,
    stage: str,
    model_name: str,
    promoted_residual_state: Mapping[str, Tensor] | None = None,
) -> tuple[nn.Module, str]:
    """Construct only one of the four preregistered diagnostic families."""
    model_config = _mapping_yaml(root / M4_MODEL_CONFIG_PATH)["model"]
    if stage == "factorial" and model_name == "a2":
        return m4_model_factory("B0", root=root, model_config=model_config), "B0"
    if stage == "factorial" and model_name == "s3":
        return m4_model_factory("S3", root=root, model_config=model_config), "S3"
    common = {
        "taps": int(model_config["taps"]),
        "num_knots": int(model_config["num_knots"]),
        "residual_channels": int(model_config["residual_channels"]),
        "slow_hidden_size": int(model_config["slow_hidden_size"]),
        "slow_decimation": int(model_config["slow_decimation"]),
    }
    if stage == "horizon" and model_name in {"rf31", "rf2047"}:
        receptive_field = 31 if model_name == "rf31" else 2_047
        return R1Mono(receptive_field=receptive_field, **common), "R1"
    if stage == "cascade" and model_name == "mono":
        return R1Mono(receptive_field=2_047, **common), "R1"
    if stage == "cascade" and model_name == "cascade":
        return (
            R1Cascade(
                receptive_field=2_047,
                promoted_residual_state=promoted_residual_state,
                **common,
            ),
            "R1",
        )
    raise ValueError(f"unsupported R1 diagnostic model: {stage}/{model_name}")


def theoretical_macs_per_sample(stage: str, model_name: str) -> float:
    """Return the same linear-MAC convention used by the M3/A2 audits."""
    if stage == "factorial" and model_name == "a2":
        return A2_LINEAR_MACS_PER_SAMPLE
    if stage == "factorial" and model_name == "s3":
        return 830.5
    if stage == "horizon" and model_name == "rf31":
        return 830.5
    if (stage, model_name) in {("horizon", "rf2047"), ("cascade", "mono")}:
        return 942.5
    if stage == "cascade" and model_name == "cascade":
        return 959.5
    raise ValueError(f"no theoretical cost for {stage}/{model_name}")


def _components(
    model: nn.Module, model_kind: str, windows: Tensor, output_samples: int
) -> tuple[Tensor, Tensor, Tensor]:
    if model_kind == "B0":
        output = model(windows, pad_start=False)
        if output.shape[-1] != output_samples:
            raise RuntimeError("A2 output does not match the locked window")
        return output, output, torch.zeros_like(output)
    provider = getattr(model, "forward_components", None)
    if provider is None:
        output = model(windows)[..., -output_samples:]
        return output, output, torch.zeros_like(output)
    output, core, residual = provider(windows)
    return (
        output[..., -output_samples:],
        core[..., -output_samples:],
        residual[..., -output_samples:],
    )


def _curvature(model: nn.Module, reference: Tensor) -> Tensor:
    core = getattr(model, "core", None)
    regularization = getattr(core, "regularization", None)
    return regularization() if callable(regularization) else reference.new_zeros(())


def _gradient_norms(model: nn.Module) -> tuple[dict[str, float], bool]:
    sums: dict[str, float] = {}
    finite = True
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        block = name.split(".", maxsplit=1)[0]
        gradient = parameter.grad.detach()
        finite = finite and bool(torch.isfinite(gradient).all())
        sums[block] = sums.get(block, 0.0) + float(gradient.double().square().sum())
    return {name: math.sqrt(value) for name, value in sorted(sums.items())}, finite


def _residual_ratio(
    model: nn.Module,
    model_kind: str,
    signal: FloatArray,
    *,
    device: torch.device,
    block_samples: int,
) -> float:
    if model_kind == "B0" or not hasattr(model, "stream_components"):
        return 0.0
    model.reset_state()
    output_energy = 0.0
    residual_energy = 0.0
    with torch.inference_mode():
        for start in range(0, signal.size, block_samples):
            block = torch.from_numpy(signal[start : start + block_samples]).to(device)
            output, _, residual = model.stream_components(block)
            output_energy += float(output.double().square().sum().cpu())
            residual_energy += float(residual.double().square().sum().cpu())
    return residual_energy / max(output_energy, np.finfo(float).eps)


def _prediction_metrics(prediction: FloatArray, target: FloatArray) -> dict[str, float]:
    return {
        **time_metrics(prediction, target),
        "output_energy": float(np.sum(np.square(prediction), dtype=np.float64)),
        "target_energy": float(np.sum(np.square(target), dtype=np.float64)),
    }


def _predict(
    model: nn.Module,
    model_kind: str,
    signal: FloatArray,
    *,
    device: torch.device,
    profile: ExecutionProfile,
    block_samples: int | None = None,
) -> FloatArray:
    return causal_predict(
        model,
        model_kind,
        signal,
        device=device,
        context_samples=profile.context_samples,
        block_samples=block_samples or profile.evaluation_block_samples,
    )


def _state_copy(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _residual_state(model: nn.Module) -> dict[str, Tensor] | None:
    residual = getattr(model, "residual", None)
    if not isinstance(residual, RF2047Residual):
        return None
    return {
        name: value.detach().cpu().clone()
        for name, value in residual.state_dict().items()
    }


# Codex: Sourcery low-code-quality (9%); refactor with behavior/protocol tests.
def run_training(
    *,
    model: nn.Module,
    model_kind: str,
    stage: str,
    model_name: str,
    loss_mode: str,
    seed: int,
    dataset: DatasetBundle,
    stage_config: Mapping[str, Any],
    m4_config: Mapping[str, Any],
    profile: ExecutionProfile,
    device: torch.device,
    checkpoint_directory: Path | None,
) -> TrainingResult:
    """Execute one exact trajectory, including common-checkpoint selection."""
    if loss_mode not in {"m4", "wright"}:
        raise ValueError("loss_mode must be m4 or wright")
    validate_locked_training_reuse(stage_config, m4_config)
    if profile.optimizer_steps != (2 if profile.preflight else 5_000):
        raise ValueError("execution profile does not match its counted status")
    seed_everything(seed)
    model = model.to(device=device, dtype=torch.float32)
    scheduler = PhaseScheduler.from_config(stage_config["phase_schedule"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(m4_config["optimizer"]["learning_rate"]),
        weight_decay=float(m4_config["optimizer"]["weight_decay"]),
    )
    wright = WrightLoss()
    mrstft: nn.Module | None = None
    if loss_mode == "m4":
        from nam._dependencies.auraloss.freq import MultiResolutionSTFTLoss

        mrstft = MultiResolutionSTFTLoss().to(device)
    residual_ramp = None
    if stage in {"horizon", "cascade"}:
        declaration = stage_config.get("residual_penalty") or {
            "start": 1.0e-2,
            "end": 1.0e-3,
            "start_step": 1,
            "end_step": 5_000,
        }
        residual_ramp = ResidualPenaltyRamp(
            start=float(declaration.get("start", 1.0e-2)),
            end=float(declaration.get("end", 1.0e-3)),
            start_step=int(declaration.get("start_step", 1)),
            end_step=int(declaration.get("end_step", 5_000)),
        )
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    snapshots: dict[str, dict[str, Any]] = {}
    states: dict[int, dict[str, Tensor]] = {}
    all_gradients_finite = True
    minimum = profile.context_samples + profile.output_samples + 1
    if dataset.train_input.size < minimum:
        raise RuntimeError("training source is shorter than one locked window")

    for step in range(1, profile.optimizer_steps + 1):
        phase = scheduler.apply(model, step)
        starts = rng.integers(
            profile.context_samples,
            dataset.train_input.size - profile.output_samples,
            size=profile.batch_size,
        )
        windows = torch.from_numpy(
            np.stack(
                [
                    dataset.train_input[
                        start - profile.context_samples : start + profile.output_samples
                    ]
                    for start in starts
                ]
            )
        ).to(device)
        targets = torch.from_numpy(
            np.stack(
                [
                    dataset.train_target[start : start + profile.output_samples]
                    for start in starts
                ]
            )
        ).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output, core, residual = _components(
            model, model_kind, windows, profile.output_samples
        )
        residual_only = phase.name == "residual_on_frozen_core_error"
        primary_prediction = residual if residual_only else output
        primary_target = targets - core.detach() if residual_only else targets
        if loss_mode == "m4":
            assert mrstft is not None
            mse = torch.nn.functional.mse_loss(primary_prediction, primary_target)
            frequency = mrstft(primary_prediction[:, None], primary_target[:, None])
            primary = mse + 0.0005 * frequency
        else:
            mse = torch.nn.functional.mse_loss(primary_prediction, primary_target)
            frequency = primary_prediction.new_zeros(())
            primary = wright(primary_prediction, primary_target)
        curvature = _curvature(model, output)
        residual_term = normalized_residual_penalty(residual, targets)
        residual_weight = (
            residual_ramp(step)
            if residual_ramp is not None
            else float(m4_config["loss"]["normalized_residual_energy"])
        )
        total = (
            primary
            + float(m4_config["loss"]["spline_curvature"]) * curvature
            + residual_weight * residual_term
        )
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite loss at optimizer step {step}")
        total.backward()
        gradient_norms, gradients_finite = _gradient_norms(model)
        all_gradients_finite = all_gradients_finite and gradients_finite
        if not gradients_finite:
            raise RuntimeError(f"non-finite gradient at optimizer step {step}")
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(m4_config["gradient_clip_norm"])
        )
        optimizer.step()
        record: dict[str, Any] = {
            "step": step,
            "phase": phase.name,
            "residual_trained_on_frozen_core_error": residual_only,
            "loss": float(total.detach()),
            "primary_loss": float(primary.detach()),
            "mse": float(mse.detach()),
            "mrstft": float(frequency.detach()),
            "spline_curvature": float(curvature.detach()),
            "normalized_residual_energy": float(residual_term.detach()),
            "residual_penalty_weight": residual_weight,
            "gradient_norms_by_block": gradient_norms,
            "gradients_finite": gradients_finite,
        }
        history.append(record)

        if step not in profile.checkpoint_steps:
            continue
        model.eval()
        validation_count = min(
            profile.validation_samples, dataset.validation_input.size
        )
        validation_input = dataset.validation_input[:validation_count]
        validation_target = dataset.validation_target[:validation_count]
        validation_prediction = _predict(
            model,
            model_kind,
            validation_input,
            device=device,
            profile=profile,
        )
        validation_metrics = _prediction_metrics(
            validation_prediction, validation_target
        )
        validation_metrics["residual_energy_ratio"] = _residual_ratio(
            model,
            model_kind,
            validation_input,
            device=device,
            block_samples=profile.evaluation_block_samples,
        )
        snapshots[str(step)] = {
            "step": step,
            "phase": phase.name,
            "validation": validation_metrics,
            "gradient_norms_by_block": gradient_norms,
            "gradients_finite": gradients_finite,
            "residual_penalty_weight": residual_weight,
        }
        state = _state_copy(model)
        states[step] = state
        if checkpoint_directory is not None:
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
            torch.save(state, checkpoint_directory / f"step-{step}.pt")

    if set(states) != set(profile.checkpoint_steps):
        raise RuntimeError("not all preregistered checkpoints were produced")
    best_step = min(
        profile.checkpoint_steps,
        key=lambda item: float(snapshots[str(item)]["validation"]["esr"]),
    )
    selected_state = states[best_step]
    model.load_state_dict(selected_state, strict=True)
    if checkpoint_directory is not None:
        shutil.copyfile(
            checkpoint_directory / f"step-{best_step}.pt",
            checkpoint_directory / "selected-model-state.pt",
        )

    selected_count = min(profile.validation_samples, dataset.validation_input.size)
    selected_input = dataset.validation_input[:selected_count]
    selected_target = dataset.validation_target[:selected_count]
    selected_prediction = _predict(
        model,
        model_kind,
        selected_input,
        device=device,
        profile=profile,
    )
    alternate_count = min(selected_input.size, 32_768)
    alternate = _predict(
        model,
        model_kind,
        selected_input[:alternate_count],
        device=device,
        profile=profile,
        block_samples=509,
    )
    block_max_abs = float(
        np.max(np.abs(selected_prediction[:alternate_count] - alternate), initial=0.0)
    )
    if block_max_abs > 2.0e-5:
        raise RuntimeError(f"post-training block parity failed: {block_max_abs:.9g}")
    selected_metrics = _prediction_metrics(selected_prediction, selected_target)
    if selected_count >= 4_096:
        selected_metrics.update(spectral_metrics(selected_prediction, selected_target))
    selected_metrics["residual_energy_ratio"] = _residual_ratio(
        model,
        model_kind,
        selected_input,
        device=device,
        block_samples=profile.evaluation_block_samples,
    )
    selected_residual = _residual_state(model)
    metrics = {
        "schema_version": 1,
        "stage": stage,
        "model": model_name,
        "loss": loss_mode,
        "seed": seed,
        "preflight": profile.preflight,
        "optimizer_steps": profile.optimizer_steps,
        "checkpoint_steps": list(profile.checkpoint_steps),
        "snapshots": snapshots,
        "selected_checkpoint": {
            "step": best_step,
            "validation_esr": snapshots[str(best_step)]["validation"]["esr"],
            "policy": "lowest_validation_esr_at_common_checkpoints",
        },
        "best_validation": snapshots[str(best_step)]["validation"],
        "selected_validation": selected_metrics,
        "checks": {
            "all_gradients_finite": all_gradients_finite,
            "alternate_block_max_abs": block_max_abs,
            "finite_selected_validation_prediction": bool(
                np.isfinite(selected_prediction).all()
            ),
            "sealed_test_opened": False,
            "residual_phase_uses_frozen_core_error": (
                True
                if stage == "factorial"
                else any(
                    item["residual_trained_on_frozen_core_error"] for item in history
                )
            ),
        },
        "theoretical_macs_per_sample": theoretical_macs_per_sample(stage, model_name),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "latency_samples": int(getattr(model, "latency_samples", 0)),
        "training_output_samples_seen": (
            profile.optimizer_steps * profile.batch_size * profile.output_samples
        ),
    }
    return TrainingResult(
        metrics=metrics,
        history=history,
        selected_state=selected_state,
        selected_residual_state=selected_residual,
        selected_validation_prediction=selected_prediction,
    )


def load_promoted_residual(
    checkpoint: Path, *, expected_sha256: str
) -> dict[str, Tensor]:
    """Load a gate-selected RF2047 residual only after digest verification."""
    if sha256_file(checkpoint) != expected_sha256:
        raise RuntimeError("promoted RF2047 residual checkpoint digest mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError("promoted RF2047 residual checkpoint is not a state mapping")
    reference = RF2047Residual().state_dict()
    if set(state) != set(reference):
        raise RuntimeError("promoted residual is not an exact RF2047 state")
    validated: dict[str, Tensor] = {}
    for name, expected in reference.items():
        value = state[name]
        if (
            not isinstance(value, Tensor)
            or value.shape != expected.shape
            or value.dtype != expected.dtype
            or not torch.isfinite(value).all()
        ):
            raise RuntimeError(f"invalid promoted RF2047 tensor: {name}")
        validated[name] = value.detach().clone()
    return validated
