"""Strict development-only training support for the SOTA prototype."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from fssr_nam.campaign.amp_sota_prototype_v1 import (
    CAMPAIGN_VERSION,
    DEVELOPMENT_DEVICES,
)
from fssr_nam.models.arch_v1 import DEPLOYMENT_PROFILES
from fssr_nam.models.prototype import (
    PROTOTYPE_ACTIVATIONS,
    PROTOTYPE_RESAMPLERS,
    PROTOTYPE_SLOW_CONTROLS,
    build_sota_prototype_candidate,
)
from fssr_nam.training.objectives import projection_gain_loss

CHECKPOINT_FORMAT = "fssr-nam-sota-prototype-checkpoint-v1"
CHECKPOINT_SCHEMA_VERSION = 1
REGISTERED_CHECKPOINT_UPDATES = (500, 1_000, 2_000, 5_000, 10_000, 15_000)
REGISTERED_SEEDS = (0, 1, 2)
TARGET_SAMPLE_RATE = 48_000


class PrototypeDataBoundaryError(RuntimeError):
    """Raised before any audio read that would cross the development boundary."""


class PrototypeCheckpointError(RuntimeError):
    """Raised when a checkpoint is incomplete, incompatible, or divergent."""


@dataclass(frozen=True)
class PrototypeModelConfig:
    """Everything required to reconstruct one prototype graph."""

    activation: str = "tanh"
    resampler: str = "kaiser_windowed_sinc"
    slow_control: str = "causal_zero_order_hold"
    profile: str = "max"
    initial_residual_scale: float = 0.5

    def __post_init__(self) -> None:
        if self.activation not in PROTOTYPE_ACTIVATIONS:
            raise ValueError("unknown prototype activation")
        if self.resampler not in PROTOTYPE_RESAMPLERS:
            raise ValueError("unknown prototype resampler")
        if self.slow_control not in PROTOTYPE_SLOW_CONTROLS:
            raise ValueError("unknown prototype slow control")
        if self.profile not in DEPLOYMENT_PROFILES:
            raise ValueError("unknown prototype deployment profile")
        if (
            not math.isfinite(self.initial_residual_scale)
            or self.initial_residual_scale <= 1.0e-4
        ):
            raise ValueError("initial residual scale must be finite and above floor")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PrototypeModelConfig:
        expected = {
            "activation",
            "resampler",
            "slow_control",
            "profile",
            "initial_residual_scale",
        }
        if set(value) != expected:
            raise PrototypeCheckpointError("checkpoint model_config fields changed")
        try:
            return cls(
                activation=value["activation"],
                resampler=value["resampler"],
                slow_control=value["slow_control"],
                profile=value["profile"],
                initial_residual_scale=value["initial_residual_scale"],
            )
        except (TypeError, ValueError) as error:
            raise PrototypeCheckpointError(
                f"invalid checkpoint model configuration: {error}"
            ) from error

    def build(self) -> nn.Module:
        return build_sota_prototype_candidate(**asdict(self))


@dataclass(frozen=True)
class DevelopmentSource:
    """One indivisible physical source file and its paired target."""

    physical_device: str
    split: str
    source_id: str
    input: NDArray[np.float32]
    target: NDArray[np.float32]
    sample_rate_hz: int = TARGET_SAMPLE_RATE

    def __post_init__(self) -> None:
        if self.physical_device not in DEVELOPMENT_DEVICES:
            raise PrototypeDataBoundaryError(
                "prototype development accepts Fulltone/BigMuff only"
            )
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("unknown physical split")
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if self.sample_rate_hz != TARGET_SAMPLE_RATE:
            raise ValueError("prototype sources must be 48 kHz")
        if (
            self.input.ndim != 1
            or self.input.shape != self.target.shape
            or self.input.size < 1
        ):
            raise ValueError("physical source must be a nonempty paired mono file")
        if not np.all(np.isfinite(self.input)) or not np.all(np.isfinite(self.target)):
            raise ValueError("physical source contains non-finite samples")


@dataclass(frozen=True)
class PrototypeTrainingResult:
    physical_device: str
    seed: int
    updates: int
    chunks_processed: int
    source_resets: int
    source_ids: tuple[str, ...]
    history: tuple[dict[str, float | int | str], ...]


@dataclass(frozen=True)
class LoadedPrototypeCheckpoint:
    model: nn.Module
    model_config: PrototypeModelConfig
    training: dict[str, Any]


def _relative_audio_path(root: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise PrototypeDataBoundaryError("manifest audio path must be relative text")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise PrototypeDataBoundaryError("manifest audio path must be relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise PrototypeDataBoundaryError("manifest audio path leaves repository root")
    return resolved


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PrototypeDataBoundaryError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise PrototypeDataBoundaryError(f"{label} must be a JSON object")
    return value


def _selected_rows(
    manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    *,
    physical_device: str,
    split: str,
) -> list[dict[str, Any]]:
    if physical_device not in DEVELOPMENT_DEVICES:
        raise PrototypeDataBoundaryError(
            "prototype development accepts Fulltone/BigMuff only"
        )
    if split not in {"train", "validation", "test"}:
        raise PrototypeDataBoundaryError("unknown development split")
    if manifest.get("name") != "r1_physical":
        raise PrototypeDataBoundaryError("unexpected physical manifest")
    if manifest.get("target_sample_rate") != TARGET_SAMPLE_RATE:
        raise PrototypeDataBoundaryError("physical manifest sample rate changed")
    if manifest.get("external_report_only_accessed") is not False:
        raise PrototypeDataBoundaryError(
            "physical manifest crossed the sealed boundary"
        )
    if split_manifest.get("leakage_check") != "passed":
        raise PrototypeDataBoundaryError("physical split leakage check is not passed")
    groups = split_manifest.get("groups")
    if not isinstance(groups, dict) or not isinstance(
        groups.get(physical_device), dict
    ):
        raise PrototypeDataBoundaryError("development split groups are missing")
    device_groups = groups[physical_device]
    if set(device_groups) != {"train", "validation", "test"}:
        raise PrototypeDataBoundaryError("development split declaration is incomplete")
    flattened: list[str] = []
    for name in ("train", "validation", "test"):
        values = device_groups[name]
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(item, str) and item for item in values)
        ):
            raise PrototypeDataBoundaryError("development source groups are invalid")
        flattened.extend(values)
    if len(flattened) != len(set(flattened)):
        raise PrototypeDataBoundaryError("source file appears in multiple splits")

    files = manifest.get("files")
    if not isinstance(files, list):
        raise PrototypeDataBoundaryError("physical manifest files are missing")
    selected: dict[str, dict[str, Any]] = {}
    for raw_row in files:
        if not isinstance(raw_row, dict):
            raise PrototypeDataBoundaryError("physical manifest row must be an object")
        if raw_row.get("device") != physical_device or raw_row.get("split") != split:
            continue
        source_id = raw_row.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise PrototypeDataBoundaryError("development source_id is invalid")
        if source_id in selected:
            raise PrototypeDataBoundaryError("duplicate development source row")
        if raw_row.get("tier") != "INTERNAL_DEV":
            raise PrototypeDataBoundaryError(
                "development row has a sealed evidence tier"
            )
        if raw_row.get("sample_rate") != TARGET_SAMPLE_RATE:
            raise PrototypeDataBoundaryError("development row sample rate changed")
        prefix = f"datasets/raw/internal_r1/{physical_device}/{split}/"
        if not all(
            isinstance(raw_row.get(key), str) and raw_row[key].startswith(prefix)
            for key in ("input_path", "target_path")
        ):
            raise PrototypeDataBoundaryError(
                "development row path is outside its canonical device/split"
            )
        selected[source_id] = raw_row
    expected_ids = device_groups[split]
    if set(selected) != set(expected_ids):
        raise PrototypeDataBoundaryError("manifest and split source files disagree")
    return [selected[source_id] for source_id in expected_ids]


def load_development_sources(
    root: Path,
    manifest_path: Path,
    split_manifest_path: Path,
    *,
    physical_device: str,
    split: str,
) -> tuple[DevelopmentSource, ...]:
    """Load only authorized dev rows, preserving every source-file boundary."""
    manifest = _read_json_mapping(manifest_path, "physical manifest")
    split_manifest = _read_json_mapping(split_manifest_path, "split manifest")
    rows = _selected_rows(
        manifest,
        split_manifest,
        physical_device=physical_device,
        split=split,
    )
    sources: list[DevelopmentSource] = []
    for row in rows:
        input_path = _relative_audio_path(root, row.get("input_path"))
        target_path = _relative_audio_path(root, row.get("target_path"))
        input_signal, input_rate = sf.read(input_path, dtype="float32")
        target_signal, target_rate = sf.read(target_path, dtype="float32")
        if input_rate != TARGET_SAMPLE_RATE or target_rate != TARGET_SAMPLE_RATE:
            raise PrototypeDataBoundaryError("development WAV sample rate changed")
        expected_samples = row.get("samples")
        if (
            isinstance(expected_samples, bool)
            or not isinstance(expected_samples, int)
            or expected_samples < 1
        ):
            raise PrototypeDataBoundaryError("development sample count is invalid")
        if (
            len(input_signal) != expected_samples
            or len(target_signal) != expected_samples
        ):
            raise PrototypeDataBoundaryError("development WAV length changed")
        sources.append(
            DevelopmentSource(
                physical_device=physical_device,
                split=split,
                source_id=row["source_id"],
                input=np.asarray(input_signal, dtype=np.float32),
                target=np.asarray(target_signal, dtype=np.float32),
            )
        )
    return tuple(sources)


def _aligned_stream_target(history: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    latency = history.shape[-1]
    if latency == 0:
        return target, history
    combined = torch.cat((history, target), dim=-1)
    return combined[..., : target.shape[-1]], combined[..., -latency:].detach()


def _reset_model(model: nn.Module) -> None:
    reset = getattr(model, "reset", None)
    if not callable(reset):
        reset = getattr(model, "reset_state", None)
    if not callable(reset):
        raise TypeError("prototype model must provide reset() or reset_state()")
    reset()


def train_development_model(
    model: nn.Module,
    sources: Sequence[DevelopmentSource],
    *,
    optimizer: torch.optim.Optimizer,
    compute_device: torch.device,
    seed: int,
    updates: int,
    checkpoint_updates: Sequence[int],
    chunk_samples: int,
    common_preroll_samples: int,
    projection_gain_weight: float,
    gradient_clip_norm: float,
    checkpoint_callback: Callable[[int, nn.Module], None] | None = None,
) -> PrototypeTrainingResult:
    """Train with TBPTT while resetting exactly at every source-file boundary."""
    if not sources:
        raise ValueError("at least one development source is required")
    physical_devices = {source.physical_device for source in sources}
    if len(physical_devices) != 1 or not physical_devices <= set(DEVELOPMENT_DEVICES):
        raise PrototypeDataBoundaryError(
            "one authorized development device is required"
        )
    if any(source.split != "train" for source in sources):
        raise PrototypeDataBoundaryError("training accepts the train split only")
    if isinstance(seed, bool) or seed not in REGISTERED_SEEDS:
        raise ValueError("seed is outside the registered development schedule")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
        raise ValueError("updates must be a positive integer")
    checkpoints = tuple(checkpoint_updates)
    if tuple(sorted(set(checkpoints))) != checkpoints or any(
        update < 1 or update > updates for update in checkpoints
    ):
        raise ValueError("checkpoint updates must be unique, ordered, and in range")
    if chunk_samples < 1 or common_preroll_samples < 0:
        raise ValueError("chunk and preroll sizes are invalid")
    if projection_gain_weight < 0.0 or gradient_clip_norm <= 0.0:
        raise ValueError("optimization weights are invalid")
    latency = getattr(model, "latency_samples", None)
    if isinstance(latency, bool) or not isinstance(latency, int) or latency < 0:
        raise TypeError("prototype model must expose a non-negative integer latency")
    if any(source.input.size <= latency + common_preroll_samples for source in sources):
        raise ValueError("a training source is shorter than the scored region")
    for method in ("stream", "detach_stream_state"):
        if not callable(getattr(model, method, None)):
            raise TypeError(f"prototype model must provide {method}()")

    torch.manual_seed(seed)
    if compute_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    model.to(compute_device)
    model.train()
    update = 0
    chunks_processed = 0
    source_resets = 0
    history_rows: list[dict[str, float | int | str]] = []
    checkpoint_set = set(checkpoints)
    while update < updates:
        for source in sources:
            _reset_model(model)
            source_resets += 1
            target_history = torch.zeros((1, latency), device=compute_device)
            position = 0
            while position < source.input.size and update < updates:
                stop = min(position + chunk_samples, source.input.size)
                input_chunk = torch.from_numpy(source.input[position:stop]).to(
                    device=compute_device
                )[None]
                target_chunk = torch.from_numpy(source.target[position:stop]).to(
                    device=compute_device
                )[None]
                aligned_target, target_history = _aligned_stream_target(
                    target_history, target_chunk
                )
                scored_start = max(0, latency + common_preroll_samples - position)
                optimizer.zero_grad(set_to_none=True)
                if scored_start >= input_chunk.shape[-1]:
                    with torch.no_grad():
                        output = model.stream(input_chunk)
                    if output.shape != input_chunk.shape:
                        raise RuntimeError("prototype stream output shape changed")
                    model.detach_stream_state()
                    position = stop
                    chunks_processed += 1
                    continue

                output = model.stream(input_chunk)
                if (
                    output.shape != input_chunk.shape
                    or not torch.isfinite(output).all()
                ):
                    raise RuntimeError("prototype stream produced invalid audio")
                scored_output = output[..., scored_start:]
                scored_target = aligned_target[..., scored_start:]
                target_energy = scored_target.square().sum().clamp_min(1.0e-8)
                audio_esr = (
                    scored_output - scored_target
                ).square().sum() / target_energy
                gain_penalty = projection_gain_loss(scored_output, scored_target)
                total = audio_esr + projection_gain_weight * gain_penalty
                if not torch.isfinite(total):
                    raise RuntimeError(
                        f"non-finite prototype loss at update {update + 1}"
                    )
                total.backward()
                if not all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise RuntimeError(
                        f"non-finite prototype gradient at update {update + 1}"
                    )
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm
                )
                optimizer.step()
                model.detach_stream_state()
                update += 1
                chunks_processed += 1
                if update == 1 or update in checkpoint_set or update == updates:
                    history_rows.append(
                        {
                            "update": update,
                            "source_id": source.source_id,
                            "audio_esr": float(audio_esr.detach().cpu()),
                            "projection_gain": float(gain_penalty.detach().cpu()),
                            "total": float(total.detach().cpu()),
                            "gradient_norm": float(gradient_norm.detach().cpu()),
                        }
                    )
                if update in checkpoint_set and checkpoint_callback is not None:
                    checkpoint_callback(update, model)
                position = stop
            if update >= updates:
                break

    return PrototypeTrainingResult(
        physical_device=next(iter(physical_devices)),
        seed=seed,
        updates=update,
        chunks_processed=chunks_processed,
        source_resets=source_resets,
        source_ids=tuple(source.source_id for source in sources),
        history=tuple(history_rows),
    )


def build_checkpoint_payload(
    model: nn.Module,
    model_config: PrototypeModelConfig,
    *,
    physical_device: str,
    seed: int,
    update: int,
    source_ids: Sequence[str],
) -> dict[str, Any]:
    """Build the exact reconstructible v1 checkpoint payload."""
    if physical_device not in DEVELOPMENT_DEVICES:
        raise PrototypeDataBoundaryError("checkpoint device is not INTERNAL_DEV")
    if isinstance(seed, bool) or seed not in REGISTERED_SEEDS:
        raise PrototypeCheckpointError("checkpoint seed is not registered")
    if isinstance(update, bool) or update not in REGISTERED_CHECKPOINT_UPDATES:
        raise PrototypeCheckpointError("checkpoint update is not registered")
    normalized_sources = list(source_ids)
    if (
        not normalized_sources
        or len(normalized_sources) != len(set(normalized_sources))
        or not all(isinstance(item, str) and item for item in normalized_sources)
    ):
        raise PrototypeCheckpointError("checkpoint source_ids are invalid")
    state_dict = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    reference = model_config.build()
    expected_attributes = {
        "activation_name": model_config.activation,
        "resampler_name": model_config.resampler,
        "slow_control_name": model_config.slow_control,
        "channels": DEPLOYMENT_PROFILES[model_config.profile][0],
    }
    if any(
        getattr(model, name, None) != value
        for name, value in expected_attributes.items()
    ):
        raise PrototypeCheckpointError(
            "model attributes do not match the declared checkpoint graph"
        )
    reference_state = reference.state_dict()
    if set(state_dict) != set(reference_state) or any(
        state_dict[name].shape != reference_state[name].shape for name in state_dict
    ):
        raise PrototypeCheckpointError(
            "model state does not match the declared checkpoint graph"
        )
    latency = getattr(model, "latency_samples", None)
    if latency != getattr(reference, "latency_samples", None):
        raise PrototypeCheckpointError(
            "model latency does not match the declared checkpoint graph"
        )
    if not all(torch.isfinite(value).all() for value in state_dict.values()):
        raise PrototypeCheckpointError("refusing non-finite model state")
    return {
        "format": CHECKPOINT_FORMAT,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "campaign_version": CAMPAIGN_VERSION,
        "model_family": "slow_long_tcn_x2",
        "sample_rate_hz": TARGET_SAMPLE_RATE,
        "latency_samples": latency,
        "model_config": asdict(model_config),
        "training": {
            "physical_device": physical_device,
            "seed": seed,
            "source_ids": normalized_sources,
            "split": "train",
            "update": update,
        },
        "state_dict": state_dict,
    }


def write_checkpoint_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one immutable checkpoint; an existing path is never replaced."""
    validate_checkpoint_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(dict(payload), stream)


def validate_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    """Validate the complete checkpoint envelope before graph construction."""
    expected = {
        "format",
        "schema_version",
        "campaign_version",
        "model_family",
        "sample_rate_hz",
        "latency_samples",
        "model_config",
        "training",
        "state_dict",
    }
    if set(payload) != expected:
        raise PrototypeCheckpointError("checkpoint envelope fields changed")
    fixed = {
        "format": CHECKPOINT_FORMAT,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "campaign_version": CAMPAIGN_VERSION,
        "model_family": "slow_long_tcn_x2",
        "sample_rate_hz": TARGET_SAMPLE_RATE,
    }
    for key, expected_value in fixed.items():
        if payload.get(key) != expected_value:
            raise PrototypeCheckpointError(f"checkpoint {key} changed")
    latency = payload.get("latency_samples")
    if (
        isinstance(latency, bool)
        or not isinstance(latency, int)
        or not 0 <= latency <= 64
    ):
        raise PrototypeCheckpointError("checkpoint latency_samples is invalid")
    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise PrototypeCheckpointError("checkpoint model_config must be a mapping")
    PrototypeModelConfig.from_mapping(model_config)
    training = payload.get("training")
    if not isinstance(training, Mapping) or set(training) != {
        "physical_device",
        "seed",
        "source_ids",
        "split",
        "update",
    }:
        raise PrototypeCheckpointError("checkpoint training fields changed")
    if training.get("physical_device") not in DEVELOPMENT_DEVICES:
        raise PrototypeCheckpointError("checkpoint crossed the development boundary")
    if training.get("seed") not in REGISTERED_SEEDS or isinstance(
        training.get("seed"), bool
    ):
        raise PrototypeCheckpointError("checkpoint seed is not registered")
    if training.get("update") not in REGISTERED_CHECKPOINT_UPDATES or isinstance(
        training.get("update"), bool
    ):
        raise PrototypeCheckpointError("checkpoint update is not registered")
    if training.get("split") != "train":
        raise PrototypeCheckpointError("checkpoint split changed")
    source_ids = training.get("source_ids")
    if (
        not isinstance(source_ids, list)
        or not source_ids
        or len(source_ids) != len(set(source_ids))
        or not all(isinstance(item, str) and item for item in source_ids)
    ):
        raise PrototypeCheckpointError("checkpoint source_ids are invalid")
    state_dict = payload.get("state_dict")
    if (
        not isinstance(state_dict, Mapping)
        or not state_dict
        or not all(
            isinstance(key, str) and isinstance(value, Tensor)
            for key, value in state_dict.items()
        )
    ):
        raise PrototypeCheckpointError("checkpoint state_dict is invalid")
    if not all(torch.isfinite(value).all() for value in state_dict.values()):
        raise PrototypeCheckpointError("checkpoint state_dict contains non-finite data")


def load_prototype_checkpoint(
    path: Path, *, map_location: str | torch.device = "cpu"
) -> LoadedPrototypeCheckpoint:
    """Load a strict v1 checkpoint and reconstruct its exact model graph."""
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise PrototypeCheckpointError(
            f"cannot load prototype checkpoint: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise PrototypeCheckpointError("checkpoint payload must be a mapping")
    validate_checkpoint_payload(payload)
    model_config = PrototypeModelConfig.from_mapping(payload["model_config"])
    model = model_config.build().to(map_location)
    if model.latency_samples != payload["latency_samples"]:
        raise PrototypeCheckpointError(
            "checkpoint latency does not match the reconstructed graph"
        )
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as error:
        raise PrototypeCheckpointError(
            f"checkpoint state_dict does not match the declared graph: {error}"
        ) from error
    _reset_model(model)
    return LoadedPrototypeCheckpoint(
        model=model,
        model_config=model_config,
        training=dict(payload["training"]),
    )
