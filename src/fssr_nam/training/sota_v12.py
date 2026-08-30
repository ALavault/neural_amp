"""Prospective synthetic training for AMP-SOTA-PROTOTYPE-v1.2."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from fssr_nam.metrics.sota_confirmation import aligned_source_metrics
from fssr_nam.metrics.time import (
    correlation,
    error_to_signal_ratio,
    gain_error,
    mean_absolute_error,
)
from fssr_nam.models.sota_v12 import V12_FAMILIES, build_sota_v12_model
from fssr_nam.training.objectives import projection_gain_loss

FloatArray = NDArray[np.float32]
ModelBuilder = Callable[[str], nn.Module]
CheckpointCallback = Callable[[int, Mapping[str, object]], None]


class SotaV12StabilityError(RuntimeError):
    """Raised for a finite-value failure that counts as scientific evidence."""


class SotaV12ResourceLimitError(RuntimeError):
    """Raised when a preregistered per-trajectory wall-clock limit is reached."""


@dataclass(frozen=True)
class SotaV12TrainingResult:
    family: str
    system: str
    seed: int
    updates: int
    snapshot_updates: tuple[int, ...]
    evaluation_updates: tuple[int, ...]
    chunk_samples: int
    common_preroll_samples_after_alignment: int
    scored_start: int
    scored_samples_per_episode: int
    total_training_samples: int
    elapsed_seconds: float
    peak_cuda_memory_bytes: int
    history: tuple[dict[str, object], ...]
    checkpoints: tuple[dict[str, object], ...]


def _as_tensor(value: Tensor | NDArray[np.floating], *, device: torch.device) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.from_numpy(np.asarray(value))
    if tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[-1] < 1:
        raise ValueError("v1.2 arrays must have shape (episodes,time)")
    if not torch.isfinite(tensor).all():
        raise ValueError("v1.2 arrays must be finite")
    return tensor.to(device=device, dtype=torch.float32)


def _delay_audio(signal: Tensor, samples: int) -> Tensor:
    if samples < 0:
        raise ValueError("audio delay must be non-negative")
    if samples == 0:
        return signal
    output = torch.zeros_like(signal)
    if samples < signal.shape[-1]:
        output[..., samples:] = signal[..., :-samples]
    return output


def _scored_pair(
    prediction: FloatArray,
    target: FloatArray,
    *,
    latency_samples: int,
    common_preroll_samples_after_alignment: int,
) -> tuple[FloatArray, FloatArray]:
    samples = min(prediction.size - latency_samples, target.size)
    if samples <= common_preroll_samples_after_alignment + 4_096:
        raise ValueError("v1.2 episode is too short for scoring")
    scored_prediction = prediction[
        latency_samples + common_preroll_samples_after_alignment : latency_samples
        + samples
    ]
    scored_target = target[common_preroll_samples_after_alignment:samples]
    return scored_prediction, scored_target


def _model_scalar(model: nn.Module, name: str) -> float:
    value = getattr(model, name, None)
    if value is None:
        value = getattr(model.branch, name)
    if not isinstance(value, Tensor) or value.numel() != 1:
        raise TypeError(f"v1.2 model {name} must be a scalar tensor")
    return float(value.detach().cpu())


def evaluate_sota_v12_model(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    common_preroll_samples_after_alignment: int,
    include_spectral: bool,
) -> dict[str, object]:
    """Evaluate independent episodes with declared latency and no fitted correction."""
    if inputs.shape != targets.shape or inputs.ndim != 2:
        raise ValueError("v1.2 evaluation arrays must be paired episodes")
    latency = getattr(model, "latency_samples", None)
    if isinstance(latency, bool) or not isinstance(latency, int) or latency < 0:
        raise TypeError("v1.2 model latency must be a non-negative integer")
    if common_preroll_samples_after_alignment < 0:
        raise ValueError("v1.2 common pre-roll must be non-negative")

    model.eval()
    source_rows: list[dict[str, float | int | bool]] = []
    scored_predictions: list[FloatArray] = []
    scored_targets: list[FloatArray] = []
    with torch.inference_mode():
        for episode in range(len(inputs)):
            prediction = model(inputs[episode : episode + 1])[0]
            if (
                prediction.shape != inputs[episode].shape
                or not torch.isfinite(prediction).all()
            ):
                raise SotaV12StabilityError(
                    "v1.2 evaluation produced non-finite or malformed audio"
                )
            prediction_array = np.asarray(
                prediction.detach().cpu().numpy(), dtype=np.float32
            )
            target_array = np.asarray(
                targets[episode].detach().cpu().numpy(), dtype=np.float32
            )
            scored_prediction, scored_target = _scored_pair(
                prediction_array,
                target_array,
                latency_samples=latency,
                common_preroll_samples_after_alignment=(
                    common_preroll_samples_after_alignment
                ),
            )
            row: dict[str, float | int | bool] = {
                "episode": episode,
                "esr": error_to_signal_ratio(scored_prediction, scored_target),
                "mae": mean_absolute_error(scored_prediction, scored_target),
                "gain_error": gain_error(scored_prediction, scored_target),
                "correlation": correlation(scored_prediction, scored_target),
                "prediction_rms": float(
                    np.sqrt(np.mean(np.square(scored_prediction), dtype=np.float64))
                ),
                "target_rms": float(
                    np.sqrt(np.mean(np.square(scored_target), dtype=np.float64))
                ),
                "scored_samples": int(scored_target.size),
                "spectral_metrics_included": include_spectral,
            }
            if include_spectral:
                full = aligned_source_metrics(
                    prediction_array,
                    target_array,
                    latency_samples=latency,
                    preroll_samples=common_preroll_samples_after_alignment,
                )
                row["log_mel"] = float(full["log_mel"])
                row["mrstft"] = float(full["mrstft"])
            source_rows.append(row)
            scored_predictions.append(scored_prediction)
            scored_targets.append(scored_target)

    prediction_all = np.concatenate(scored_predictions)
    target_all = np.concatenate(scored_targets)
    aggregate: dict[str, float | int] = {
        "esr": error_to_signal_ratio(prediction_all, target_all),
        "mae": mean_absolute_error(prediction_all, target_all),
        "gain_error": gain_error(prediction_all, target_all),
        "correlation": correlation(prediction_all, target_all),
        "prediction_rms": float(
            np.sqrt(np.mean(np.square(prediction_all), dtype=np.float64))
        ),
        "target_rms": float(np.sqrt(np.mean(np.square(target_all), dtype=np.float64))),
        "scored_samples": int(target_all.size),
    }
    median_metrics = ("esr", "mae", "gain_error", "correlation")
    if include_spectral:
        median_metrics = (*median_metrics, "log_mel", "mrstft")
    source_median = {
        metric: float(np.median([float(row[metric]) for row in source_rows]))
        for metric in median_metrics
    }
    return {
        "aggregate": aggregate,
        "source_median": source_median,
        "sources": source_rows,
        "latency_samples": latency,
        "common_preroll_samples_after_alignment": (
            common_preroll_samples_after_alignment
        ),
        "dry_gain": _model_scalar(model, "dry_gain"),
        "residual_scale": _model_scalar(model, "residual_scale"),
    }


def train_sota_v12_trajectory(
    *,
    family: str,
    system: str,
    train_inputs: Tensor | NDArray[np.floating],
    train_targets: Tensor | NDArray[np.floating],
    internal_dev_inputs: Tensor | NDArray[np.floating],
    internal_dev_targets: Tensor | NDArray[np.floating],
    snapshot_updates: Sequence[int],
    evaluation_updates: Sequence[int],
    common_preroll_samples_after_alignment: int,
    chunk_samples: int,
    learning_rate: float,
    projection_gain_weight: float,
    gradient_clip_norm: float,
    maximum_elapsed_seconds: float,
    device: torch.device,
    seed: int,
    model_builder: ModelBuilder = build_sota_v12_model,
    checkpoint_callback: CheckpointCallback | None = None,
) -> tuple[nn.Module, SotaV12TrainingResult, dict[int, dict[str, Tensor]]]:
    """Train one uninterrupted, deterministic, state-carrying v1.2 trajectory."""
    if family not in V12_FAMILIES:
        raise ValueError("unknown v1.2 family")
    snapshots = tuple(snapshot_updates)
    evaluations = tuple(evaluation_updates)
    if not snapshots or tuple(sorted(set(snapshots))) != snapshots:
        raise ValueError("v1.2 snapshot updates must be unique and ordered")
    if not evaluations or tuple(sorted(set(evaluations))) != evaluations:
        raise ValueError("v1.2 evaluation updates must be unique and ordered")
    if not set(evaluations) <= set(snapshots):
        raise ValueError("v1.2 evaluations require saved snapshots")
    if seed not in {0, 1, 2} or isinstance(seed, bool):
        raise ValueError("v1.2 seed is outside the frozen schedule")
    if (
        chunk_samples < 1
        or common_preroll_samples_after_alignment < 0
        or learning_rate <= 0.0
        or projection_gain_weight < 0.0
        or gradient_clip_norm <= 0.0
        or isinstance(maximum_elapsed_seconds, bool)
        or not isinstance(maximum_elapsed_seconds, (int, float))
        or not np.isfinite(maximum_elapsed_seconds)
        or maximum_elapsed_seconds <= 0.0
    ):
        raise ValueError("v1.2 optimization parameters are invalid")

    train_x = _as_tensor(train_inputs, device=device)
    train_y = _as_tensor(train_targets, device=device)
    dev_x = _as_tensor(internal_dev_inputs, device=device)
    dev_y = _as_tensor(internal_dev_targets, device=device)
    if train_x.shape != train_y.shape or dev_x.shape != dev_y.shape:
        raise ValueError("v1.2 inputs and targets must be paired")
    if train_x.shape[-1] != dev_x.shape[-1]:
        raise ValueError("v1.2 train and INTERNAL_DEV episode lengths differ")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    model = model_builder(family).to(device)
    latency = getattr(model, "latency_samples", None)
    if isinstance(latency, bool) or not isinstance(latency, int) or latency < 0:
        raise TypeError("v1.2 model latency must be a non-negative integer")
    scored_start = latency + common_preroll_samples_after_alignment
    if scored_start >= train_x.shape[-1] - 4_096:
        raise ValueError("v1.2 pre-roll leaves an insufficient scored region")

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("v1.2 model has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    delayed_targets = _delay_audio(train_y, latency)
    snapshot_set = set(snapshots)
    evaluation_set = set(evaluations)
    checkpoint_states: dict[int, dict[str, Tensor]] = {}
    checkpoint_rows: list[dict[str, object]] = []
    history: list[dict[str, object]] = []
    episode = 0
    position = scored_start
    peak_cuda_memory_bytes = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def prime_episode() -> None:
        model.reset_state()
        with torch.no_grad():
            primed = model.stream(train_x[episode : episode + 1, :scored_start])
        if primed.shape[-1] != scored_start or not torch.isfinite(primed).all():
            raise SotaV12StabilityError("v1.2 pre-roll produced invalid audio")
        model.detach_stream_state()

    started = time.perf_counter()

    def enforce_time_limit() -> None:
        if time.perf_counter() - started >= maximum_elapsed_seconds:
            raise SotaV12ResourceLimitError(
                "v1.2 trajectory reached its frozen wall-clock limit"
            )

    prime_episode()
    model.train()
    for update in range(1, snapshots[-1] + 1):
        enforce_time_limit()
        optimizer.zero_grad(set_to_none=True)
        remaining = chunk_samples
        outputs: list[Tensor] = []
        references: list[Tensor] = []
        while remaining:
            take = min(remaining, train_x.shape[-1] - position)
            stop = position + take
            output = model.stream(train_x[episode : episode + 1, position:stop])
            if output.shape[-1] != take or not torch.isfinite(output).all():
                raise SotaV12StabilityError(
                    f"v1.2 stream output failed at update {update}"
                )
            outputs.append(output)
            references.append(delayed_targets[episode : episode + 1, position:stop])
            model.detach_stream_state()
            position = stop
            remaining -= take
            if position == train_x.shape[-1]:
                episode = (episode + 1) % len(train_x)
                position = scored_start
                prime_episode()

        output = torch.cat(outputs, dim=-1)
        target = torch.cat(references, dim=-1)
        target_energy = target.square().sum().clamp_min(1.0e-8)
        audio_esr = (output - target).square().sum() / target_energy
        gain_penalty = projection_gain_loss(output, target)
        total = audio_esr + projection_gain_weight * gain_penalty
        if not torch.isfinite(total):
            raise SotaV12StabilityError(
                f"v1.2 loss became non-finite at update {update}"
            )
        total.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in parameters
        ):
            raise SotaV12StabilityError(
                f"v1.2 gradients became non-finite at update {update}"
            )
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
        if not torch.isfinite(gradient_norm):
            raise SotaV12StabilityError(
                f"v1.2 gradient norm became non-finite at update {update}"
            )
        optimizer.step()

        if update == 1 or update in snapshot_set:
            row: dict[str, object] = {
                "update": update,
                "last_chunk": {
                    "total": float(total.detach().cpu()),
                    "audio_esr": float(audio_esr.detach().cpu()),
                    "projection_gain_loss": float(gain_penalty.detach().cpu()),
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    "samples": int(output.shape[-1]),
                    "episode_segments": len(outputs),
                },
            }
            if update in evaluation_set:
                evaluation = evaluate_sota_v12_model(
                    model,
                    dev_x,
                    dev_y,
                    common_preroll_samples_after_alignment=(
                        common_preroll_samples_after_alignment
                    ),
                    include_spectral=update == evaluations[-1],
                )
                checkpoint_rows.append({"update": update, "internal_dev": evaluation})
                row["internal_dev"] = evaluation
                model.train()
            history.append(row)
            if checkpoint_callback is not None:
                checkpoint_callback(update, row)
            enforce_time_limit()
        if update in snapshot_set:
            checkpoint_states[update] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    elapsed_seconds = time.perf_counter() - started
    if device.type == "cuda":
        peak_cuda_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    return (
        model,
        SotaV12TrainingResult(
            family=family,
            system=system,
            seed=seed,
            updates=snapshots[-1],
            snapshot_updates=snapshots,
            evaluation_updates=evaluations,
            chunk_samples=chunk_samples,
            common_preroll_samples_after_alignment=(
                common_preroll_samples_after_alignment
            ),
            scored_start=scored_start,
            scored_samples_per_episode=train_x.shape[-1] - scored_start,
            total_training_samples=snapshots[-1] * chunk_samples,
            elapsed_seconds=elapsed_seconds,
            peak_cuda_memory_bytes=peak_cuda_memory_bytes,
            history=tuple(history),
            checkpoints=tuple(checkpoint_rows),
        ),
        checkpoint_states,
    )
