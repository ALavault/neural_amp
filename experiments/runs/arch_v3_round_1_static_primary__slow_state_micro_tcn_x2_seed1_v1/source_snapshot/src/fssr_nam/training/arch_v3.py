"""State-carrying checkpointed training for AMP-QUALITY-ARCH-v3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from fssr_nam.losses import esr_loss
from fssr_nam.metrics.time import correlation, error_to_signal_ratio, gain_error
from fssr_nam.models import build_arch_v3_candidate
from fssr_nam.training.arch_v1 import delay_audio


def projection_gain_loss(
    output: Tensor, target: Tensor, epsilon: float = 1.0e-8
) -> Tensor:
    """Penalize signed least-squares projection gain without detaching gradients."""
    if output.shape != target.shape:
        raise ValueError("projection gain arrays must have identical shapes")
    target_energy = target.square().sum()
    projection_gain = (output * target).sum() / (target_energy + epsilon)
    return (projection_gain - 1.0).square()


@dataclass(frozen=True)
class ArchV3TrainingResult:
    family: str
    system: str
    seed: int
    initial_residual_scale: float
    updates: int
    chunk_samples: int
    total_training_samples: int
    history: tuple[dict[str, object], ...]
    checkpoints: tuple[dict[str, object], ...]


def _as_tensor(value: Tensor | NDArray[np.floating], *, device: torch.device) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.from_numpy(np.asarray(value))
    if tensor.ndim != 2 or not torch.isfinite(tensor).all():
        raise ValueError("v3 training arrays must be finite with shape (episodes,time)")
    return tensor.to(device=device, dtype=torch.float32)


def evaluate_arch_v3_model(
    model: torch.nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    scored_start: int,
) -> dict[str, float]:
    """Evaluate complete episodes independently after the frozen pre-roll."""
    if inputs.shape != targets.shape or inputs.ndim != 2:
        raise ValueError("v3 evaluation arrays must be paired episodes")
    if not 0 <= scored_start < inputs.shape[-1]:
        raise ValueError("v3 scored start is outside the episode")
    model.eval()
    predictions: list[NDArray[np.float32]] = []
    references: list[NDArray[np.float32]] = []
    saturated = 0
    drive_samples = 0

    def capture_drive(
        _module: torch.nn.Module, _inputs: object, output: Tensor
    ) -> None:
        nonlocal saturated, drive_samples
        saturated += int(torch.count_nonzero(output.detach().abs() > 2.0).cpu())
        drive_samples += output.numel()

    hook = model.branch.output_projection.register_forward_hook(capture_drive)
    latency = int(getattr(model, "latency_samples", 0))
    try:
        with torch.inference_mode():
            for episode in range(len(inputs)):
                prediction = model(inputs[episode : episode + 1])[0, scored_start:]
                reference = delay_audio(targets[episode : episode + 1], latency)[
                    0, scored_start:
                ]
                predictions.append(prediction.detach().cpu().numpy())
                references.append(reference.detach().cpu().numpy())
    finally:
        hook.remove()
    prediction_array = np.concatenate(predictions)
    reference_array = np.concatenate(references)
    return {
        "esr": error_to_signal_ratio(prediction_array, reference_array),
        "gain_error": gain_error(prediction_array, reference_array),
        "correlation": correlation(prediction_array, reference_array),
        "prediction_rms": float(np.sqrt(np.mean(np.square(prediction_array)))),
        "target_rms": float(np.sqrt(np.mean(np.square(reference_array)))),
        "residual_drive_abs_gt_2_fraction": saturated / max(drive_samples, 1),
        "dry_gain": float(model.dry_gain.detach().cpu()),
        "residual_scale": float(model.residual_scale.detach().cpu()),
    }


def train_arch_v3_trajectory(
    *,
    family: str,
    system: str,
    train_inputs: Tensor | NDArray[np.floating],
    train_targets: Tensor | NDArray[np.floating],
    internal_dev_inputs: Tensor | NDArray[np.floating],
    internal_dev_targets: Tensor | NDArray[np.floating],
    profile: str,
    initial_residual_scale: float,
    checkpoints: tuple[int, ...],
    scored_start: int,
    chunk_samples: int,
    learning_rate: float,
    projection_gain_weight: float,
    device: torch.device,
    seed: int,
) -> tuple[
    torch.nn.Module,
    ArchV3TrainingResult,
    dict[int, dict[str, Tensor]],
]:
    """Train deterministic truncated sequences while carrying causal state."""
    if not checkpoints or tuple(sorted(set(checkpoints))) != checkpoints:
        raise ValueError("v3 checkpoints must be unique and strictly increasing")
    if chunk_samples < 1 or learning_rate <= 0.0 or projection_gain_weight < 0.0:
        raise ValueError("v3 optimization parameters are invalid")
    train_x = _as_tensor(train_inputs, device=device)
    train_y = _as_tensor(train_targets, device=device)
    dev_x = _as_tensor(internal_dev_inputs, device=device)
    dev_y = _as_tensor(internal_dev_targets, device=device)
    if train_x.shape != train_y.shape or dev_x.shape != dev_y.shape:
        raise ValueError("v3 inputs and targets must be paired")
    if not 0 < scored_start < train_x.shape[-1]:
        raise ValueError("v3 scored start is outside the training episode")
    if dev_x.shape[-1] != train_x.shape[-1]:
        raise ValueError("v3 train and INTERNAL_DEV episode lengths differ")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_arch_v3_candidate(
        family,
        profile=profile,
        initial_residual_scale=initial_residual_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.0
    )
    delayed_targets = delay_audio(train_y, int(model.latency_samples))
    checkpoint_set = set(checkpoints)
    history: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    checkpoint_states: dict[int, dict[str, Tensor]] = {}
    episode = 0
    position = scored_start

    def prime_episode() -> None:
        model.reset_state()
        with torch.no_grad():
            model.stream(train_x[episode : episode + 1, :scored_start])

    prime_episode()
    model.train()
    for update in range(1, checkpoints[-1] + 1):
        optimizer.zero_grad(set_to_none=True)
        remaining = chunk_samples
        outputs: list[Tensor] = []
        targets: list[Tensor] = []
        while remaining:
            take = min(remaining, train_x.shape[-1] - position)
            stop = position + take
            outputs.append(model.stream(train_x[episode : episode + 1, position:stop]))
            targets.append(delayed_targets[episode : episode + 1, position:stop])
            model.detach_stream_state()
            position = stop
            remaining -= take
            if position == train_x.shape[-1]:
                episode = (episode + 1) % len(train_x)
                position = scored_start
                prime_episode()
        output = torch.cat(outputs, dim=-1)
        target = torch.cat(targets, dim=-1)
        audio_loss = esr_loss(output, target)
        gain_loss = projection_gain_loss(output, target)
        total = audio_loss + projection_gain_weight * gain_loss
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite v3 loss at update {update}")
        total.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise RuntimeError(f"non-finite v3 gradient at update {update}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update in checkpoint_set:
            training_metrics = evaluate_arch_v3_model(
                model, train_x, train_y, scored_start=scored_start
            )
            internal_dev_metrics = evaluate_arch_v3_model(
                model, dev_x, dev_y, scored_start=scored_start
            )
            checkpoint_rows.append(
                {
                    "update": update,
                    "training": training_metrics,
                    "internal_dev": internal_dev_metrics,
                }
            )
            history.append(
                {
                    "update": update,
                    "last_chunk": {
                        "total": float(total.detach().cpu()),
                        "audio_esr": float(audio_loss.detach().cpu()),
                        "projection_gain_loss": float(gain_loss.detach().cpu()),
                        "gradient_norm": float(gradient_norm.detach().cpu()),
                        "samples": output.shape[-1],
                        "episode_segments": len(outputs),
                    },
                    "training": training_metrics,
                    "internal_dev": internal_dev_metrics,
                }
            )
            checkpoint_states[update] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            model.train()
    return (
        model,
        ArchV3TrainingResult(
            family=family,
            system=system,
            seed=seed,
            initial_residual_scale=initial_residual_scale,
            updates=checkpoints[-1],
            chunk_samples=chunk_samples,
            total_training_samples=checkpoints[-1] * chunk_samples,
            history=tuple(history),
            checkpoints=tuple(checkpoint_rows),
        ),
        checkpoint_states,
    )
