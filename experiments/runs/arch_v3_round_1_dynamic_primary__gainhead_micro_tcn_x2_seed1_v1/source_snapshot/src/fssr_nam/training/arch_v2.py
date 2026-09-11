"""Checkpointed training for AMP-COMPETENCE-ARCH-v2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from fssr_nam.losses import esr_loss
from fssr_nam.models import build_arch_v1_candidate
from fssr_nam.training.arch_v1 import (
    _forward_components,
    delay_audio,
    evaluate_mechanism_model,
    wet_feature_targets,
)


@dataclass(frozen=True)
class CheckpointedTrainingResult:
    family: str
    system: str
    seed: int
    auxiliary_weight: float
    updates: int
    history: tuple[dict[str, float | int], ...]
    checkpoints: tuple[dict[str, object], ...]


def train_checkpointed_trajectory(
    *,
    family: str,
    system: str,
    train_inputs: Tensor,
    train_targets: Tensor,
    validation_inputs: Tensor,
    validation_targets: Tensor,
    profile: str,
    checkpoints: tuple[int, ...],
    tail_samples: int,
    learning_rate: float,
    auxiliary_weight: float,
    device: torch.device,
    seed: int,
) -> tuple[torch.nn.Module, CheckpointedTrainingResult, dict[int, dict[str, Tensor]]]:
    """Train one uninterrupted trajectory and capture every frozen checkpoint."""
    if train_inputs.shape != train_targets.shape or train_inputs.ndim != 2:
        raise ValueError("training arrays must be paired episodes")
    if validation_inputs.shape != validation_targets.shape:
        raise ValueError("validation arrays must be paired episodes")
    if not checkpoints or tuple(sorted(set(checkpoints))) != checkpoints:
        raise ValueError("checkpoints must be unique and strictly increasing")
    if checkpoints[0] < 1 or learning_rate <= 0.0:
        raise ValueError("checkpoints and learning rate must be positive")
    if not 1 <= tail_samples <= train_inputs.shape[-1]:
        raise ValueError("tail_samples is outside the training episode")
    if auxiliary_weight < 0.0:
        raise ValueError("auxiliary weight must be non-negative")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_arch_v1_candidate(family, profile=profile).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.0
    )
    device_inputs = train_inputs.to(device)
    device_targets = delay_audio(
        train_targets, int(getattr(model, "latency_samples", 0))
    ).to(device)
    device_auxiliary_targets = wet_feature_targets(train_targets).to(device)
    validation_inputs_device = validation_inputs.to(device)
    validation_targets_device = validation_targets.to(device)
    checkpoint_set = set(checkpoints)
    history: list[dict[str, float | int]] = []
    checkpoint_rows: list[dict[str, object]] = []
    checkpoint_states: dict[int, dict[str, Tensor]] = {}
    for update in range(1, checkpoints[-1] + 1):
        episode = (update - 1) % len(device_inputs)
        inputs = device_inputs[episode : episode + 1]
        targets = device_targets[episode : episode + 1]
        wet_features = device_auxiliary_targets[episode : episode + 1]
        optimizer.zero_grad(set_to_none=True)
        audio, state_prediction = _forward_components(model, inputs)
        audio_loss = esr_loss(audio[..., -tail_samples:], targets[..., -tail_samples:])
        auxiliary_loss = torch.nn.functional.mse_loss(
            state_prediction[..., -tail_samples:], wet_features[..., -tail_samples:]
        )
        total = audio_loss + auxiliary_weight * auxiliary_loss
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite v2 loss at update {update}")
        total.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise RuntimeError(f"non-finite v2 gradient at update {update}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update == 1 or update % 100 == 0 or update in checkpoint_set:
            history.append(
                {
                    "update": update,
                    "total": float(total.detach().cpu()),
                    "audio_esr": float(audio_loss.detach().cpu()),
                    "auxiliary_mse": float(auxiliary_loss.detach().cpu()),
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                }
            )
        if update in checkpoint_set:
            validation = evaluate_mechanism_model(
                model,
                validation_inputs_device,
                validation_targets_device,
                tail_samples=tail_samples,
            )
            checkpoint_rows.append({"update": update, "validation": validation})
            checkpoint_states[update] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            model.train()
    return (
        model,
        CheckpointedTrainingResult(
            family=family,
            system=system,
            seed=seed,
            auxiliary_weight=auxiliary_weight,
            updates=checkpoints[-1],
            history=tuple(history),
            checkpoints=tuple(checkpoint_rows),
        ),
        checkpoint_states,
    )
