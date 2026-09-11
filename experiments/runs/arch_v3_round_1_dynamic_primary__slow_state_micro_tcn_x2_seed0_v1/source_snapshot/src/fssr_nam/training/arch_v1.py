"""Training primitives for the prospective AMP-QUALITY-ARCH-v1 campaign."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from fssr_nam.losses import esr_loss
from fssr_nam.metrics.time import correlation, error_to_signal_ratio, gain_error
from fssr_nam.models import CausalBlockFeatureBus, build_arch_v1_candidate


@dataclass(frozen=True)
class MechanismTrainingResult:
    family: str
    system: str
    auxiliary_weight: float
    updates: int
    history: tuple[dict[str, float | int], ...]
    validation: dict[str, float]


def delay_audio(signal: Tensor, samples: int) -> Tensor:
    """Delay batched audio by an exact number of samples without wraparound."""
    if signal.ndim != 2:
        raise ValueError("audio must have shape (batch,time)")
    if samples < 0:
        raise ValueError("delay must be non-negative")
    if samples == 0:
        return signal.clone()
    if samples >= signal.shape[-1]:
        return torch.zeros_like(signal)
    return torch.nn.functional.pad(signal, (samples, 0))[..., :-samples]


def wet_feature_targets(target: Tensor, decimation: int = 64) -> Tensor:
    """Compute target-only descriptors used solely as auxiliary supervision."""
    if target.ndim != 2:
        raise ValueError("target audio must have shape (batch,time)")
    with torch.no_grad():
        return CausalBlockFeatureBus(decimation=decimation)(target)


def _forward_components(
    model: torch.nn.Module, inputs: Tensor
) -> tuple[Tensor, Tensor]:
    forward_components = getattr(model, "forward_components", None)
    if forward_components is None:
        output = model(inputs)
        return output, output.new_zeros((len(output), 6, output.shape[-1]))
    result = forward_components(inputs)
    return result.audio, result.state_prediction


def evaluate_mechanism_model(
    model: torch.nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    tail_samples: int,
) -> dict[str, float]:
    """Evaluate episodes independently with the frozen latency convention."""
    if inputs.shape != targets.shape or inputs.ndim != 2:
        raise ValueError("mechanism evaluation arrays must be paired episodes")
    if not 1 <= tail_samples <= inputs.shape[-1]:
        raise ValueError("tail_samples is outside the mechanism episode")
    model.eval()
    predictions: list[NDArray[np.float32]] = []
    references: list[NDArray[np.float32]] = []
    latency = int(getattr(model, "latency_samples", 0))
    with torch.inference_mode():
        for episode in range(len(inputs)):
            prediction = model(inputs[episode : episode + 1])[0, -tail_samples:]
            reference = delay_audio(targets[episode : episode + 1], latency)[
                0, -tail_samples:
            ]
            predictions.append(prediction.detach().cpu().numpy())
            references.append(reference.detach().cpu().numpy())
    prediction_array = np.concatenate(predictions)
    reference_array = np.concatenate(references)
    return {
        "esr": error_to_signal_ratio(prediction_array, reference_array),
        "gain_error": gain_error(prediction_array, reference_array),
        "correlation": correlation(prediction_array, reference_array),
        "prediction_rms": float(np.sqrt(np.mean(np.square(prediction_array)))),
        "target_rms": float(np.sqrt(np.mean(np.square(reference_array)))),
    }


def train_mechanism_model(
    *,
    family: str,
    system: str,
    train_inputs: Tensor,
    train_targets: Tensor,
    validation_inputs: Tensor,
    validation_targets: Tensor,
    profile: str,
    updates: int,
    tail_samples: int,
    learning_rate: float,
    auxiliary_weight: float,
    device: torch.device,
    seed: int,
) -> tuple[torch.nn.Module, MechanismTrainingResult]:
    """Fit one deterministic synthetic mechanism trajectory."""
    if train_inputs.shape != train_targets.shape or train_inputs.ndim != 2:
        raise ValueError("training arrays must be paired episodes")
    if validation_inputs.shape != validation_targets.shape:
        raise ValueError("validation arrays must be paired episodes")
    if updates < 1 or learning_rate <= 0.0:
        raise ValueError("updates and learning rate must be positive")
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
    delayed_targets = delay_audio(
        train_targets, int(getattr(model, "latency_samples", 0))
    )
    auxiliary_targets = wet_feature_targets(train_targets)
    history: list[dict[str, float | int]] = []
    for update in range(1, updates + 1):
        episode = (update - 1) % len(train_inputs)
        inputs = train_inputs[episode : episode + 1].to(device)
        targets = delayed_targets[episode : episode + 1].to(device)
        wet_features = auxiliary_targets[episode : episode + 1].to(device)
        optimizer.zero_grad(set_to_none=True)
        audio, state_prediction = _forward_components(model, inputs)
        audio_loss = esr_loss(audio[..., -tail_samples:], targets[..., -tail_samples:])
        auxiliary_loss = torch.nn.functional.mse_loss(
            state_prediction[..., -tail_samples:],
            wet_features[..., -tail_samples:],
        )
        total = audio_loss + auxiliary_weight * auxiliary_loss
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite mechanism loss at update {update}")
        total.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise RuntimeError(f"non-finite mechanism gradient at update {update}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update == 1 or update % 50 == 0 or update == updates:
            history.append(
                {
                    "update": update,
                    "total": float(total.detach().cpu()),
                    "audio_esr": float(audio_loss.detach().cpu()),
                    "auxiliary_mse": float(auxiliary_loss.detach().cpu()),
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                }
            )
    validation = evaluate_mechanism_model(
        model,
        validation_inputs.to(device),
        validation_targets.to(device),
        tail_samples=tail_samples,
    )
    return model, MechanismTrainingResult(
        family=family,
        system=system,
        auxiliary_weight=auxiliary_weight,
        updates=updates,
        history=tuple(history),
        validation=validation,
    )
