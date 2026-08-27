"""Faithful training helpers for the pinned DAFx-19 recurrent baseline."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

from fssr_nam.losses import WrightLoss, esr_loss
from fssr_nam.models.wright import WrightLSTM

LSTMState = tuple[Tensor, Tensor]


def frame_audio(signal: np.ndarray, frame_samples: int) -> Tensor:
    """Split a mono file into complete non-overlapping frames, dropping its tail."""
    if signal.ndim != 1 or frame_samples < 1:
        raise ValueError("signal must be mono and frame_samples must be positive")
    frame_count = signal.size // frame_samples
    if frame_count < 1:
        raise ValueError("signal is shorter than one frame")
    trimmed = np.ascontiguousarray(signal[: frame_count * frame_samples])
    return torch.from_numpy(trimmed.reshape(frame_count, frame_samples))


def train_epoch(
    model: WrightLSTM,
    input_frames: Tensor,
    target_frames: Tensor,
    loss_function: WrightLoss,
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    warmup_samples: int,
    tbptt_samples: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[float, int]:
    """Run the reference shuffled-frame and truncated-BPTT update sequence."""
    if input_frames.shape != target_frames.shape or input_frames.ndim != 2:
        raise ValueError("training frames must be paired two-dimensional tensors")
    if not 0 <= warmup_samples < input_frames.shape[-1]:
        raise ValueError("warmup_samples is outside the frame")
    permutation = torch.randperm(len(input_frames), generator=generator)
    total_loss = 0.0
    update_count = 0
    batch_count = 0
    model.train()

    for batch_start in range(0, len(permutation), batch_size):
        indices = permutation[batch_start : batch_start + batch_size]
        inputs = input_frames[indices].to(device)
        targets = target_frames[indices].to(device)
        state: LSTMState | None = None
        if warmup_samples:
            _, state = model.forward_with_state(inputs[:, :warmup_samples], state)
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        chunk_count = 0
        for start in range(warmup_samples, inputs.shape[-1], tbptt_samples):
            stop = min(start + tbptt_samples, inputs.shape[-1])
            output, state = model.forward_with_state(inputs[:, start:stop], state)
            loss = loss_function(output, targets[:, start:stop])
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite Wright training loss")
            loss.backward()
            if not all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise RuntimeError("non-finite Wright training gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state = tuple(value.detach() for value in state)
            value = float(loss.detach())
            total_loss += value
            batch_loss += value
            update_count += 1
            chunk_count += 1
        if chunk_count < 1 or not np.isfinite(batch_loss):
            raise RuntimeError("empty or non-finite Wright training batch")
        batch_count += 1

    if batch_count < 1 or update_count < 1:
        raise RuntimeError("Wright epoch produced no optimizer update")
    return total_loss / update_count, update_count


def predict_streaming(
    model: WrightLSTM,
    signal: np.ndarray,
    *,
    device: torch.device,
    chunk_samples: int = 32768,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Predict a complete file with causal state carried between bounded chunks."""
    if signal.ndim != 1 or chunk_samples < 1:
        raise ValueError("signal must be mono and chunk_samples must be positive")
    model.eval()
    model.reset_state()
    output = np.empty(signal.shape, dtype=np.float32)
    try:
        with torch.inference_mode():
            for start in range(0, signal.size, chunk_samples):
                stop = min(start + chunk_samples, signal.size)
                chunk = torch.from_numpy(np.ascontiguousarray(signal[start:stop])).to(
                    device
                )
                output[start:stop] = model.stream(chunk).cpu().numpy()
                if progress is not None:
                    progress(stop, signal.size)
    finally:
        model.reset_state()
    if not np.isfinite(output).all():
        raise RuntimeError("non-finite Wright prediction")
    return output


def evaluate_prediction(
    prediction: np.ndarray, target: np.ndarray, loss_function: WrightLoss
) -> tuple[float, float]:
    """Return reference loss and plain ESR without post-inference correction."""
    prediction_tensor = torch.from_numpy(prediction)
    target_tensor = torch.from_numpy(np.asarray(target, dtype=np.float32))
    reference_loss = float(loss_function(prediction_tensor, target_tensor))
    plain_esr = float(esr_loss(prediction_tensor, target_tensor))
    return reference_loss, plain_esr
