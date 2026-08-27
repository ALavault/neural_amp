"""Loss formulation used by the DAFx-19 recurrent baseline."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def preemphasize(signal: Tensor, coefficient: float = 0.85) -> Tensor:
    """Apply the causal filter ``y[n] = x[n] - coefficient*x[n-1]``."""
    if signal.shape[-1] < 1:
        raise ValueError("audio must contain at least one sample")
    previous = torch.nn.functional.pad(signal[..., :-1], (1, 0))
    return signal - coefficient * previous


def esr_loss(output: Tensor, target: Tensor, epsilon: float = 1.0e-5) -> Tensor:
    """Return the original global error-to-signal ratio."""
    if output.shape != target.shape:
        raise ValueError("output and target shapes differ")
    return (target - output).square().mean() / (target.square().mean() + epsilon)


def dc_loss(output: Tensor, target: Tensor, epsilon: float = 1.0e-5) -> Tensor:
    """Return Wright's normalized per-sequence DC error."""
    if output.shape != target.shape:
        raise ValueError("output and target shapes differ")
    difference = target.mean(dim=-1) - output.mean(dim=-1)
    return difference.square().mean() / (target.square().mean() + epsilon)


class WrightLoss(nn.Module):
    """Weighted pre-emphasized ESR and DC loss from the reference code."""

    def __init__(
        self,
        preemphasis: float = 0.85,
        esr_weight: float = 0.75,
        dc_weight: float = 0.25,
        epsilon: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.preemphasis = preemphasis
        self.esr_weight = esr_weight
        self.dc_weight = dc_weight
        self.epsilon = epsilon

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        emphasized_output = preemphasize(output, self.preemphasis)
        emphasized_target = preemphasize(target, self.preemphasis)
        return self.esr_weight * esr_loss(
            emphasized_output, emphasized_target, self.epsilon
        ) + self.dc_weight * dc_loss(output, target, self.epsilon)
