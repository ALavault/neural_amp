"""Shared differentiable objectives for amplifier-model training."""

from __future__ import annotations

from torch import Tensor


def projection_gain_loss(
    output: Tensor, target: Tensor, epsilon: float = 1.0e-8
) -> Tensor:
    """Penalize signed least-squares projection gain without detaching gradients."""
    if output.shape != target.shape:
        raise ValueError("projection gain arrays must have identical shapes")
    target_energy = target.square().sum()
    projection_gain = (output * target).sum() / (target_energy + epsilon)
    return (projection_gain - 1.0).square()
