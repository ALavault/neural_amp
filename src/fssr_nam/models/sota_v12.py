"""Frozen model pair for AMP-SOTA-PROTOTYPE-v1.2."""

from __future__ import annotations

from torch import nn

from .prototype import SOTAPrototypeAmplifier, build_sota_prototype_candidate

V12_CANDIDATE = "slow_long_tcn_x2"
V12_ZERO_MODULATION_CONTROL = "slow_long_tcn_x2_zero_modulation"
V12_FAMILIES = (V12_CANDIDATE, V12_ZERO_MODULATION_CONTROL)


def build_sota_v12_model(family: str) -> SOTAPrototypeAmplifier:
    """Build the preregistered pair with identical initialization order."""
    if family not in V12_FAMILIES:
        raise ValueError("unknown AMP-SOTA-PROTOTYPE-v1.2 family")
    model = build_sota_prototype_candidate(
        activation="tanh",
        resampler="kaiser_windowed_sinc",
        slow_control="causal_zero_order_hold",
        profile="max",
        initial_residual_scale=0.5,
    )
    model.family = family
    if family == V12_ZERO_MODULATION_CONTROL:
        model.set_slow_modulation_enabled(False)
        for module in (model.observer, model.slow_control):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    return model


def trainable_parameter_count(model: nn.Module) -> int:
    """Return the parameters that the frozen optimizer is allowed to update."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
