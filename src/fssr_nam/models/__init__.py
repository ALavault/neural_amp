"""Causal model components and FSSR-NAM variants."""

from .fssr import S1Slow, S2Residual, S3FastSlowResidual, S4Antialiased
from .oversampling import LocalOversampledSpline2x
from .residual import FastResidualTCN
from .slow import SlowStateController
from .spline import SmoothHermiteSpline
from .structured import CausalFIR, S0Structured

__all__ = [
    "CausalFIR",
    "FastResidualTCN",
    "LocalOversampledSpline2x",
    "S0Structured",
    "S1Slow",
    "S2Residual",
    "S3FastSlowResidual",
    "S4Antialiased",
    "SlowStateController",
    "SmoothHermiteSpline",
]
