"""Causal model components and FSSR-NAM variants."""

from .fssr import S1Slow, S2Residual, S3FastSlowResidual
from .residual import FastResidualTCN
from .slow import SlowStateController
from .spline import SmoothHermiteSpline
from .structured import CausalFIR, S0Structured

__all__ = [
    "CausalFIR",
    "FastResidualTCN",
    "S0Structured",
    "S1Slow",
    "S2Residual",
    "S3FastSlowResidual",
    "SlowStateController",
    "SmoothHermiteSpline",
]
