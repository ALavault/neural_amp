"""Causal model components and FSSR-NAM variants."""

from .spline import SmoothHermiteSpline
from .structured import CausalFIR, S0Structured

__all__ = ["CausalFIR", "S0Structured", "SmoothHermiteSpline"]
