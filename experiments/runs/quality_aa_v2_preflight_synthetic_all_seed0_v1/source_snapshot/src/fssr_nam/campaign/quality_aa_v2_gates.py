"""QUALITY-AA-v2 bindings for the unchanged scientific gates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .quality_aa_gates import (
    QualityAAGateEvidenceError,
)
from .quality_aa_gates import (
    evaluate_mechanism_gate as _evaluate_mechanism_gate,
)
from .quality_aa_gates import (
    evaluate_native_gate as _evaluate_native_gate,
)
from .quality_aa_gates import (
    evaluate_preflight_gate as _evaluate_preflight_gate,
)
from .quality_aa_gates import (
    final_backend_decision as _final_backend_decision,
)
from .quality_aa_v2 import CAMPAIGN_VERSION


def evaluate_preflight_gate(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return _evaluate_preflight_gate(evidence, campaign_version=CAMPAIGN_VERSION)


def evaluate_mechanism_gate(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return _evaluate_mechanism_gate(evidence, campaign_version=CAMPAIGN_VERSION)


def evaluate_native_gate(
    evidence: Mapping[str, Any], mechanism_gate: Mapping[str, Any]
) -> dict[str, Any]:
    return _evaluate_native_gate(
        evidence, mechanism_gate, campaign_version=CAMPAIGN_VERSION
    )


def final_backend_decision(
    mechanism_gate: Mapping[str, Any], native_gate: Mapping[str, Any]
) -> dict[str, Any]:
    return _final_backend_decision(
        mechanism_gate, native_gate, campaign_version=CAMPAIGN_VERSION
    )


__all__ = [
    "QualityAAGateEvidenceError",
    "evaluate_mechanism_gate",
    "evaluate_native_gate",
    "evaluate_preflight_gate",
    "final_backend_decision",
]
