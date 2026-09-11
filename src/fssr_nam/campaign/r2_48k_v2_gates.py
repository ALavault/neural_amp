"""Serialization-aware mechanism gate for FSSR-R2-48K-v2."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from fssr_nam.reporting.json_evidence import (
    VALUE_KEY,
    is_extended_real_object,
)

from .r2_48k_gates import evaluate_mechanism_gate as evaluate_v1_scientific_gate
from .r2_48k_v2 import CAMPAIGN_VERSION


class R248KV2EvidenceError(ValueError):
    """Raised when v2 evidence violates its strict transport contract."""


DECISION_ROW_FIELDS = (
    "asr_db",
    "reference_192khz_asr_db",
    "fundamental_complex_error",
    "latency_samples",
    "internal_sample_rate_hz",
    "residual_energy_ratio",
)


def _walk(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str]]:
    tags: list[tuple[tuple[str, ...], str]] = []
    if is_extended_real_object(value):
        tags.append((path, str(value[VALUE_KEY])))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            tags.extend(_walk(item, (*path, str(key))))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            tags.extend(_walk(item, (*path, str(index))))
    elif isinstance(value, float) and not math.isfinite(value):
        raise R248KV2EvidenceError(
            f"bare non-finite float bypassed canonical encoding at {'/'.join(path)}"
        )
    return tags


def validate_mechanism_evidence(evidence: Any) -> dict[str, Any]:
    """Validate strict transport without evaluating any scientific threshold."""
    if not isinstance(evidence, dict):
        raise R248KV2EvidenceError("v2 mechanism evidence must be an object")
    literals = {
        "campaign_version": CAMPAIGN_VERSION,
        "evidence_encoding": "fssr-extended-real-json-v1",
        "reference_kind": "synthetic_192khz",
        "physical_hardware_reference_used": False,
        "physical_audio_samples_read": 0,
        "parent_numeric_evidence_reused": False,
    }
    for name, expected in literals.items():
        if evidence.get(name) != expected:
            raise R248KV2EvidenceError(f"v2 mechanism {name} must equal {expected!r}")
    rows = evidence.get("rows")
    if not isinstance(rows, list) or len(rows) != 24:
        raise R248KV2EvidenceError("v2 mechanism requires exactly 24 aggregate rows")
    tags = _walk(evidence)
    nan_paths = ["/".join(path) for path, token in tags if token == "nan"]
    if nan_paths:
        raise R248KV2EvidenceError(
            f"NaN diagnostics invalidate v2 evidence: {nan_paths[:3]}"
        )
    for row in rows:
        if not isinstance(row, dict):
            raise R248KV2EvidenceError("v2 mechanism rows must be objects")
        for name in DECISION_ROW_FIELDS:
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise R248KV2EvidenceError(f"decision field {name} must be numeric")
            if not math.isfinite(float(value)):
                raise R248KV2EvidenceError(f"decision field {name} must be finite")
        conditions = row.get("conditions")
        if not isinstance(conditions, list) or len(conditions) != 9:
            raise R248KV2EvidenceError(
                "each v2 mechanism row requires the exact nine-condition grid"
            )
    return {"rows": rows, "extended_real_diagnostics": len(tags)}


def evaluate_mechanism_gate(evidence: Any) -> dict[str, Any]:
    """Apply unchanged v1 scientific thresholds after v2 transport validation."""
    validated = validate_mechanism_evidence(evidence)
    result = evaluate_v1_scientific_gate(validated["rows"])
    result.update(
        {
            "campaign_version": CAMPAIGN_VERSION,
            "evidence_encoding": "fssr-extended-real-json-v1",
            "extended_real_diagnostics": validated["extended_real_diagnostics"],
            "scientific_thresholds_changed_from_v1": False,
        }
    )
    return result
