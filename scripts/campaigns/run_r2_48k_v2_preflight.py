#!/usr/bin/env python3
"""Validate and record the R2-48K-v2 serialization-only preflight."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fssr_nam.campaign.r2_48k_v2 import CAMPAIGN_VERSION, validate_repository_configs
from fssr_nam.campaign.r2_48k_v2_registry import append_gate_event
from fssr_nam.reporting.json_evidence import (
    dumps_strict_evidence,
    encode_extended_reals,
    loads_strict_json,
)

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / ".codex_campaign/r2_48k_v2"


def _write_new_json(path: Path, payload: Any) -> None:
    serialized = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == serialized:
            return
        raise RuntimeError(f"refusing to overwrite immutable v2 preflight: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)


def _update_maturity() -> None:
    path = CAMPAIGN_DIR / "MATURITY.json"
    maturity = json.loads(path.read_text(encoding="utf-8"))
    maturity["current_stage"] = "data"
    maturity["gates"]["preflight"] = "passed"
    maturity["gates"]["data"] = "pending"
    maturity["status"] = "preflight_passed_data_audit_pending"
    serialized = json.dumps(maturity, allow_nan=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    resolved = validate_repository_configs(ROOT)
    transport_probe = {
        "exact_zero_db": float("-inf"),
        "unbounded_ratio": float("inf"),
        "finite": [-1.0, 0.0, 1.0],
    }
    serialized_probe = dumps_strict_evidence(transport_probe, indent=None)
    parsed_probe = loads_strict_json(serialized_probe)
    if parsed_probe != encode_extended_reals(transport_probe):
        raise RuntimeError("v2 strict-JSON canonical roundtrip changed evidence")
    parent_preflight = json.loads(
        (ROOT / ".codex_campaign/r2_48k/PREFLIGHT.json").read_text(encoding="utf-8")
    )
    if parent_preflight.get("status") != "passed":
        raise RuntimeError("parent model preflight is not a passing setup fact")
    report = {
        "format": "fssr-r2-48k-v2-preflight-v1",
        "campaign_version": CAMPAIGN_VERSION,
        "status": "passed",
        "amendment_scope": "evidence_serialization_only",
        "resolved_protocol_campaign": resolved["campaign_version"],
        "scientific_thresholds_changed": False,
        "fixtures_modes_probes_changed": False,
        "strict_json_roundtrip_passed": True,
        "canonical_extended_real_format": "fssr-extended-real-json-v1",
        "hypothesis_property_test_required": True,
        "parent_model_checks_reused_without_architecture_change": True,
        "model_checks": parent_preflight["model_checks"],
        "parent_numeric_mechanism_evidence_reused": False,
        "physical_waveform_samples_read": 0,
        "internal_validation_test_opened": False,
        "external_report_only_locked": True,
        "fm9_proxy_used": False,
    }
    path = CAMPAIGN_DIR / "PREFLIGHT.json"
    _write_new_json(path, report)
    append_gate_event(
        CAMPAIGN_DIR / "GATE_LEDGER.jsonl",
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "preflight",
            "status": "passed",
            "evidence_path": str(path.relative_to(ROOT)),
        },
    )
    _update_maturity()
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
