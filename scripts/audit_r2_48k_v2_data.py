#!/usr/bin/env python3
"""Reuse the immutable metadata-only v1 data audit for R2-48K-v2."""

from __future__ import annotations

import json
from pathlib import Path

from fssr_nam.campaign.r2_48k_v2 import (
    CAMPAIGN_VERSION,
    validate_repository_configs,
    validate_stage_authorization,
)
from fssr_nam.campaign.r2_48k_v2_registry import append_gate_event, gate_decisions

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / ".codex_campaign/r2_48k_v2"


def _write_new_json(path: Path, payload: dict) -> None:
    serialized = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == serialized:
            return
        raise RuntimeError(f"refusing to overwrite immutable v2 data audit: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)


def _update_maturity() -> None:
    path = CAMPAIGN_DIR / "MATURITY.json"
    maturity = json.loads(path.read_text(encoding="utf-8"))
    maturity["current_stage"] = "mechanism"
    maturity["gates"]["data"] = "passed"
    maturity["gates"]["mechanism"] = "pending_measurement_evidence"
    maturity["status"] = "ready_for_single_v2_synthetic_mechanism_run"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(maturity, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    validate_repository_configs(ROOT)
    ledger = CAMPAIGN_DIR / "GATE_LEDGER.jsonl"
    validate_stage_authorization("data", gate_decisions(ledger))
    parent = json.loads(
        (ROOT / ".codex_campaign/r2_48k/DATA_AUDIT.json").read_text(encoding="utf-8")
    )
    required = {
        "status": "passed",
        "metadata_only": True,
        "waveform_samples_read": False,
        "file_pairs": 18,
        "sample_rate_hz": 48_000,
        "physical_192khz_reference_available": False,
        "fm9_proxy_used": False,
        "internal_validation_test_waveforms_opened": False,
        "external_report_only_locked": True,
    }
    for name, expected in required.items():
        if parent.get(name) != expected:
            raise RuntimeError(f"parent data audit {name} changed")
    report = dict(parent)
    report.update(
        {
            "format": "fssr-r2-48k-v2-data-audit-v1",
            "campaign_version": CAMPAIGN_VERSION,
            "source_evidence_path": ".codex_campaign/r2_48k/DATA_AUDIT.json",
            "parent_metadata_reused": True,
            "parent_waveform_data_read": False,
            "v2_waveform_samples_read": False,
            "parent_numeric_mechanism_evidence_reused": False,
        }
    )
    path = CAMPAIGN_DIR / "DATA_AUDIT.json"
    _write_new_json(path, report)
    append_gate_event(
        ledger,
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "data",
            "status": "passed",
            "evidence_path": str(path.relative_to(ROOT)),
        },
    )
    _update_maturity()
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
