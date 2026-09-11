"""Read campaign status across historical and lineage-specific schemas."""

from __future__ import annotations

import json
from pathlib import Path


def _boolean(value: object) -> str:
    return str(bool(value)).lower()


def format_status(root: Path) -> str:
    """Format old and lineage-specific campaign schemas without guessing access."""
    active_path = root / "ACTIVE_CAMPAIGN"
    if active_path.exists():
        active = active_path.read_text(encoding="utf-8").strip()
        active_root = root / active
        maturity = json.loads(
            (active_root / "MATURITY.json").read_text(encoding="utf-8")
        )
        freeze = json.loads(
            (active_root / "EXTERNAL_FREEZE.json").read_text(encoding="utf-8")
        )
        fields = [
            f"campaign={active}",
            f"maturity={maturity.get('current_level', maturity.get('current_stage'))}",
            f"status={maturity['status']}",
        ]
        if "external_retest_authorized" in freeze:
            fields.append(
                "external_retest_authorized="
                f"{_boolean(freeze['external_retest_authorized'])}"
            )
        if "internal_validation_outputs_locked" in freeze:
            fields.append(
                "internal_validation_outputs_locked="
                f"{_boolean(freeze['internal_validation_outputs_locked'])}"
            )
        if "external_report_only_locked" in freeze:
            fields.append(
                "external_report_only_locked="
                f"{_boolean(freeze['external_report_only_locked'])}"
            )
        for name in ("blackstar_accessed", "ua1176_accessed"):
            if name in freeze:
                fields.append(f"{name}={_boolean(freeze[name])}")
        return " ".join(fields)
    maturity = json.loads((root / "MATURITY.json").read_text(encoding="utf-8"))
    freeze = json.loads((root / "EXTERNAL_FREEZE.json").read_text(encoding="utf-8"))
    return (
        f"maturity={maturity['current_level']} "
        f"status={maturity['status']} "
        "external_retest_authorized="
        f"{_boolean(freeze['external_retest_authorized'])}"
    )
