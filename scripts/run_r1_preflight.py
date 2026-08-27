#!/usr/bin/env python3
"""Audit the prospective R1 lineage without unlocking a scientific gate."""

from __future__ import annotations

import json
from pathlib import Path

from fssr_nam.reporting.r1_preflight import (
    validate_external_freeze,
    validate_historical_bytes,
    validate_lock_digest,
    validate_prepared_manifests,
    validate_protocol_counts,
    validate_stage_configs,
    validate_submodule_pins,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    checks = {
        "historical": validate_historical_bytes(ROOT),
        "stage_configs": validate_stage_configs(ROOT),
        "counts": validate_protocol_counts(ROOT),
        "external_freeze": validate_external_freeze(ROOT),
        "submodules": validate_submodule_pins(ROOT),
        "data": validate_prepared_manifests(ROOT),
        "diagnostic_protocol_sha256": validate_lock_digest(ROOT),
    }
    print(json.dumps({"status": "passed", "checks": checks}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
