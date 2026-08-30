"""Metadata-only public-data audit for SOTA-PROTOTYPE-v1.1."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

REQUIRED_DEVICE_TOKENS = {
    "fulltone": "Fulltone",
    "bigmuff": "Big Muff",
    "blackstar": "Blackstar",
    "ua1176": "1176",
}


class SotaDataAuditError(ValueError):
    """Selected public metadata cannot support the frozen research use."""


def audit_sota_data_metadata(root: Path) -> dict[str, Any]:
    """Audit YAML metadata only; never open an archive or audio member."""
    data = yaml.safe_load(
        (root / "configs/data/r1_physical.yaml").read_text(encoding="utf-8")
    )
    catalog = yaml.safe_load(
        (root / "datasets/manifests/catalog.yaml").read_text(encoding="utf-8")
    )
    if not isinstance(data, Mapping) or not isinstance(catalog, Mapping):
        raise SotaDataAuditError("data metadata must be mappings")
    resources = catalog.get("resources")
    if not isinstance(resources, list):
        raise SotaDataAuditError("catalog resources must be a list")
    devices = data.get("devices")
    if not isinstance(devices, Mapping):
        raise SotaDataAuditError("physical devices must be a mapping")
    rows = []
    for device, token in REQUIRED_DEVICE_TOKENS.items():
        declaration = devices.get(device)
        if not isinstance(declaration, Mapping):
            raise SotaDataAuditError(f"missing device metadata: {device}")
        matches = [
            resource
            for resource in resources
            if isinstance(resource, Mapping) and token in str(resource.get("name", ""))
        ]
        if len(matches) != 1:
            raise SotaDataAuditError(f"ambiguous catalog resource: {device}")
        resource = matches[0]
        if resource.get("license") != "CC-BY-NC-4.0":
            raise SotaDataAuditError(f"unexpected dataset license: {device}")
        if "research" not in str(resource.get("allowed_usage", "")).lower():
            raise SotaDataAuditError(f"research use is not declared: {device}")
        splits = declaration.get("splits")
        if not isinstance(splits, Mapping) or set(splits) != {
            "train",
            "validation",
            "test",
        }:
            raise SotaDataAuditError(f"incomplete split metadata: {device}")
        source_sets: dict[str, set[str]] = {}
        for split, items in splits.items():
            if not isinstance(items, list) or not items:
                raise SotaDataAuditError(f"empty split metadata: {device}/{split}")
            source_sets[str(split)] = {str(item.get("source_id", "")) for item in items}
            if "" in source_sets[str(split)]:
                raise SotaDataAuditError(f"missing source ID: {device}/{split}")
        if any(
            source_sets[first] & source_sets[second]
            for first, second in (
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            )
        ):
            raise SotaDataAuditError(f"source groups overlap: {device}")
        rows.append(
            {
                "device": device,
                "license": resource["license"],
                "source_ids_by_split": {
                    split: sorted(values) for split, values in source_sets.items()
                },
            }
        )
    return {
        "format": "sota-public-data-metadata-audit-v1",
        "passed": True,
        "research_only": True,
        "commercial_claim_supported": False,
        "audio_files_read": 0,
        "devices": rows,
    }
