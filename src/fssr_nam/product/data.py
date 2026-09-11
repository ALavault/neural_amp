"""Prepared 48 kHz pairs available to the product lane."""

from __future__ import annotations

import json
from pathlib import Path

# Sealed for the scientific lane's single future confirmation (D-PRODUCT-001).
SEALED_DEVICES = ("blackstar_ht1", "blackstar", "ua1176", "ua_6176", "ua6176")


def device_pairs(
    manifest_path: Path, device: str, *, root: Path
) -> dict[str, tuple[Path, Path]]:
    """Return the train/validation/test input and target paths of one device."""
    if any(sealed in device.lower() for sealed in SEALED_DEVICES):
        raise PermissionError(f"{device} is sealed for the scientific lane")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs = {
        entry["split"]: (root / entry["input_path"], root / entry["target_path"])
        for entry in manifest["files"]
        if entry["device"] == device
    }
    if set(pairs) != {"train", "validation", "test"}:
        raise ValueError(f"incomplete prepared pairs for {device}")
    return pairs
