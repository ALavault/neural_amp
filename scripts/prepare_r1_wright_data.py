#!/usr/bin/env python3
"""Validate and manifest the exact native-rate DAFx-19 Big Muff files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/data/r1_wright_bigmuff_native.yaml"
MANIFEST_PATH = ROOT / "datasets/manifests/r1_wright_bigmuff_native.json"
SPLIT_PATH = ROOT / "datasets/splits/r1_wright_bigmuff_native.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(samples: np.ndarray) -> dict[str, int | float]:
    return {
        "samples": int(samples.size),
        "peak": float(np.max(np.abs(samples), initial=0.0)),
        "rms": float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))),
        "nonfinite": int(np.count_nonzero(~np.isfinite(samples))),
        "clipped_samples": int(np.count_nonzero(np.abs(samples) >= 0.999)),
    }


def copy_once(source: Path, destination: Path, expected: str) -> None:
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise RuntimeError(f"official source checksum mismatch: {source}")
    if destination.exists():
        if destination.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite divergent data: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


def write_json_once(path: Path, payload: dict[str, object]) -> None:
    serialized = json.dumps(payload, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(f"refusing to overwrite divergent manifest: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding="utf-8")


def main() -> None:
    config_bytes = CONFIG_PATH.read_bytes()
    config = yaml.safe_load(config_bytes)
    if config["campaign_version"] != "FSSR-R1-v1":
        raise RuntimeError("unexpected R1 campaign version")
    archive_path = ROOT / config["archive_path"]
    if sha256(archive_path) != config["archive_sha256"]:
        raise RuntimeError("Big Muff licensing archive checksum mismatch")
    reference_path = ROOT / config["reference_model"]["path"]
    if sha256(reference_path) != config["reference_model"]["sha256"]:
        raise RuntimeError("published Wright model checksum mismatch")
    expected_rate = int(config["sample_rate"])
    files: list[dict[str, object]] = []
    groups: dict[str, list[str]] = {}

    for split_name, split in config["splits"].items():
        input_path = ROOT / split["input_path"]
        target_path = ROOT / split["target_path"]
        copy_once(ROOT / split["input_source"], input_path, split["input_sha256"])
        copy_once(ROOT / split["target_source"], target_path, split["target_sha256"])
        input_signal, input_rate = sf.read(input_path, dtype="float32")
        target_signal, target_rate = sf.read(target_path, dtype="float32")
        if input_signal.ndim != 1 or target_signal.ndim != 1:
            raise RuntimeError(f"non-mono data in {split_name}")
        if input_rate != expected_rate or target_rate != expected_rate:
            raise RuntimeError(f"sample-rate mismatch in {split_name}")
        if input_signal.shape != target_signal.shape:
            raise RuntimeError(f"pair length mismatch in {split_name}")
        if not np.isfinite(input_signal).all() or not np.isfinite(target_signal).all():
            raise RuntimeError(f"non-finite sample in {split_name}")
        files.append(
            {
                "split": split_name,
                "tier": config["tier"],
                "source_group": split["source_group"],
                "sample_rate": expected_rate,
                "input_source": split["input_source"],
                "target_source": split["target_source"],
                "input_path": split["input_path"],
                "target_path": split["target_path"],
                "input_sha256": split["input_sha256"],
                "target_sha256": split["target_sha256"],
                "input_summary": summary(input_signal),
                "target_summary": summary(target_signal),
                "alignment": "official_published_pair_no_compensation",
                "normalization": "none",
            }
        )
        groups[split_name] = [split["source_group"]]

    manifest = {
        "schema_version": 1,
        "campaign_version": config["campaign_version"],
        "name": config["name"],
        "tier": config["tier"],
        "configuration": str(CONFIG_PATH.relative_to(ROOT)),
        "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source_repository": config["source_repository"],
        "source_commit": config["source_commit"],
        "code_license": config["code_license"],
        "dataset_source": config["dataset_source"],
        "dataset_license": config["dataset_license"],
        "archive_path": config["archive_path"],
        "archive_sha256": config["archive_sha256"],
        "reference_model": config["reference_model"],
        "files": files,
        "limitations": [
            "Performer and session identities are not published.",
            "This dataset is used only for the R1 competence gate and INTERNAL_DEV.",
        ],
    }
    splits = {
        "schema_version": 1,
        "campaign_version": config["campaign_version"],
        "dataset": config["name"],
        "rule": "complete official files remain in one split",
        "leakage_check": "passed_at_published_file_level",
        "groups": groups,
        "path_overlap": [],
    }
    write_json_once(MANIFEST_PATH, manifest)
    write_json_once(SPLIT_PATH, splits)
    print(json.dumps({"files": len(files), "manifest": str(MANIFEST_PATH)}, indent=2))


if __name__ == "__main__":
    main()
