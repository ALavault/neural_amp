#!/usr/bin/env python3
"""Prepare bounded, source-file-disjoint physical data for the M4 smoke test."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from fssr_nam.data.physical import prepare_pair

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/data/m4_internal.yaml"
MANIFEST_PATH = ROOT / "datasets/manifests/m4_internal.json"
SPLIT_PATH = ROOT / "datasets/splits/m4_internal.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_member(archive: Path, member: str) -> tuple[np.ndarray, int, str]:
    with zipfile.ZipFile(archive) as bundle:
        payload = bundle.read(member)
    samples, sample_rate = sf.read(io.BytesIO(payload), dtype="float32")
    if samples.ndim != 1:
        raise ValueError(f"{archive}:{member} is not mono")
    return samples, int(sample_rate), hashlib.sha256(payload).hexdigest()


def sample_summary(samples: np.ndarray) -> dict[str, object]:
    return {
        "samples": int(samples.size),
        "peak": float(np.max(np.abs(samples), initial=0.0)),
        "rms": float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))),
        "nonfinite": int(np.count_nonzero(~np.isfinite(samples))),
        "clipped_samples": int(np.count_nonzero(np.abs(samples) >= 0.999)),
    }


def verify_archive(path: Path, expected: str, cache: dict[Path, str]) -> str:
    actual = cache.setdefault(path, sha256(path))
    if actual != expected:
        raise RuntimeError(f"archive checksum mismatch for {path}")
    return actual


def main() -> None:
    config_bytes = CONFIG_PATH.read_bytes()
    config = yaml.safe_load(config_bytes)
    target_rate = int(config["target_sample_rate"])
    output_root = ROOT / config["output_directory"]
    archive_hashes: dict[Path, str] = {}
    files: list[dict[str, object]] = []
    groups: dict[str, dict[str, list[str]]] = {}

    for device_name, device in config["devices"].items():
        input_archive = ROOT / device["input_archive"]
        target_archive = ROOT / device["target_archive"]
        verify_archive(input_archive, device["input_archive_sha256"], archive_hashes)
        verify_archive(target_archive, device["target_archive_sha256"], archive_hashes)
        groups[device_name] = {}
        for split_name, split in device["splits"].items():
            input_signal, input_rate, input_member_sha = read_member(
                input_archive, split["input_member"]
            )
            target_signal, target_input_rate, target_member_sha = read_member(
                target_archive, split["target_member"]
            )
            expected_rate = int(device["source_rate"])
            if input_rate != expected_rate or target_input_rate != expected_rate:
                raise RuntimeError(
                    f"unexpected source rate for {device_name}/{split_name}"
                )
            prepared = prepare_pair(
                input_signal,
                target_signal,
                source_rate=expected_rate,
                target_rate=target_rate,
                trim_start_seconds=float(device["trim_start_seconds"]),
                trim_end_seconds=float(device["trim_end_seconds"]),
                maximum_seconds=float(config["maximum_seconds"][split_name]),
            )
            output_dir = output_root / device_name
            output_dir.mkdir(parents=True, exist_ok=True)
            input_path = output_dir / f"{split_name}_input.wav"
            target_path = output_dir / f"{split_name}_target.wav"
            sf.write(input_path, prepared.input, target_rate, subtype="FLOAT")
            sf.write(target_path, prepared.target, target_rate, subtype="FLOAT")
            files.append(
                {
                    "device": device_name,
                    "behavior": device["behavior"],
                    "setting": device["setting"],
                    "split": split_name,
                    "tier": "INTERNAL_DEV",
                    "source_group": split["source_group"],
                    "source_rate": expected_rate,
                    "sample_rate": target_rate,
                    "input_source": (
                        f"{device['input_archive']}:{split['input_member']}"
                    ),
                    "target_source": (
                        f"{device['target_archive']}:{split['target_member']}"
                    ),
                    "input_member_sha256": input_member_sha,
                    "target_member_sha256": target_member_sha,
                    "input_path": str(input_path.relative_to(ROOT)),
                    "target_path": str(target_path.relative_to(ROOT)),
                    "input_sha256": sha256(input_path),
                    "target_sha256": sha256(target_path),
                    "input_summary": sample_summary(prepared.input),
                    "target_summary": sample_summary(prepared.target),
                    "trim_start_seconds": float(device["trim_start_seconds"]),
                    "trim_end_seconds": float(device["trim_end_seconds"]),
                    "maximum_seconds": float(config["maximum_seconds"][split_name]),
                    "resampling": (
                        "none"
                        if expected_rate == target_rate
                        else (
                            "scipy.signal.resample_poly Kaiser beta=8.6, "
                            "jointly applied"
                        )
                    ),
                    "alignment": device["alignment"],
                    "normalization": "none",
                }
            )
            groups[device_name][split_name] = [split["source_group"]]

    manifest = {
        "schema_version": 1,
        "name": "m4_internal",
        "tier": "INTERNAL_DEV",
        "configuration": str(CONFIG_PATH.relative_to(ROOT)),
        "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "files": files,
        "limitations": [
            (
                "Big Muff upstream split files do not identify performer/session "
                "provenance."
            ),
            (
                "Big Muff nonlinear correlation did not support a robust extra "
                "delay correction."
            ),
            (
                "The bounded M4 excerpts are smoke-test evidence, not "
                "final-campaign evidence."
            ),
        ],
    }
    split_manifest = {
        "schema_version": 1,
        "dataset": "m4_internal",
        "rule": "complete upstream files remain in one split; no window crosses a file",
        "leakage_check": "passed_at_published_file_level",
        "groups": groups,
        "path_overlap": [],
        "known_limitation": "Big Muff performer/session identity is not published",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    SPLIT_PATH.write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(files), "manifest": str(MANIFEST_PATH)}, indent=2))


if __name__ == "__main__":
    main()
