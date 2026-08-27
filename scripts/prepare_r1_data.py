#!/usr/bin/env python3
"""Prepare source-disjoint, equal-duration physical pairs for FSSR-R1."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml

from fssr_nam.data.physical import prepare_pair

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/data/r1_physical.yaml"
MANIFEST_PATH = ROOT / "datasets/manifests/r1_physical.json"
SPLIT_PATH = ROOT / "datasets/splits/r1_physical.json"


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def member_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_member(archive: Path, member: str) -> tuple[np.ndarray, int, str]:
    with zipfile.ZipFile(archive) as bundle:
        try:
            payload = bundle.read(member)
        except KeyError as error:
            raise RuntimeError(f"archive member missing: {archive}:{member}") from error
    signal, sample_rate = sf.read(io.BytesIO(payload), dtype="float32")
    if signal.ndim != 1 or not np.all(np.isfinite(signal)):
        raise RuntimeError(f"invalid mono audio member: {archive}:{member}")
    return (
        np.asarray(signal, dtype=np.float32),
        int(sample_rate),
        member_digest(payload),
    )


def verify_archives(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    for name, specification in config["archives"].items():
        path = ROOT / specification["path"]
        if not path.is_file():
            raise RuntimeError(f"required R1 archive is absent: {path}")
        size = path.stat().st_size
        if "size_bytes" in specification and size != int(specification["size_bytes"]):
            raise RuntimeError(f"archive size mismatch: {path}")
        sha256 = file_digest(path)
        if sha256 != specification["sha256"]:
            raise RuntimeError(f"archive SHA-256 mismatch: {path}")
        details: dict[str, Any] = {
            "path": specification["path"],
            "size_bytes": size,
            "sha256": sha256,
        }
        if "published_md5" in specification:
            md5 = file_digest(path, "md5")
            if md5 != specification["published_md5"]:
                raise RuntimeError(f"published archive MD5 mismatch: {path}")
            details["published_md5"] = md5
        with zipfile.ZipFile(path) as bundle:
            bad_member = bundle.testzip()
        if bad_member is not None:
            raise RuntimeError(f"corrupt archive member: {path}:{bad_member}")
        details["zip_integrity"] = "passed"
        verified[name] = details
    return verified


def write_audio_once(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    if path.exists():
        existing, existing_rate = sf.read(path, dtype="float32")
        if existing_rate != sample_rate or not np.array_equal(existing, samples):
            raise RuntimeError(f"refusing to overwrite divergent prepared data: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, sample_rate, subtype="FLOAT")


def write_json_once(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(f"refusing to overwrite divergent manifest: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def _validate_source_ids(devices: dict[str, Any]) -> None:
    for device_name, device in devices.items():
        seen: dict[str, str] = {}
        if set(device["splits"]) != {"train", "validation", "test"}:
            raise ValueError(f"incomplete split declaration for {device_name}")
        for split, sources in device["splits"].items():
            if not sources:
                raise ValueError(f"empty split declaration for {device_name}/{split}")
            for source in sources:
                source_id = str(source["source_id"])
                previous = seen.setdefault(source_id, split)
                if previous != split:
                    raise ValueError(
                        f"source leakage for {device_name}/{source_id}: "
                        f"{previous} and {split}"
                    )


def prepare(config_path: Path = CONFIG_PATH) -> tuple[dict[str, Any], dict[str, Any]]:
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if config["campaign_version"] != "FSSR-R1-v1":
        raise ValueError("unexpected R1 campaign version")
    _validate_source_ids(config["devices"])
    archives = verify_archives(config)
    target_rate = int(config["target_sample_rate"])
    output_root = ROOT / config["output_directory"]
    files: list[dict[str, Any]] = []
    groups: dict[str, dict[str, list[str]]] = {}

    for device_name, device in config["devices"].items():
        groups[device_name] = {}
        for split, sources in device["splits"].items():
            groups[device_name][split] = []
            split_samples = 0
            for source in sources:
                source_id = str(source["source_id"])
                input_archive = (
                    ROOT / config["archives"][source["input_archive"]]["path"]
                )
                target_archive = (
                    ROOT / config["archives"][source["target_archive"]]["path"]
                )
                input_signal, input_rate, input_member_sha256 = read_member(
                    input_archive, source["input_member"]
                )
                target_signal, target_input_rate, target_member_sha256 = read_member(
                    target_archive, source["target_member"]
                )
                expected_rate = int(device["source_rate"])
                if input_rate != expected_rate or target_input_rate != expected_rate:
                    raise RuntimeError(
                        f"source-rate mismatch for {device_name}/{split}/{source_id}"
                    )
                if input_signal.shape != target_signal.shape:
                    raise RuntimeError(
                        f"pair-length mismatch for {device_name}/{split}/{source_id}"
                    )
                prepared = prepare_pair(
                    input_signal,
                    target_signal,
                    source_rate=expected_rate,
                    target_rate=target_rate,
                    trim_start_seconds=float(device["trim_seconds"]),
                    trim_end_seconds=float(device["trim_seconds"]),
                    maximum_seconds=float(source["maximum_seconds"]),
                )
                expected_samples = round(float(source["maximum_seconds"]) * target_rate)
                if prepared.input.size != expected_samples:
                    raise RuntimeError(
                        "insufficient source duration for "
                        f"{device_name}/{split}/{source_id}"
                    )
                output_dir = output_root / device_name / split
                input_path = output_dir / f"{source_id}_input.wav"
                target_path = output_dir / f"{source_id}_target.wav"
                write_audio_once(input_path, prepared.input, target_rate)
                write_audio_once(target_path, prepared.target, target_rate)
                split_samples += expected_samples
                groups[device_name][split].append(source_id)
                files.append(
                    {
                        "device": device_name,
                        "tier": device["r1_tier"],
                        "behavior": device["behavior"],
                        "setting": device["setting"],
                        "split": split,
                        "source_id": source_id,
                        "source_rate": expected_rate,
                        "sample_rate": target_rate,
                        "samples": expected_samples,
                        "input_source": (
                            f"{input_archive.relative_to(ROOT)}:{source['input_member']}"
                        ),
                        "target_source": (
                            f"{target_archive.relative_to(ROOT)}:{source['target_member']}"
                        ),
                        "input_member_sha256": input_member_sha256,
                        "target_member_sha256": target_member_sha256,
                        "input_path": str(input_path.relative_to(ROOT)),
                        "target_path": str(target_path.relative_to(ROOT)),
                        "input_sha256": file_digest(input_path),
                        "target_sha256": file_digest(target_path),
                        "trim_seconds_each_end": float(device["trim_seconds"]),
                        "maximum_seconds": float(source["maximum_seconds"]),
                        "alignment": "published_pair_no_independent_delay_correction",
                        "normalization": "none",
                        "resampling": (
                            "none"
                            if expected_rate == target_rate
                            else "joint_resample_poly_kaiser_beta_8.6"
                        ),
                    }
                )
            expected_total = round(
                float(config["equal_total_seconds"][split]) * target_rate
            )
            if split_samples != expected_total:
                raise RuntimeError(
                    f"unequal audio allocation for {device_name}/{split}: "
                    f"{split_samples} != {expected_total}"
                )

    manifest = {
        "schema_version": 1,
        "campaign_version": config["campaign_version"],
        "name": "r1_physical",
        "configuration": str(config_path.relative_to(ROOT)),
        "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "target_sample_rate": target_rate,
        "equal_total_seconds": config["equal_total_seconds"],
        "archives": archives,
        "files": files,
        "external_report_only_accessed": False,
    }
    split_manifest = {
        "schema_version": 1,
        "campaign_version": config["campaign_version"],
        "dataset": "r1_physical",
        "rule": (
            "each source_id belongs to exactly one split and no window crosses a source"
        ),
        "groups": groups,
        "path_overlap": [],
        "leakage_check": "passed",
        "sealed_test_policy": (
            "test audio unavailable to training and checkpoint selection"
        ),
    }
    write_json_once(MANIFEST_PATH, manifest)
    write_json_once(SPLIT_PATH, split_manifest)
    return manifest, split_manifest


def audit_existing() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file() or not SPLIT_PATH.is_file():
        raise RuntimeError("R1 prepared manifests are absent; run without --audit-only")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    split_manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if split_manifest.get("leakage_check") != "passed":
        raise RuntimeError("R1 split leakage audit is not passed")
    verified = 0
    for entry in manifest["files"]:
        input_path = ROOT / entry["input_path"]
        target_path = ROOT / entry["target_path"]
        if file_digest(input_path) != entry["input_sha256"]:
            raise RuntimeError(f"prepared input checksum mismatch: {input_path}")
        if file_digest(target_path) != entry["target_sha256"]:
            raise RuntimeError(f"prepared target checksum mismatch: {target_path}")
        input_info = sf.info(input_path)
        target_info = sf.info(target_path)
        if (
            input_info.samplerate != manifest["target_sample_rate"]
            or target_info.samplerate != manifest["target_sample_rate"]
            or input_info.frames != entry["samples"]
            or target_info.frames != entry["samples"]
        ):
            raise RuntimeError(f"prepared pair metadata mismatch: {input_path}")
        verified += 1
    return {"files_verified": verified, "leakage_check": "passed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_existing() if args.audit_only else prepare()[1]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
