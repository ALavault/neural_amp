"""Fail-closed audit of the external synchronous R2 capture manifest."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .r2_resampling import (
    DERIVATION_TOLERANCE,
    FILTER_BETA,
    FILTER_ID,
    FILTER_TAPS,
    FrozenDecimator2,
)

CAPTURE_DEVICES = ("fulltone", "bigmuff")
CAPTURE_ROLES = ("dry", "hardware", "loopback")
RATE_KEYS = {"native_192000": 192_000, "derived_96000": 96_000, "derived_48000": 48_000}
SPLIT_REQUIREMENTS = {
    "train": (5, 60),
    "validation": (2, 30),
    "test": (2, 30),
}


class CaptureAuditError(RuntimeError):
    """An instrumentation or manifest failure that makes R2 evidence invalid."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CaptureAuditError(f"{label} must be a mapping")
    return value


def _path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CaptureAuditError(f"{label} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _scan_finite(path: Path) -> None:
    for block in sf.blocks(path, blocksize=1 << 18, dtype="float32", always_2d=False):
        if not np.isfinite(block).all():
            raise CaptureAuditError(f"non-finite capture samples: {path}")


def _audit_triplet(
    files: Mapping[str, Any],
    *,
    root: Path,
    sample_rate: int,
    expected_duration_seconds: int | None,
    require_pcm24: bool,
    scan_samples: bool,
    label: str,
) -> None:
    if set(files) != set(CAPTURE_ROLES):
        raise CaptureAuditError(f"{label} must contain dry, hardware, and loopback")
    frame_counts: list[int] = []
    for role in CAPTURE_ROLES:
        path = _path(root, files[role], f"{label}.{role}")
        if not path.is_file():
            raise CaptureAuditError(f"missing capture file: {path}")
        info = sf.info(path)
        if info.samplerate != sample_rate or info.channels != 1:
            raise CaptureAuditError(
                f"{path} must be mono at {sample_rate} Hz, got "
                f"{info.channels} channels at {info.samplerate} Hz"
            )
        if require_pcm24 and info.subtype != "PCM_24":
            raise CaptureAuditError(f"{path} must be PCM_24")
        if expected_duration_seconds is not None:
            expected_frames = sample_rate * expected_duration_seconds
            if info.frames != expected_frames:
                raise CaptureAuditError(
                    f"{path} must contain exactly {expected_frames} frames"
                )
        frame_counts.append(info.frames)
        if scan_samples:
            _scan_finite(path)
    if len(set(frame_counts)) != 1:
        raise CaptureAuditError(f"{label} synchronous channel lengths differ")


def _validate_file_declarations(
    files: object,
    *,
    root: Path,
    duration_seconds: int | None,
    scan_samples: bool,
    label: str,
    check_files: bool,
) -> None:
    declarations = _mapping(files, f"{label}.files")
    if set(declarations) != set(RATE_KEYS):
        raise CaptureAuditError(f"{label}.files must contain frozen 192/96/48 rates")
    for rate_key, sample_rate in RATE_KEYS.items():
        triplet = _mapping(declarations[rate_key], f"{label}.{rate_key}")
        if check_files:
            _audit_triplet(
                triplet,
                root=root,
                sample_rate=sample_rate,
                expected_duration_seconds=duration_seconds,
                require_pcm24=rate_key == "native_192000",
                scan_samples=scan_samples,
                label=f"{label}.{rate_key}",
            )
        elif set(triplet) != set(CAPTURE_ROLES):
            raise CaptureAuditError(
                f"{label}.{rate_key} must declare dry, hardware, and loopback"
            )


def _verify_derived_triplets(
    declarations: Mapping[str, Any], *, root: Path, label: str
) -> None:
    native = _mapping(declarations["native_192000"], f"{label}.native")
    rate_96 = _mapping(declarations["derived_96000"], f"{label}.rate96")
    rate_48 = _mapping(declarations["derived_48000"], f"{label}.rate48")
    for role in CAPTURE_ROLES:
        native_path = _path(root, native[role], f"{label}.{role}.native")
        rate_96_path = _path(root, rate_96[role], f"{label}.{role}.rate96")
        rate_48_path = _path(root, rate_48[role], f"{label}.{role}.rate48")
        first = FrozenDecimator2()
        second = FrozenDecimator2()
        maximum = 0.0
        with (
            sf.SoundFile(native_path) as source,
            sf.SoundFile(rate_96_path) as observed_96,
            sf.SoundFile(rate_48_path) as observed_48,
        ):
            while True:
                block = source.read(1 << 18, dtype="float64", always_2d=False)
                if not len(block):
                    break
                expected_96 = first.process(block)
                actual_96 = observed_96.read(
                    len(expected_96), dtype="float64", always_2d=False
                )
                if len(actual_96) != len(expected_96):
                    raise CaptureAuditError(f"{label}.{role} 96 kHz length drift")
                expected_48 = second.process(expected_96)
                actual_48 = observed_48.read(
                    len(expected_48), dtype="float64", always_2d=False
                )
                if len(actual_48) != len(expected_48):
                    raise CaptureAuditError(f"{label}.{role} 48 kHz length drift")
                maximum = max(
                    maximum,
                    float(np.max(np.abs(actual_96 - expected_96), initial=0.0)),
                    float(np.max(np.abs(actual_48 - expected_48), initial=0.0)),
                )
            if len(observed_96.read(1, dtype="float64")) or len(
                observed_48.read(1, dtype="float64")
            ):
                raise CaptureAuditError(
                    f"{label}.{role} derived audio has extra frames"
                )
        if maximum > DERIVATION_TOLERANCE:
            raise CaptureAuditError(
                f"{label}.{role} derived FIR mismatch: {maximum:.9g}"
            )


def audit_capture_manifest(
    manifest: Mapping[str, Any], *, root: Path, check_files: bool = True
) -> dict[str, Any]:
    """Validate all capture counts while never reading sealed test samples."""
    if manifest.get("campaign_version") != "FSSR-R2-v1":
        raise CaptureAuditError("capture manifest campaign_version must be FSSR-R2-v1")
    if manifest.get("status") != "complete":
        raise CaptureAuditError("capture manifest status must be complete")
    capture = _mapping(manifest.get("capture"), "capture")
    expected_capture = {
        "sample_rate_hz": 192_000,
        "bit_depth": 24,
        "synchronous": True,
        "independent_normalization": False,
    }
    for name, expected in expected_capture.items():
        if capture.get(name) != expected:
            raise CaptureAuditError(f"capture.{name} must equal {expected!r}")
    derivation = _mapping(manifest.get("derivation"), "derivation")
    if derivation.get("rates_hz") != [96_000, 48_000]:
        raise CaptureAuditError("derived capture rates must be exactly 96/48 kHz")
    if derivation.get("chain") != "frozen_causal_fir":
        raise CaptureAuditError("capture derivation must use the frozen causal FIR")
    if derivation.get("independent_normalization") is not False:
        raise CaptureAuditError("independent derived normalization is forbidden")
    expected_derivation = {
        "filter_id": FILTER_ID,
        "filter_taps": FILTER_TAPS,
        "kaiser_beta": FILTER_BETA,
        "decimation_phase": 0,
        "stages": {"derived_96000": [2], "derived_48000": [2, 2]},
        "verification_max_abs_error": DERIVATION_TOLERANCE,
    }
    for name, expected in expected_derivation.items():
        if derivation.get(name) != expected:
            raise CaptureAuditError(f"derivation.{name} must equal {expected!r}")
    devices = _mapping(manifest.get("devices"), "devices")
    if set(devices) != set(CAPTURE_DEVICES):
        raise CaptureAuditError("capture manifest must contain Fulltone and Big Muff")
    total_recordings = 0
    sealed_metadata_only = 0
    for device in CAPTURE_DEVICES:
        device_data = _mapping(devices[device], device)
        recordings = device_data.get("recordings")
        if not isinstance(recordings, list):
            raise CaptureAuditError(f"{device}.recordings must be a list")
        split_counts: Counter[str] = Counter()
        performance_ids: list[str] = []
        recording_ids: list[str] = []
        for index, raw_recording in enumerate(recordings):
            recording = _mapping(raw_recording, f"{device}.recordings[{index}]")
            split = recording.get("split")
            if split not in SPLIT_REQUIREMENTS:
                raise CaptureAuditError(f"invalid {device} capture split: {split!r}")
            _, duration = SPLIT_REQUIREMENTS[split]
            if recording.get("duration_seconds") != duration:
                raise CaptureAuditError(
                    f"{device} {split} recording duration must be {duration} seconds"
                )
            if recording.get("sealed") is not (split == "test"):
                raise CaptureAuditError(
                    f"{device} {split} sealed declaration is invalid"
                )
            performance_id = recording.get("performance_id")
            recording_id = recording.get("recording_id")
            if not isinstance(performance_id, str) or not performance_id:
                raise CaptureAuditError("performance_id must be a non-empty string")
            if not isinstance(recording_id, str) or not recording_id:
                raise CaptureAuditError("recording_id must be a non-empty string")
            performance_ids.append(performance_id)
            recording_ids.append(recording_id)
            split_counts[split] += 1
            _validate_file_declarations(
                recording.get("files"),
                root=root,
                duration_seconds=duration,
                scan_samples=split != "test",
                label=f"{device}.{recording_id}",
                check_files=check_files,
            )
            if check_files and split != "test":
                _verify_derived_triplets(
                    _mapping(recording.get("files"), "recording files"),
                    root=root,
                    label=f"{device}.{recording_id}",
                )
            if split == "test":
                sealed_metadata_only += 1
        for split, (count, _) in SPLIT_REQUIREMENTS.items():
            if split_counts[split] != count:
                raise CaptureAuditError(
                    f"{device} {split} requires {count} recordings, "
                    f"got {split_counts[split]}"
                )
        if len(set(performance_ids)) != len(performance_ids):
            raise CaptureAuditError(f"{device} repeats a performance across splits")
        if len(set(recording_ids)) != len(recording_ids):
            raise CaptureAuditError(f"{device} repeats a recording_id")
        probes = device_data.get("probes")
        if not isinstance(probes, list) or len(probes) != 6:
            raise CaptureAuditError(f"{device} requires exactly six probe repetitions")
        probe_counts: Counter[str] = Counter()
        for index, raw_probe in enumerate(probes):
            probe = _mapping(raw_probe, f"{device}.probes[{index}]")
            kind = probe.get("kind")
            if kind not in {"coherent_sine_bank", "two_tone"}:
                raise CaptureAuditError(f"invalid {device} probe kind: {kind!r}")
            if probe.get("repetition") not in {1, 2, 3}:
                raise CaptureAuditError("probe repetition must be one, two, or three")
            probe_counts[kind] += 1
            _validate_file_declarations(
                probe.get("files"),
                root=root,
                duration_seconds=None,
                scan_samples=True,
                label=f"{device}.probe{index}",
                check_files=check_files,
            )
            if check_files:
                _verify_derived_triplets(
                    _mapping(probe.get("files"), "probe files"),
                    root=root,
                    label=f"{device}.probe{index}",
                )
        if probe_counts != Counter({"coherent_sine_bank": 3, "two_tone": 3}):
            raise CaptureAuditError(f"{device} probe repetition matrix is incomplete")
        total_recordings += len(recordings)
    if manifest.get("external_report_only_locked") is not True:
        raise CaptureAuditError("EXTERNAL_REPORT_ONLY must remain locked")
    return {
        "campaign_version": "FSSR-R2-v1",
        "valid": True,
        "status": "passed",
        "recordings": total_recordings,
        "probe_repetitions": 12,
        "sealed_test_files_sample_data_read": False,
        "sealed_test_recordings_metadata_checked": sealed_metadata_only,
    }


def load_and_audit_capture_manifest(
    path: Path, *, root: Path, check_files: bool = True
) -> dict[str, Any]:
    if not path.is_file():
        raise CaptureAuditError(f"R2 capture manifest is missing: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CaptureAuditError(f"invalid R2 capture manifest: {error}") from error
    return audit_capture_manifest(manifest, root=root, check_files=check_files)
