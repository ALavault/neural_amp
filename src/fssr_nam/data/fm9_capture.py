"""Metadata-only validation for the prospective FM9 capture protocol."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

PROTOCOL_NAME = "FM9-ORACLE-v1"
PROTOCOL_STATUS = "protocol_only_no_capture_authorized"
PROTOCOL_PATH = Path("configs/data/fm9_capture_protocol.yaml")
SAMPLE_RATE_HZ = 48_000
CHANNELS = 1
PRECISION = "float32"
SPLIT_MINIMUM_SECONDS = {
    "train": 120.0,
    "development": 30.0,
    "test": 30.0,
}
REQUIRED_MANIFEST_FIELDS = (
    "fm9_firmware",
    "preset_identifier",
    "scene_identifier",
    "input_level_dbfs",
    "output_level_dbfs",
    "bypassed_cabinet",
    "dry_file",
    "wet_file",
    "source_group",
    "split",
)
REQUIRED_RECORDING_METADATA = (
    "sample_rate_hz",
    "channels",
    "precision",
    "duration_seconds",
    "finite",
    "clipped_sample_fraction",
    "dry_peak_dbfs",
    "wet_peak_dbfs",
    "dry_rms_dbfs",
    "wet_rms_dbfs",
    "alignment_marker_present",
    "integer_delay_samples",
    "fractional_delay_samples",
)
AUTHORIZATION_FIELDS = (
    "capture_authorized",
    "render_import_authorized",
    "selection_use_authorized",
    "human_feedback_use_authorized",
)


class FM9ProtocolError(ValueError):
    """The frozen prospective FM9 protocol is missing or has drifted."""


class FM9ManifestError(ValueError):
    """FM9 manifest metadata cannot support valid evidence."""


class FM9AuthorizationError(PermissionError):
    """The prospective protocol does not authorize capture or evidence use."""


def _mapping(value: object, label: str, error: type[ValueError]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error(f"{label} must be a mapping")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise FM9ManifestError(f"{label} must be a sequence")
    return value


def _require_equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise FM9ProtocolError(f"{label} must be {expected!r}, got {value!r}")


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FM9ManifestError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise FM9ManifestError(f"{label} must be a finite number")
    return number


def _non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FM9ManifestError(f"{label} must be a non-empty string")
    return value


def validate_fm9_protocol(protocol: Mapping[str, Any]) -> None:
    """Reject any drift in the decision-bearing FM9 protocol fields."""
    _require_equal(protocol.get("schema_version"), 1, "schema_version")
    _require_equal(protocol.get("protocol"), PROTOCOL_NAME, "protocol")
    _require_equal(protocol.get("status"), PROTOCOL_STATUS, "status")
    _require_equal(protocol.get("sample_rate_hz"), SAMPLE_RATE_HZ, "sample_rate_hz")
    _require_equal(protocol.get("channels"), CHANNELS, "channels")
    _require_equal(protocol.get("precision"), PRECISION, "precision")
    _require_equal(protocol.get("capture_owner"), "user", "capture_owner")
    _require_equal(
        protocol.get("required_manifest_fields"),
        list(REQUIRED_MANIFEST_FIELDS),
        "required_manifest_fields",
    )

    splits = _mapping(protocol.get("splits"), "splits", FM9ProtocolError)
    _require_equal(set(splits), set(SPLIT_MINIMUM_SECONDS), "splits")
    for split, minimum in SPLIT_MINIMUM_SECONDS.items():
        declaration = _mapping(splits[split], f"splits.{split}", FM9ProtocolError)
        _require_equal(
            declaration.get("minimum_seconds"),
            int(minimum),
            f"splits.{split}.minimum_seconds",
        )

    _require_equal(protocol.get("source_groups_disjoint"), True, "source groups")
    _require_equal(
        protocol.get("fixed_preset_and_scene_per_model"),
        True,
        "fixed preset and scene",
    )

    alignment = _mapping(protocol.get("alignment"), "alignment", FM9ProtocolError)
    _require_equal(alignment.get("marker_required"), True, "alignment.marker")
    _require_equal(
        alignment.get("integer_and_fractional_delay_report_required"),
        True,
        "alignment.delay report",
    )
    _require_equal(
        alignment.get("automatic_delay_correction_allowed"),
        False,
        "alignment.automatic correction",
    )

    guards = _mapping(protocol.get("signal_guards"), "signal_guards", FM9ProtocolError)
    expected_guards = {
        "finite_required": True,
        "clipped_sample_fraction_maximum": 0.0,
        "dry_peak_dbfs_maximum": -1.0,
        "wet_peak_dbfs_maximum": -1.0,
        "dry_rms_dbfs_minimum": -40.0,
        "wet_rms_dbfs_minimum": -40.0,
    }
    for name, expected in expected_guards.items():
        _require_equal(guards.get(name), expected, f"signal_guards.{name}")

    boundaries = _mapping(protocol.get("boundaries"), "boundaries", FM9ProtocolError)
    expected_boundaries = {
        "render_import_authorized": False,
        "selection_use_authorized": False,
        "human_feedback_use_authorized": False,
        "user_authorization_required_to_change": True,
    }
    for name, expected in expected_boundaries.items():
        _require_equal(boundaries.get(name), expected, f"boundaries.{name}")


def load_fm9_protocol(root: Path) -> dict[str, Any]:
    """Load and validate protocol YAML without accessing capture audio."""
    path = root / PROTOCOL_PATH
    if not path.is_file():
        raise FM9ProtocolError(f"missing FM9 protocol: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    protocol = _mapping(loaded, "FM9 protocol", FM9ProtocolError)
    validate_fm9_protocol(protocol)
    return dict(protocol)


def assert_fm9_operation_authorized(
    protocol: Mapping[str, Any], operation: str
) -> None:
    """Fail closed while the repository contains a protocol-only declaration."""
    validate_fm9_protocol(protocol)
    if operation not in {"capture", "render_import"}:
        raise FM9AuthorizationError(f"unknown FM9 operation: {operation}")
    if operation == "capture":
        raise FM9AuthorizationError(
            "FM9 capture is not authorized by protocol_only_no_capture_authorized"
        )
    raise FM9AuthorizationError("FM9 render import is not authorized")


def _validate_recording(
    recording: Mapping[str, Any],
    *,
    index: int,
    guards: Mapping[str, Any],
) -> tuple[str, str, float, tuple[str, str]]:
    label = f"recordings[{index}]"
    required = (*REQUIRED_MANIFEST_FIELDS, *REQUIRED_RECORDING_METADATA)
    missing = [name for name in required if name not in recording]
    if missing:
        raise FM9ManifestError(f"{label} missing fields: {', '.join(missing)}")

    for name in (
        "fm9_firmware",
        "preset_identifier",
        "scene_identifier",
        "dry_file",
        "wet_file",
        "source_group",
    ):
        _non_empty_string(recording[name], f"{label}.{name}")
    if recording["dry_file"] == recording["wet_file"]:
        raise FM9ManifestError(f"{label} dry_file and wet_file must differ")
    if not isinstance(recording["bypassed_cabinet"], bool):
        raise FM9ManifestError(f"{label}.bypassed_cabinet must be boolean")

    if recording["sample_rate_hz"] != SAMPLE_RATE_HZ:
        raise FM9ManifestError(f"{label} must declare 48000 Hz")
    if recording["channels"] != CHANNELS:
        raise FM9ManifestError(f"{label} must declare mono audio")
    if recording["precision"] != PRECISION:
        raise FM9ManifestError(f"{label} must declare float32 precision")

    split = recording["split"]
    if split not in SPLIT_MINIMUM_SECONDS:
        raise FM9ManifestError(f"{label}.split must be train, development, or test")
    duration = _finite_number(recording["duration_seconds"], f"{label}.duration")
    if duration <= 0.0:
        raise FM9ManifestError(f"{label}.duration_seconds must be positive")

    for name in ("input_level_dbfs", "output_level_dbfs"):
        _finite_number(recording[name], f"{label}.{name}")
    if recording["finite"] is not True:
        raise FM9ManifestError(f"{label}.finite must be true")

    clipped = _finite_number(
        recording["clipped_sample_fraction"], f"{label}.clipped_sample_fraction"
    )
    if clipped < 0.0 or clipped > guards["clipped_sample_fraction_maximum"]:
        raise FM9ManifestError(f"{label}.clipped_sample_fraction exceeds guard")

    levels = {
        name: _finite_number(recording[name], f"{label}.{name}")
        for name in (
            "dry_peak_dbfs",
            "wet_peak_dbfs",
            "dry_rms_dbfs",
            "wet_rms_dbfs",
        )
    }
    for role in ("dry", "wet"):
        peak = levels[f"{role}_peak_dbfs"]
        rms = levels[f"{role}_rms_dbfs"]
        if peak > guards[f"{role}_peak_dbfs_maximum"]:
            raise FM9ManifestError(f"{label}.{role}_peak_dbfs exceeds guard")
        if rms < guards[f"{role}_rms_dbfs_minimum"]:
            raise FM9ManifestError(f"{label}.{role}_rms_dbfs is below guard")
        if rms > peak:
            raise FM9ManifestError(f"{label}.{role}_rms_dbfs exceeds peak")

    if recording["alignment_marker_present"] is not True:
        raise FM9ManifestError(f"{label}.alignment_marker_present must be true")
    delay = recording["integer_delay_samples"]
    if isinstance(delay, bool) or not isinstance(delay, int):
        raise FM9ManifestError(f"{label}.integer_delay_samples must be an integer")
    _finite_number(
        recording["fractional_delay_samples"],
        f"{label}.fractional_delay_samples",
    )

    source_group = str(recording["source_group"])
    preset_scene = (
        str(recording["preset_identifier"]),
        str(recording["scene_identifier"]),
    )
    return str(split), source_group, duration, preset_scene


def validate_fm9_manifest(
    manifest: Mapping[str, Any], *, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate declared metadata only; never open or inspect dry/wet files."""
    validate_fm9_protocol(protocol)
    if manifest.get("protocol") != PROTOCOL_NAME:
        raise FM9ManifestError(f"manifest.protocol must be {PROTOCOL_NAME}")
    for name in AUTHORIZATION_FIELDS:
        if manifest.get(name, False) is not False:
            raise FM9ManifestError(f"manifest cannot claim {name}")

    raw_recordings = _sequence(manifest.get("recordings"), "recordings")
    if not raw_recordings:
        raise FM9ManifestError("recordings must not be empty")

    guards = _mapping(protocol["signal_guards"], "signal_guards", FM9ProtocolError)
    durations = defaultdict(float)
    groups_by_split: dict[str, set[str]] = {
        split: set() for split in SPLIT_MINIMUM_SECONDS
    }
    preset_scene: tuple[str, str] | None = None
    file_paths: set[str] = set()
    for index, value in enumerate(raw_recordings):
        recording = _mapping(value, f"recordings[{index}]", FM9ManifestError)
        split, source_group, duration, current_preset_scene = _validate_recording(
            recording, index=index, guards=guards
        )
        durations[split] += duration
        groups_by_split[split].add(source_group)
        if preset_scene is None:
            preset_scene = current_preset_scene
        elif current_preset_scene != preset_scene:
            raise FM9ManifestError("preset and scene must stay fixed within a manifest")
        for role in ("dry_file", "wet_file"):
            path = str(recording[role])
            if path in file_paths:
                raise FM9ManifestError(f"audio path declared more than once: {path}")
            file_paths.add(path)

    for split, minimum in SPLIT_MINIMUM_SECONDS.items():
        if durations[split] < minimum:
            raise FM9ManifestError(
                f"{split} duration must be at least {minimum:g} seconds"
            )
    split_names = tuple(SPLIT_MINIMUM_SECONDS)
    for first_index, first in enumerate(split_names):
        for second in split_names[first_index + 1 :]:
            overlap = groups_by_split[first] & groups_by_split[second]
            if overlap:
                raise FM9ManifestError(
                    f"source groups overlap between {first} and {second}: "
                    f"{', '.join(sorted(overlap))}"
                )

    return {
        "protocol": PROTOCOL_NAME,
        "recording_count": len(raw_recordings),
        "duration_seconds_by_split": dict(durations),
        "source_groups_by_split": {
            split: sorted(groups) for split, groups in groups_by_split.items()
        },
        "audio_files_read": False,
        "capture_authorized": False,
        "render_import_authorized": False,
    }
