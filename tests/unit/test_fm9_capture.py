from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from fssr_nam.data.fm9_capture import (
    FM9AuthorizationError,
    FM9ManifestError,
    FM9ProtocolError,
    assert_fm9_operation_authorized,
    load_fm9_protocol,
    validate_fm9_manifest,
    validate_fm9_protocol,
)

ROOT = Path(__file__).resolve().parents[2]


def _protocol() -> dict[str, object]:
    return yaml.safe_load(
        (ROOT / "configs/data/fm9_capture_protocol.yaml").read_text(encoding="utf-8")
    )


def _recording(split: str, duration_seconds: float) -> dict[str, object]:
    return {
        "fm9_firmware": "prospective-firmware",
        "preset_identifier": "prospective-preset",
        "scene_identifier": "prospective-scene",
        "input_level_dbfs": -12.0,
        "output_level_dbfs": -12.0,
        "bypassed_cabinet": True,
        "dry_file": f"unread/{split}-dry.wav",
        "wet_file": f"unread/{split}-wet.wav",
        "source_group": f"source-{split}",
        "split": split,
        "sample_rate_hz": 48_000,
        "channels": 1,
        "precision": "float32",
        "duration_seconds": duration_seconds,
        "finite": True,
        "clipped_sample_fraction": 0.0,
        "dry_peak_dbfs": -3.0,
        "wet_peak_dbfs": -2.0,
        "dry_rms_dbfs": -20.0,
        "wet_rms_dbfs": -18.0,
        "alignment_marker_present": True,
        "integer_delay_samples": 37,
        "fractional_delay_samples": 0.25,
    }


def _manifest() -> dict[str, object]:
    return {
        "protocol": "FM9-ORACLE-v1",
        "recordings": [
            _recording("train", 120.0),
            _recording("development", 30.0),
            _recording("test", 30.0),
        ],
    }


def test_frozen_protocol_validates_but_authorizes_no_operation() -> None:
    protocol = load_fm9_protocol(ROOT)
    for operation in ("capture", "render_import"):
        with pytest.raises(FM9AuthorizationError, match="not authorized"):
            assert_fm9_operation_authorized(protocol, operation)


def test_protocol_rejects_capture_or_render_authorization_drift() -> None:
    protocol = _protocol()
    protocol["status"] = "capture_authorized"
    with pytest.raises(FM9ProtocolError, match="status"):
        validate_fm9_protocol(protocol)

    protocol = _protocol()
    protocol["boundaries"]["render_import_authorized"] = True
    with pytest.raises(FM9ProtocolError, match="render_import_authorized"):
        validate_fm9_protocol(protocol)


def test_manifest_metadata_passes_without_opening_declared_audio() -> None:
    report = validate_fm9_manifest(_manifest(), protocol=_protocol())
    assert report == {
        "protocol": "FM9-ORACLE-v1",
        "recording_count": 3,
        "duration_seconds_by_split": {
            "train": 120.0,
            "development": 30.0,
            "test": 30.0,
        },
        "source_groups_by_split": {
            "train": ["source-train"],
            "development": ["source-development"],
            "test": ["source-test"],
        },
        "audio_files_read": False,
        "capture_authorized": False,
        "render_import_authorized": False,
    }


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("sample_rate_hz", 44_100, "48000 Hz"),
        ("channels", 2, "mono"),
        ("precision", "pcm24", "float32"),
        ("finite", False, "finite must be true"),
        ("clipped_sample_fraction", 0.001, "exceeds guard"),
        ("dry_peak_dbfs", -0.9, "peak_dbfs exceeds guard"),
        ("wet_rms_dbfs", -41.0, "rms_dbfs is below guard"),
        ("alignment_marker_present", False, "marker_present must be true"),
        ("integer_delay_samples", 3.5, "must be an integer"),
        ("fractional_delay_samples", float("nan"), "finite number"),
    ],
)
def test_manifest_rejects_invalid_signal_and_alignment_metadata(
    field: str, replacement: object, message: str
) -> None:
    manifest = _manifest()
    manifest["recordings"][0][field] = replacement
    with pytest.raises(FM9ManifestError, match=message):
        validate_fm9_manifest(manifest, protocol=_protocol())


def test_manifest_requires_every_frozen_field() -> None:
    manifest = _manifest()
    manifest["recordings"][0].pop("fm9_firmware")
    with pytest.raises(FM9ManifestError, match="missing fields: fm9_firmware"):
        validate_fm9_manifest(manifest, protocol=_protocol())


def test_manifest_enforces_minimum_split_duration() -> None:
    manifest = _manifest()
    manifest["recordings"][1]["duration_seconds"] = 29.99
    with pytest.raises(FM9ManifestError, match="development duration"):
        validate_fm9_manifest(manifest, protocol=_protocol())


def test_manifest_rejects_source_group_overlap_between_splits() -> None:
    manifest = _manifest()
    manifest["recordings"][2]["source_group"] = "source-train"
    with pytest.raises(FM9ManifestError, match="source groups overlap"):
        validate_fm9_manifest(manifest, protocol=_protocol())


def test_manifest_keeps_preset_and_scene_fixed() -> None:
    manifest = _manifest()
    manifest["recordings"][2]["scene_identifier"] = "other-scene"
    with pytest.raises(FM9ManifestError, match="preset and scene"):
        validate_fm9_manifest(manifest, protocol=_protocol())


@pytest.mark.parametrize("field", ["capture_authorized", "render_import_authorized"])
def test_manifest_cannot_claim_current_authorization(field: str) -> None:
    manifest = copy.deepcopy(_manifest())
    manifest[field] = True
    with pytest.raises(FM9ManifestError, match=field):
        validate_fm9_manifest(manifest, protocol=_protocol())
