from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from fssr_nam.data.r2_capture import CaptureAuditError, audit_capture_manifest


def _files(prefix: str) -> dict[str, dict[str, str]]:
    return {
        rate: {
            role: f"datasets/raw/r2/{prefix}_{rate}_{role}.wav"
            for role in ("dry", "hardware", "loopback")
        }
        for rate in ("native_192000", "derived_96000", "derived_48000")
    }


def _manifest() -> dict:
    devices = {}
    for device in ("fulltone", "bigmuff"):
        recordings = []
        for split, count, duration in (
            ("train", 5, 60),
            ("validation", 2, 30),
            ("test", 2, 30),
        ):
            for index in range(count):
                recording_id = f"{device}_{split}_{index}"
                recordings.append(
                    {
                        "recording_id": recording_id,
                        "performance_id": f"performance_{recording_id}",
                        "split": split,
                        "duration_seconds": duration,
                        "sealed": split == "test",
                        "files": _files(recording_id),
                    }
                )
        probes = [
            {
                "kind": kind,
                "repetition": repetition,
                "files": _files(f"{device}_{kind}_{repetition}"),
            }
            for kind in ("coherent_sine_bank", "two_tone")
            for repetition in (1, 2, 3)
        ]
        devices[device] = {"recordings": recordings, "probes": probes}
    return {
        "campaign_version": "FSSR-R2-v1",
        "status": "complete",
        "capture": {
            "sample_rate_hz": 192000,
            "bit_depth": 24,
            "synchronous": True,
            "independent_normalization": False,
        },
        "derivation": {
            "rates_hz": [96000, 48000],
            "chain": "frozen_causal_fir",
            "filter_id": "r2-causal-kaiser65-beta8p6-v1",
            "filter_taps": 65,
            "kaiser_beta": 8.6,
            "decimation_phase": 0,
            "stages": {"derived_96000": [2], "derived_48000": [2, 2]},
            "verification_max_abs_error": 0.000002,
            "independent_normalization": False,
        },
        "devices": devices,
        "external_report_only_locked": True,
    }


def test_r2_capture_contract_has_exact_counts_and_keeps_tests_sealed() -> None:
    report = audit_capture_manifest(_manifest(), root=Path("/"), check_files=False)
    assert report["recordings"] == 18
    assert report["probe_repetitions"] == 12
    assert report["sealed_test_recordings_metadata_checked"] == 4
    assert report["sealed_test_files_sample_data_read"] is False


def test_r2_capture_rejects_performance_overlap_across_splits() -> None:
    manifest = _manifest()
    recordings = manifest["devices"]["fulltone"]["recordings"]
    recordings[-1]["performance_id"] = recordings[0]["performance_id"]
    with pytest.raises(CaptureAuditError, match="repeats a performance"):
        audit_capture_manifest(manifest, root=Path("/"), check_files=False)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("status", "pending"), "status must be complete"),
        (("external_report_only_locked", False), "EXTERNAL_REPORT_ONLY"),
    ],
)
def test_r2_capture_incomplete_or_unlocked_evidence_is_invalid(
    mutation: tuple[str, object], message: str
) -> None:
    manifest = deepcopy(_manifest())
    manifest[mutation[0]] = mutation[1]
    with pytest.raises(CaptureAuditError, match=message):
        audit_capture_manifest(manifest, root=Path("/"), check_files=False)


def test_r2_capture_rejects_missing_probe_repetition() -> None:
    manifest = _manifest()
    manifest["devices"]["bigmuff"]["probes"].pop()
    with pytest.raises(CaptureAuditError, match="six probe"):
        audit_capture_manifest(manifest, root=Path("/"), check_files=False)
