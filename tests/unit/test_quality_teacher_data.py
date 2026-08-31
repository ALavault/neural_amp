from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import fssr_nam.data.quality_teacher as quality_teacher_data
from fssr_nam.data.quality_teacher import (
    CAMPAIGN_DEVICES,
    REQUIRED_TEST_GATES,
    PairDescriptor,
    QualityTeacherDataError,
    SealedTestAccessError,
    assigned_split,
    audit_pair_arrays,
    authorize_test_access,
    load_data_contract,
    materialize_pair_paths,
    prepare_authorized_pair,
    reject_post_training_corrections,
    validate_data_contract,
    validate_source_isolation,
)

ROOT = Path(__file__).resolve().parents[2]


def _descriptor(
    device: str,
    source_id: str,
    upstream_split: str,
) -> PairDescriptor:
    return PairDescriptor(
        device=device,
        source_id=source_id,
        upstream_split=upstream_split,
        input_member=f"{device}/{upstream_split}/{source_id}-input.wav",
        target_member=f"{device}/{upstream_split}/{source_id}-target.wav",
        source_rate_hz=44_100 if device == "bigmuff" else 48_000,
    )


def _synthetic_marked_pair() -> tuple[np.ndarray, np.ndarray]:
    dry = np.zeros(64, dtype=np.float32)
    wet = np.zeros(64, dtype=np.float32)
    dry[[3, 60]] = (0.4, -0.5)
    wet[[4, 59]] = (0.3, 0.45)
    dry[24:40] = np.linspace(-0.2, 0.2, 16, dtype=np.float32)
    wet[24:40] = 0.6 * dry[24:40]
    return dry, wet


def test_tonetwist_contract_pins_exact_public_metadata_without_waveform_reads() -> None:
    contract = load_data_contract(ROOT)
    assert contract["catalog_repository"] == (
        "https://github.com/mcomunita/tonetwist-afx-dataset"
    )
    assert contract["catalog_commit"] == ("76ae7c875781bc7e2cec2df73b395d1a12d9cdbd")
    assert contract["license"] == "CC-BY-NC-4.0"
    assert contract["read_waveforms_during_metadata_audit"] is False
    assert contract["download_confirmation_archives_before_lock"] is False
    assert contract["dry_markers"] == {
        "zenodo_record": 10455730,
        "archive_name": "DRY-with-markers.zip",
        "local_archive": (
            "datasets/raw/external/tone_twist_dry_markers/DRY-with-markers.zip"
        ),
        "size_bytes": 865850744,
        "published_checksum": "md5:bc1d1490f6c6cfe5643c0798eb5fb5a4",
    }

    expected_devices = {
        "fulltone": (
            "Fulltone Full Drive 2",
            "development",
            10794615,
            "Fulltone-FullDrive2.zip",
            "datasets/raw/external/tone_twist_fulltone/Fulltone-FullDrive2.zip",
            1646555282,
            "md5:0bb9809efe4545071ea117f86adce529",
            48_000,
            "shared_dry_internal",
            "V100_T050_O050_B000",
            "V100_T050_O050_B000",
        ),
        "bigmuff": (
            "Electro-Harmonix Big Muff",
            "development",
            10891515,
            "ElectroHarmonix-BigMuff.zip",
            ("datasets/raw/external/tone_twist_bigmuff/ElectroHarmonix-BigMuff.zip"),
            49975728,
            "md5:45bdd8ea776e1182b9db1d60f62b7930",
            44_100,
            "published_train_val_test",
            "S050_V100",
            "S050_V100",
        ),
        "ampeg": (
            "Ampeg Optocomp",
            "development",
            10465454,
            "Ampeg-OptoComp.zip",
            "datasets/raw/external/tone_twist_ampeg/Ampeg-OptoComp.zip",
            1911870009,
            "md5:024a79afaff34155ff99938863099c67",
            48_000,
            "shared_dry_internal",
            "C050_R050_O060",
            "C050_R050_L060",
        ),
        "rodent": (
            "Harley Benton Rodent",
            "sealed_confirmation",
            10796378,
            "HarleyBenton-Rodent.zip",
            None,
            2913794564,
            "md5:7ab43d078cc195f7595c8b1e1fd09723",
            48_000,
            "shared_dry_internal",
            "V100_F050_D050_MNormal",
            None,
        ),
        "fuzzy_logic": (
            "Harley Benton Fuzzy Logic",
            "sealed_confirmation",
            10796322,
            "HarleyBenton-FuzzyLogic.zip",
            None,
            1272358120,
            "md5:a702262de22616643a47df5d090c68e1",
            48_000,
            "shared_dry_internal",
            "V100_F050",
            None,
        ),
    }
    assert tuple(contract["devices"]) == CAMPAIGN_DEVICES
    for device, expected in expected_devices.items():
        declaration = contract["devices"][device]
        assert tuple(declaration.values()) == expected


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("catalog_repository",), "https://example.invalid/catalog"),
        (("dry_markers", "zenodo_record"), 0),
        (("devices", "rodent", "size_bytes"), 1),
        (("resampling", "beta"), 9.0),
        (("split_policy", "shared_dry_internal", "validation"), "other"),
        (("test_access", "all_five_test_directories_open_together"), False),
        (("pair_audit", "marker_alignment_tolerance_samples"), 2),
    ],
)
def test_tonetwist_contract_rejects_metadata_drift(
    path: tuple[str, ...], replacement: object
) -> None:
    contract = copy.deepcopy(load_data_contract(ROOT))
    target: dict[str, Any] = contract
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(QualityTeacherDataError):
        validate_data_contract(contract)


@pytest.mark.parametrize("device", ["fulltone", "ampeg", "rodent", "fuzzy_logic"])
def test_internal_tonetwist_archives_preserve_source_groups(device: str) -> None:
    assert assigned_split(device, "idmt-gtr4-sg", "trainval") == "validation"
    assert assigned_split(device, "egdb-gtr1", "trainval") == "train"
    assert assigned_split(device, "upstream-test-source", "test") == "test"


def test_big_muff_keeps_its_published_split() -> None:
    assert assigned_split("bigmuff", "published-train", "trainval") == "train"
    assert assigned_split("bigmuff", "published-val", "trainval") == "validation"
    assert assigned_split("bigmuff", "published-test", "test") == "test"
    with pytest.raises(QualityTeacherDataError, match="published split changed"):
        assigned_split("bigmuff", "idmt-gtr4-sg", "trainval")


def test_source_isolation_rejects_any_group_crossing_a_split_boundary() -> None:
    validate_source_isolation(
        [
            _descriptor("fulltone", "train-source", "trainval"),
            _descriptor("ampeg", "idmt-gtr4-sg", "trainval"),
            _descriptor("rodent", "sealed-source", "test"),
        ]
    )
    with pytest.raises(QualityTeacherDataError, match="crosses split boundary"):
        validate_source_isolation(
            [
                _descriptor("fulltone", "shared-source", "trainval"),
                _descriptor("fulltone", "shared-source", "test"),
            ]
        )
    with pytest.raises(QualityTeacherDataError, match="crosses split boundary"):
        validate_source_isolation(
            [
                _descriptor("fulltone", "shared-dry-source", "trainval"),
                _descriptor("ampeg", "shared-dry-source", "test"),
            ]
        )


def test_test_paths_remain_unmaterialized_until_all_four_locks_pass(
    tmp_path: Path,
) -> None:
    pair = _descriptor("rodent", "published-test", "test")
    with pytest.raises(SealedTestAccessError, match="before freeze"):
        materialize_pair_paths(tmp_path, pair)

    all_passed = dict.fromkeys(REQUIRED_TEST_GATES, "passed")
    for missing_gate in REQUIRED_TEST_GATES:
        incomplete = dict(all_passed)
        incomplete[missing_gate] = "pending"
        with pytest.raises(SealedTestAccessError, match=missing_gate):
            authorize_test_access(incomplete)

    authorization = authorize_test_access(all_passed)
    for device in CAMPAIGN_DEVICES:
        materialized = materialize_pair_paths(
            tmp_path,
            _descriptor(device, "published-test", "test"),
            authorization=authorization,
        )
        assert materialized.split == "test"
        assert materialized.input_path == (
            tmp_path
            / "datasets/raw/amp_quality_teacher_v1"
            / f"{device}/test/published-test-input.wav"
        )


def test_pair_audit_accepts_aligned_finite_unclipped_synthetic_arrays() -> None:
    dry, wet = _synthetic_marked_pair()
    audit = audit_pair_arrays(
        dry,
        wet,
        sample_rate_hz=48_000,
        marker_search_samples=8,
    )
    assert audit.samples == 64
    assert audit.dry_marker_indices == (3, 60)
    assert audit.wet_marker_indices == (4, 59)
    assert audit.clipping_samples == 0


def test_pair_audit_rejects_shape_finiteness_markers_alignment_and_clipping() -> None:
    dry, wet = _synthetic_marked_pair()
    with pytest.raises(QualityTeacherDataError, match="equal-length mono"):
        audit_pair_arrays(dry, wet[:-1], sample_rate_hz=48_000)

    nonfinite = wet.copy()
    nonfinite[20] = np.nan
    with pytest.raises(QualityTeacherDataError, match="non-finite"):
        audit_pair_arrays(dry, nonfinite, sample_rate_hz=48_000)

    missing_marker = wet.copy()
    missing_marker[-8:] = 0.0
    with pytest.raises(QualityTeacherDataError, match="marker is missing"):
        audit_pair_arrays(
            dry,
            missing_marker,
            sample_rate_hz=48_000,
            marker_search_samples=8,
        )

    misaligned = wet.copy()
    misaligned[:8] = 0.0
    misaligned[6] = 0.4
    with pytest.raises(QualityTeacherDataError, match="misaligned"):
        audit_pair_arrays(
            dry,
            misaligned,
            sample_rate_hz=48_000,
            marker_search_samples=8,
        )

    clipped = dry.copy()
    clipped[20] = 1.0
    with pytest.raises(QualityTeacherDataError, match="full scale"):
        audit_pair_arrays(
            clipped,
            wet,
            sample_rate_hz=48_000,
            marker_search_samples=8,
        )


def test_pair_preparation_uses_one_joint_chain_and_removes_120_second_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = 121 * 44_100
    dry = np.broadcast_to(np.float32(0.2), (samples,))
    wet = np.broadcast_to(np.float32(-0.1), (samples,))
    sentinel = object()
    call: dict[str, object] = {}

    def fake_prepare_pair(
        input_signal: object,
        target_signal: object,
        **kwargs: object,
    ) -> object:
        call.update(
            input_signal=input_signal,
            target_signal=target_signal,
            **kwargs,
        )
        return sentinel

    monkeypatch.setattr(quality_teacher_data, "prepare_pair", fake_prepare_pair)
    assert (
        quality_teacher_data.prepare_authorized_pair(
            dry,
            wet,
            source_rate_hz=44_100,
        )
        is sentinel
    )
    assert call == {
        "input_signal": dry,
        "target_signal": wet,
        "source_rate": 44_100,
        "target_rate": 48_000,
        "trim_start_seconds": 0.0,
        "trim_end_seconds": 0.0,
        "maximum_seconds": None,
    }


def test_pair_preparation_resamples_without_independent_normalization() -> None:
    dry = np.linspace(-0.4, 0.4, 4_410, dtype=np.float32)
    wet = -0.5 * dry
    pair = prepare_authorized_pair(dry, wet, source_rate_hz=44_100)
    assert pair.sample_rate == 48_000
    assert pair.input.shape == pair.target.shape == (4_800,)
    np.testing.assert_allclose(pair.target, -0.5 * pair.input, atol=2.0e-6)


@pytest.mark.parametrize(
    ("correction", "value"),
    [
        ("gain", 0.01),
        ("dc", -0.001),
        ("integer_delay", 1),
        ("fractional_delay", 0.5),
        ("normalization", "peak"),
    ],
)
def test_post_training_corrections_are_rejected(correction: str, value: object) -> None:
    with pytest.raises(QualityTeacherDataError, match=correction):
        reject_post_training_corrections({correction: value})
    reject_post_training_corrections({correction: "none"})
