"""Fail-closed ToneTwist data contract for AMP-QUALITY-TEACHER-v1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray

from .physical import PreparedPair, prepare_pair

CAMPAIGN_VERSION = "AMP-QUALITY-TEACHER-v1"
DATA_CONFIG_PATH = Path("configs/data/amp_quality_teacher_v1.yaml")
CATALOG_COMMIT = "76ae7c875781bc7e2cec2df73b395d1a12d9cdbd"
DEVELOPMENT_DEVICES = ("fulltone", "bigmuff", "ampeg")
CONFIRMATION_DEVICES = ("rodent", "fuzzy_logic")
CAMPAIGN_DEVICES = (*DEVELOPMENT_DEVICES, *CONFIRMATION_DEVICES)
REQUIRED_TEST_GATES = (
    "candidate_lock",
    "comparator_lock",
    "recipes_lock",
    "confirmation_checkpoints_lock",
)


class QualityTeacherDataError(ValueError):
    """Raised when data metadata or paired samples violate the frozen contract."""


class SealedTestAccessError(PermissionError):
    """Raised before any sealed test path or waveform may be materialized."""


@dataclass(frozen=True)
class PairDescriptor:
    device: str
    source_id: str
    upstream_split: str
    input_member: str
    target_member: str
    source_rate_hz: int


@dataclass(frozen=True)
class MaterializedPair:
    device: str
    source_id: str
    split: str
    input_path: Path
    target_path: Path
    source_rate_hz: int


@dataclass(frozen=True)
class TestAccessAuthorization:
    campaign_version: str
    gates: tuple[str, ...]


@dataclass(frozen=True)
class PairAudit:
    samples: int
    sample_rate_hz: int
    dry_marker_indices: tuple[int, int]
    wet_marker_indices: tuple[int, int]
    dry_peak: float
    wet_peak: float
    clipping_samples: int


def _require_equal(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise QualityTeacherDataError(f"{label} must equal {expected!r}, got {value!r}")


def load_data_contract(root: Path) -> dict[str, Any]:
    path = root / DATA_CONFIG_PATH
    if not path.is_file():
        raise QualityTeacherDataError(f"missing teacher data contract: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise QualityTeacherDataError("teacher data contract must be a mapping")
    validate_data_contract(value)
    return value


def validate_data_contract(contract: Mapping[str, Any]) -> None:
    """Validate metadata only; never open an archive or waveform."""
    expected_top = {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "catalog_repository": "https://github.com/mcomunita/tonetwist-afx-dataset",
        "catalog_commit": CATALOG_COMMIT,
        "license": "CC-BY-NC-4.0",
        "allowed_usage": "noncommercial_research_with_attribution",
        "target_sample_rate_hz": 48_000,
        "channels": 1,
        "precision": "float32",
        "read_waveforms_during_metadata_audit": False,
        "download_confirmation_archives_before_lock": False,
        "maximum_seconds_per_file": None,
        "independent_normalization_allowed": False,
        "post_training_gain_dc_or_delay_correction_allowed": False,
    }
    for key, expected in expected_top.items():
        _require_equal(contract.get(key), expected, key)
    devices = contract.get("devices")
    if not isinstance(devices, Mapping):
        raise QualityTeacherDataError("devices must be a mapping")
    _require_equal(tuple(devices), CAMPAIGN_DEVICES, "device order")
    _require_equal(
        contract.get("dry_markers"),
        {
            "zenodo_record": 10455730,
            "archive_name": "DRY-with-markers.zip",
            "local_archive": (
                "datasets/raw/external/tone_twist_dry_markers/DRY-with-markers.zip"
            ),
            "size_bytes": 865850744,
            "published_checksum": "md5:bc1d1490f6c6cfe5643c0798eb5fb5a4",
        },
        "dry_markers",
    )
    expected_records = {
        "fulltone": (
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
    fields = (
        "role",
        "zenodo_record",
        "archive_name",
        "local_archive",
        "size_bytes",
        "published_checksum",
        "source_rate_hz",
        "source_layout",
        "setting",
        "archive_setting",
    )
    for device, expected in expected_records.items():
        declaration = devices.get(device)
        if not isinstance(declaration, Mapping):
            raise QualityTeacherDataError(f"{device} declaration must be a mapping")
        _require_equal(
            {field: declaration.get(field) for field in fields},
            dict(zip(fields, expected, strict=True)),
            f"{device}.metadata",
        )
    _require_equal(
        contract.get("resampling"),
        {
            "method": "scipy_signal_resample_poly",
            "window": "kaiser",
            "beta": 8.6,
            "dry_and_wet_processed_together": True,
            "normalization": "none",
            "delay_correction": "none",
            "dc_correction": "none",
        },
        "resampling",
    )
    split = contract.get("split_policy", {})
    _require_equal(
        split.get("expected_internal_trainval_sources"),
        ["idmt-gtr2", "idmt-gtr4-sg", "nam", "prvt-gtr", "yt-bass"],
        "internal trainval sources",
    )
    _require_equal(
        split.get("shared_dry_internal"),
        {
            "train": "every_trainval_source_except_idmt-gtr4-sg",
            "validation": "idmt-gtr4-sg",
            "test": "upstream_test_directory_sealed",
        },
        "split shared dry",
    )
    _require_equal(
        split.get("published_train_val_test"),
        {
            "train": "published-train",
            "validation": "published-val",
            "test": "published-test_sealed",
        },
        "split published",
    )
    _require_equal(split.get("preserve_source_groups"), True, "split groups")
    _require_equal(
        split.get("cross_split_source_overlap_allowed"), False, "split overlap"
    )
    access = contract.get("test_access", {})
    _require_equal(
        access.get("all_five_test_directories_open_together"),
        True,
        "test single opening",
    )
    _require_equal(
        access.get("development_tests_selection_eligible"),
        False,
        "test development selection",
    )
    _require_equal(
        access.get("required_passed_gates"), list(REQUIRED_TEST_GATES), "test gates"
    )
    _require_equal(
        access.get("confirmation_primary_devices"),
        list(CONFIRMATION_DEVICES),
        "confirmation devices",
    )
    closed = contract.get("closed_resources", {})
    _require_equal(closed.get("devices"), ["blackstar", "ua1176"], "closed devices")
    _require_equal(closed.get("tiers"), ["EXTERNAL_REPORT_ONLY"], "closed tiers")
    audit = contract.get("pair_audit", {})
    _require_equal(
        audit.get("checks"),
        [
            "license",
            "published_checksum",
            "synchronization_markers",
            "equal_length_alignment",
            "finite_samples",
            "clipping",
            "dry_wet_correspondence",
        ],
        "pair audit checks",
    )
    _require_equal(audit.get("marker_search_samples"), 48_000, "marker search")
    _require_equal(
        audit.get("marker_alignment_tolerance_samples"), 1, "marker tolerance"
    )
    _require_equal(audit.get("clipping_absolute_threshold"), 1.0, "clipping")


def assigned_split(device: str, source_id: str, upstream_split: str) -> str:
    """Map one upstream source group without inspecting waveform content."""
    if device not in CAMPAIGN_DEVICES:
        raise QualityTeacherDataError(f"device is outside campaign: {device}")
    if not source_id:
        raise QualityTeacherDataError("source_id must be nonempty")
    if upstream_split == "test":
        return "test"
    if device == "bigmuff":
        mapping = {
            ("published-train", "trainval"): "train",
            ("published-val", "trainval"): "validation",
        }
        try:
            return mapping[(source_id, upstream_split)]
        except KeyError as error:
            raise QualityTeacherDataError("Big Muff published split changed") from error
    if upstream_split != "trainval":
        raise QualityTeacherDataError(
            "internal ToneTwist source is outside trainval/test"
        )
    return "validation" if source_id == "idmt-gtr4-sg" else "train"


def validate_source_isolation(pairs: Sequence[PairDescriptor]) -> None:
    """Reject any source group reused across train, validation, or test."""
    observed: dict[str, str] = {}
    for pair in pairs:
        split = assigned_split(pair.device, pair.source_id, pair.upstream_split)
        prior = observed.get(pair.source_id)
        if prior is not None and prior != split:
            raise QualityTeacherDataError(
                f"source group crosses split boundary: {pair.source_id}"
            )
        observed[pair.source_id] = split


def authorize_test_access(
    gate_decisions: Mapping[str, str],
) -> TestAccessAuthorization:
    """Create a token only after every candidate, recipe, and checkpoint lock."""
    missing = [
        gate for gate in REQUIRED_TEST_GATES if gate_decisions.get(gate) != "passed"
    ]
    if missing:
        raise SealedTestAccessError(
            f"test waveforms remain sealed; missing passed gates: {missing}"
        )
    return TestAccessAuthorization(CAMPAIGN_VERSION, REQUIRED_TEST_GATES)


def materialize_pair_paths(
    root: Path,
    pair: PairDescriptor,
    *,
    authorization: TestAccessAuthorization | None = None,
) -> MaterializedPair:
    """Resolve paths only after enforcing the test boundary."""
    split = assigned_split(pair.device, pair.source_id, pair.upstream_split)
    if split == "test":
        if (
            authorization is None
            or authorization.campaign_version != CAMPAIGN_VERSION
            or authorization.gates != REQUIRED_TEST_GATES
        ):
            raise SealedTestAccessError("test path access is forbidden before freeze")
    base = root / "datasets/raw/amp_quality_teacher_v1"
    members = (Path(pair.input_member), Path(pair.target_member))
    if any(member.is_absolute() or ".." in member.parts for member in members):
        raise QualityTeacherDataError("pair path escaped campaign data root")
    input_path, target_path = (base / member for member in members)
    return MaterializedPair(
        device=pair.device,
        source_id=pair.source_id,
        split=split,
        input_path=input_path,
        target_path=target_path,
        source_rate_hz=pair.source_rate_hz,
    )


def _marker_indices(
    signal: NDArray[np.float32], *, search_samples: int, minimum_amplitude: float
) -> tuple[int, int]:
    window = min(search_samples, len(signal) // 2)
    if window < 1:
        raise QualityTeacherDataError("pair is too short for synchronization markers")
    start = int(np.argmax(np.abs(signal[:window])))
    end_offset = int(np.argmax(np.abs(signal[-window:])))
    end = len(signal) - window + end_offset
    if (
        abs(float(signal[start])) < minimum_amplitude
        or abs(float(signal[end])) < minimum_amplitude
    ):
        raise QualityTeacherDataError("synchronization marker is missing")
    return start, end


def audit_pair_arrays(
    dry: ArrayLike,
    wet: ArrayLike,
    *,
    sample_rate_hz: int,
    marker_search_samples: int = 48_000,
    marker_minimum_amplitude: float = 0.05,
    marker_tolerance_samples: int = 1,
    clipping_threshold: float = 1.0,
) -> PairAudit:
    """Audit one authorized pair before trimming or joint resampling."""
    x = np.asarray(dry, dtype=np.float32)
    y = np.asarray(wet, dtype=np.float32)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or len(x) < 2:
        raise QualityTeacherDataError("dry/wet pair must be equal-length mono audio")
    if sample_rate_hz not in {44_100, 48_000}:
        raise QualityTeacherDataError("source sample rate is outside frozen contract")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise QualityTeacherDataError("dry/wet pair contains non-finite samples")
    dry_markers = _marker_indices(
        x,
        search_samples=marker_search_samples,
        minimum_amplitude=marker_minimum_amplitude,
    )
    wet_markers = _marker_indices(
        y,
        search_samples=marker_search_samples,
        minimum_amplitude=marker_minimum_amplitude,
    )
    if any(
        abs(dry_index - wet_index) > marker_tolerance_samples
        for dry_index, wet_index in zip(dry_markers, wet_markers, strict=True)
    ):
        raise QualityTeacherDataError("dry/wet synchronization markers are misaligned")
    dry_for_clipping = np.abs(x).copy()
    wet_for_clipping = np.abs(y).copy()
    dry_for_clipping[list(dry_markers)] = 0.0
    wet_for_clipping[list(wet_markers)] = 0.0
    clipping = int(
        np.count_nonzero(dry_for_clipping >= clipping_threshold)
        + np.count_nonzero(wet_for_clipping >= clipping_threshold)
    )
    if clipping:
        raise QualityTeacherDataError("dry/wet pair exceeds digital full scale")
    return PairAudit(
        samples=len(x),
        sample_rate_hz=sample_rate_hz,
        dry_marker_indices=dry_markers,
        wet_marker_indices=wet_markers,
        dry_peak=float(np.max(np.abs(x))),
        wet_peak=float(np.max(np.abs(y))),
        clipping_samples=clipping,
    )


def prepare_authorized_pair(
    dry: ArrayLike,
    wet: ArrayLike,
    *,
    source_rate_hz: int,
) -> PreparedPair:
    """Jointly convert the complete eligible pair to 48 kHz without correction."""
    return prepare_pair(
        dry,
        wet,
        source_rate=source_rate_hz,
        target_rate=48_000,
        trim_start_seconds=0.0,
        trim_end_seconds=0.0,
        maximum_seconds=None,
    )


def reject_post_training_corrections(corrections: Mapping[str, object]) -> None:
    prohibited = {"gain", "dc", "integer_delay", "fractional_delay", "normalization"}

    def inactive(value: object) -> bool:
        return (
            value is None
            or value is False
            or (isinstance(value, (int, float)) and value == 0)
            or (isinstance(value, str) and value == "none")
        )

    active = sorted(
        key
        for key, value in corrections.items()
        if key in prohibited and not inactive(value)
    )
    if active:
        raise QualityTeacherDataError(
            f"post-training corrections are forbidden: {active}"
        )
