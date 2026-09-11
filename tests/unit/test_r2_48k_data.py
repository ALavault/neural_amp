from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from fssr_nam.data.r2_48k import R248KDataAuditError, audit_r2_48k_archive

ROOT = Path(__file__).resolve().parents[2]


def test_r2_48k_archive_audit_is_metadata_only_and_preserves_roles() -> None:
    report = audit_r2_48k_archive(ROOT)
    assert report["status"] == "passed"
    assert report["sample_rate_hz"] == 48_000
    assert report["file_pairs"] == 18
    assert report["waveform_samples_read"] is False
    assert report["internal_validation_test_waveforms_opened"] is False
    assert report["development_tests_previously_observed"] is True
    assert report["physical_192khz_reference_available"] is False
    assert report["fm9_proxy_used"] is False
    assert report["split_counts"]["ua1176"] == {
        "train": 5,
        "validation": 2,
        "test": 2,
    }


def test_r2_48k_archive_rejects_192k_or_proxy_relabelling(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (ROOT / "configs/data/r2_48k_archive.yaml").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (ROOT / "datasets/manifests/r1_physical.json").read_text(encoding="utf-8")
    )
    (tmp_path / "configs/data").mkdir(parents=True)
    (tmp_path / "datasets/manifests").mkdir(parents=True)
    config["proxy_hardware_allowed"] = True
    (tmp_path / "configs/data/r2_48k_archive.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    (tmp_path / "datasets/manifests/r1_physical.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(R248KDataAuditError, match="proxy_hardware_allowed"):
        audit_r2_48k_archive(tmp_path, require_prepared_files=False)

    config["proxy_hardware_allowed"] = False
    manifest["target_sample_rate"] = 192_000
    (tmp_path / "configs/data/r2_48k_archive.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    (tmp_path / "datasets/manifests/r1_physical.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(R248KDataAuditError, match="not 48 kHz"):
        audit_r2_48k_archive(tmp_path, require_prepared_files=False)
