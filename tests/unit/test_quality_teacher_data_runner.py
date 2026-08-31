from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

from fssr_nam.data.quality_teacher import assigned_split, load_data_contract

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_quality_teacher_data_audit_test",
    ROOT / "scripts/run_quality_teacher_data_audit.py",
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def test_data_runner_resolves_every_development_trainval_pair_only() -> None:
    pairs = RUNNER.build_pair_specs(load_data_contract(ROOT))
    assert len(pairs) == 12
    assert Counter(pair.descriptor.device for pair in pairs) == {
        "fulltone": 5,
        "ampeg": 5,
        "bigmuff": 2,
    }
    splits = Counter(
        assigned_split(
            pair.descriptor.device,
            pair.descriptor.source_id,
            pair.descriptor.upstream_split,
        )
        for pair in pairs
    )
    assert splits == {"train": 9, "validation": 3}
    for pair in pairs:
        assert "/trainval/" in pair.descriptor.input_member
        assert "/trainval/" in pair.descriptor.target_member
        assert "/test/" not in pair.descriptor.input_member
        assert "/test/" not in pair.descriptor.target_member
    ampeg_targets = [
        pair.descriptor.target_member
        for pair in pairs
        if pair.descriptor.device == "ampeg"
    ]
    assert all("C050_R050_L060" in member for member in ampeg_targets)
    assert all("O060" not in member for member in ampeg_targets)


def test_data_runner_md5_is_only_the_published_external_checksum(
    tmp_path: Path,
) -> None:
    path = tmp_path / "archive.zip"
    path.write_bytes(b"published archive fixture")
    assert RUNNER._published_md5(path) == "md5:2bec4ec3ffd6b1b766e141727753e1ea"


def test_data_runner_rejects_test_member_before_archive_access(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="restricted to trainval"):
        RUNNER._read_member(tmp_path / "absent.zip", "device/test/secret.wav")
