from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_quality_teacher_preflight_test",
    ROOT / "scripts/run_quality_teacher_preflight.py",
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_preflight_snapshot_guard_requires_clean_non_main_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        ("status", "--porcelain", "--untracked-files=normal"): "",
        ("rev-parse", "HEAD"): "abc123",
        ("rev-parse", "--abbrev-ref", "HEAD"): "quality-teacher-exec",
    }
    monkeypatch.setattr(RUNNER, "_git", lambda *arguments: values[arguments])
    assert RUNNER._assert_clean_execution_snapshot() == "abc123"

    values[("status", "--porcelain", "--untracked-files=normal")] = " M dirty"
    with pytest.raises(RuntimeError, match="clean worktree"):
        RUNNER._assert_clean_execution_snapshot()
    values[("status", "--porcelain", "--untracked-files=normal")] = ""
    values[("rev-parse", "--abbrev-ref", "HEAD")] = "main"
    with pytest.raises(RuntimeError, match="dedicated snapshot branch"):
        RUNNER._assert_clean_execution_snapshot()


def test_preflight_model_contract_is_exact() -> None:
    contract = RUNNER._model_contract()
    assert contract == {
        "candidate_control_state_dict_equal": True,
        "candidate_receptive_field_samples": 8_191,
        "candidate_latency_samples": 32,
        "dense_wavenet_parameters": 21_913,
    }


def test_invalid_attempt_numbers_are_append_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(RUNNER, "CAMPAIGN_DIR", tmp_path)
    assert RUNNER._next_invalid_attempt_number() == 1
    (tmp_path / "PREFLIGHT_ATTEMPT_001_INVALID.json").write_text("{}\n")
    assert RUNNER._next_invalid_attempt_number() == 2
    (tmp_path / "PREFLIGHT_ATTEMPT_bad_INVALID.json").write_text("{}\n")
    with pytest.raises(RuntimeError, match="malformed"):
        RUNNER._next_invalid_attempt_number()
