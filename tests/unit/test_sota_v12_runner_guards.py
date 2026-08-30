from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


competence = _load_script(
    "run_sota_v12_competence_test", "scripts/run_sota_v12_competence.py"
)
slow_value = _load_script(
    "run_sota_v12_slow_value_test", "scripts/run_sota_v12_slow_value.py"
)
preflight = _load_script(
    "run_sota_v12_preflight_test", "scripts/run_sota_v12_preflight.py"
)


def _protocol() -> dict[str, object]:
    return yaml.safe_load(
        (ROOT / "configs/amp_sota_prototype_v1_2/protocol.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_v12_runner_source_snapshots_are_closed() -> None:
    for relative in set(competence.SOURCE_FILES) | set(slow_value.SOURCE_FILES):
        assert (ROOT / relative).is_file(), relative


def test_v12_runner_budget_guard_reserves_next_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol()
    monkeypatch.setattr(slow_value, "_used_gpu_hours", lambda: 107.9)
    monkeypatch.setattr(slow_value, "_run_disk_gib", lambda: 0.0)
    with pytest.raises(RuntimeError, match="GPU budget"):
        slow_value._ensure_capacity(protocol, 0.2)

    monkeypatch.setattr(slow_value, "_used_gpu_hours", lambda: 0.0)
    monkeypatch.setattr(slow_value, "_run_disk_gib", lambda: 19.96)
    with pytest.raises(RuntimeError, match="disk budget"):
        slow_value._ensure_capacity(protocol, 0.0)


def test_v12_slow_runner_checks_saved_input_and_target_pairing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(slow_value, "ROOT", tmp_path)
    source = np.arange(16, dtype=np.float32)
    target = np.tanh(source)
    candidate_paths = []
    control_paths = []
    for episode in range(4):
        candidate = tmp_path / f"candidate_{episode}.npz"
        control = tmp_path / f"control_{episode}.npz"
        np.savez(candidate, input=source, target=target, prediction=source)
        np.savez(control, input=source, target=target, prediction=source)
        candidate_paths.append(candidate.name)
        control_paths.append(control.name)
    candidate_rows = [
        {
            "system": "dynamic_primary",
            "seed": 0,
            "prediction_paths": candidate_paths,
        }
    ]
    control_rows = [
        {
            "system": "dynamic_primary",
            "seed": 0,
            "prediction_paths": control_paths,
        }
    ]
    slow_value._assert_prediction_pairing(candidate_rows, control_rows)
    np.savez(
        tmp_path / control_paths[-1],
        input=source,
        target=target + 1.0,
        prediction=source,
    )
    with pytest.raises(RuntimeError, match="targets are not paired"):
        slow_value._assert_prediction_pairing(candidate_rows, control_rows)


def test_v12_failed_provenance_has_explicit_ledger_marker() -> None:
    entry = competence._ledger_entry(
        run_id="fixture",
        system="dynamic_primary",
        seed=0,
        status="failed",
        failure_reason="capture failed",
        run_dir=competence.ROOT / "experiments/runs/fixture",
        provenance=None,
        config_digest="a" * 64,
        data_digest="b" * 64,
        elapsed_seconds=0.0,
    )
    assert entry["commit"] == "PROVENANCE_CAPTURE_FAILED"


def test_v12_preflight_records_infrastructure_failure_without_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_dir = tmp_path / ".codex_campaign/amp_sota_prototype_v1_2"
    campaign_dir.mkdir(parents=True)
    first = campaign_dir / "PREFLIGHT_ATTEMPT_001_INVALID.json"
    first.write_text("{}\n", encoding="utf-8")
    maturity_path = campaign_dir / "MATURITY.json"
    maturity_path.write_text(
        json.dumps(
            {
                "campaign_version": "AMP-SOTA-PROTOTYPE-v1.2",
                "current_stage": "preflight_pending",
                "infrastructure_invalid_preflight_attempts": 1,
                "scientific_runs_launched": 0,
                "status": "active",
                "verdict": None,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    monkeypatch.setattr(preflight, "CAMPAIGN_DIR", campaign_dir)
    monkeypatch.setattr(
        preflight,
        "GATE_LEDGER",
        campaign_dir / "GATE_LEDGER.jsonl",
    )
    monkeypatch.setattr(
        preflight,
        "SUMMARY",
        tmp_path / "experiments/summaries/amp_sota_prototype_v1_2/preflight.json",
    )
    error = subprocess.CalledProcessError(2, ["make", "test"])
    attempt = preflight._record_invalid_attempt(
        error,
        failure_stage="make_test",
        validation_status={
            "data_audit": "passed",
            "test": "not_run",
            "lint": "not_run",
        },
        execution_commit="deadbeef",
    )
    assert attempt.name == "PREFLIGHT_ATTEMPT_002_INVALID.json"
    record = json.loads(attempt.read_text(encoding="utf-8"))
    assert record["attempt_status"] == "INVALID"
    assert record["gate_evaluated"] is False
    assert record["failed_command"] == ["make", "test"]
    assert record["failed_command_exit_code"] == 2
    assert not preflight.GATE_LEDGER.exists()
    assert not preflight.SUMMARY.exists()
    maturity = json.loads(maturity_path.read_text(encoding="utf-8"))
    assert maturity["infrastructure_invalid_preflight_attempts"] == 2
    assert maturity["status"] == "active"
    assert maturity["verdict"] is None
