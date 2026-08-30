from __future__ import annotations

import importlib.util
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
