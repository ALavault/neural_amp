import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from fssr_nam.campaign.r1 import R1Executor, RunSpec
from fssr_nam.reporting.ledger import read_runs
from fssr_nam.training.r1_competence import (
    CompetenceProtocolError,
    evaluate_competence_gate,
    seed_zero_allows_additional_runs,
    validate_competence_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/training/r1_competence.yaml"
LEGACY_RUN_ID = "r1_competence_bigmuff_wright_lstm64_wright_seed0_v1"

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_r1_competence", ROOT / "scripts/run_r1_competence.py"
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_locked_competence_protocol_and_canonical_ids() -> None:
    config = _config()
    validate_competence_protocol(config)
    assert [
        RunSpec("competence", "bigmuff", "lstm64", "wright", seed).run_id
        for seed in config["seeds"]
    ] == [
        "r1_competence_bigmuff_lstm64_wright_seed0_v1",
        "r1_competence_bigmuff_lstm64_wright_seed1_v1",
        "r1_competence_bigmuff_lstm64_wright_seed2_v1",
    ]


def test_locked_competence_protocol_rejects_training_drift() -> None:
    config = copy.deepcopy(_config())
    config["optimizer"]["weight_decay"] = 0.0
    with pytest.raises(CompetenceProtocolError, match="optimizer"):
        validate_competence_protocol(config)

    config = copy.deepcopy(_config())
    config["batch_size"] = True
    with pytest.raises(CompetenceProtocolError, match="batch_size"):
        validate_competence_protocol(config)


def test_seed_zero_and_three_seed_gates_are_inclusive() -> None:
    config = _config()
    assert seed_zero_allows_additional_runs(config, 0.15)
    assert not seed_zero_allows_additional_runs(config, 0.150_000_1)

    stopped = evaluate_competence_gate(config, {0: 0.150_000_1})
    assert stopped.status == "failed"
    assert not stopped.seed_zero_continuation
    assert stopped.median_esr is None

    passed = evaluate_competence_gate(config, {0: 0.15, 1: 0.14, 2: 0.20})
    assert passed.status == "passed"
    assert passed.median_esr == 0.15
    assert passed.maximum_esr == 0.20

    failed = evaluate_competence_gate(config, {0: 0.14, 1: 0.16, 2: 0.19})
    assert failed.status == "failed"
    assert failed.median_esr == 0.16


def test_quarantined_noncanonical_attempt_is_audited_but_not_counted(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / ".codex_campaign/RUN_LEDGER.jsonl"
    ledger.parent.mkdir(parents=True)
    legacy = {
        "run_id": LEGACY_RUN_ID,
        "status": "failed",
        "failure_reason": "unauthorized nonconforming post-lock implementation",
    }
    ledger.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    executor = R1Executor(
        tmp_path,
        stage_configs={"competence": _config()},
    )
    assert executor.stage_usage("competence") == (0, 3)
    assert executor.quarantined_noncanonical_attempts() == (legacy,)


def test_test_audio_guard_precedes_any_data_access() -> None:
    with pytest.raises(RuntimeError, match="sealed test audio"):
        runner._load_audio("test", {}, {})


def test_runner_has_no_resume_interface() -> None:
    with pytest.raises(SystemExit):
        runner._parser().parse_args(["--resume"])


def test_seed_execution_finalizes_through_r1_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    executor = R1Executor(tmp_path, stage_configs={"competence": config})
    spec = RunSpec("competence", "bigmuff", "lstm64", "wright", 0)

    def fake_train_seed(*args, **kwargs) -> dict:
        del kwargs
        run_dir = args[1]
        metrics = {"test_esr": 0.1}
        for relative in (
            "environment.json",
            "training-history.json",
            "test-seal.json",
            "checkpoints/index.json",
        ):
            runner._write_json(run_dir / relative, {})
        runner._write_json(run_dir / "metrics.json", metrics)
        (run_dir / "checkpoints/best-model.pt").write_bytes(b"checkpoint")
        (run_dir / "predictions/test-output.wav").write_bytes(b"prediction")
        return metrics

    monkeypatch.setattr(runner, "_train_seed", fake_train_seed)
    result = runner._run_seed(
        spec,
        executor,
        config,
        {},
        {},
        data_sha256="a" * 64,
        commit="deadbeef",
        command="unit-test",
    )
    assert result == {"test_esr": 0.1}
    entries = read_runs(tmp_path / ".codex_campaign/RUN_LEDGER.jsonl")
    assert len(entries) == 1
    assert entries[0]["run_id"] == spec.run_id
    assert entries[0]["status"] == "completed"


def test_seed_execution_registers_failure_through_r1_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    executor = R1Executor(tmp_path, stage_configs={"competence": config})
    spec = RunSpec("competence", "bigmuff", "lstm64", "wright", 0)

    def fail_training(*args, **kwargs) -> dict:
        del args, kwargs
        raise RuntimeError("injected failure")

    monkeypatch.setattr(runner, "_train_seed", fail_training)
    with pytest.raises(RuntimeError, match="injected failure"):
        runner._run_seed(
            spec,
            executor,
            config,
            {},
            {},
            data_sha256="a" * 64,
            commit="deadbeef",
            command="unit-test",
        )
    entries = read_runs(tmp_path / ".codex_campaign/RUN_LEDGER.jsonl")
    assert len(entries) == 1
    assert entries[0]["run_id"] == spec.run_id
    assert entries[0]["status"] == "failed"
    assert entries[0]["failure_reason"] == "RuntimeError: injected failure"


def test_preflight_is_synthetic_non_scientific_and_write_free() -> None:
    result = runner._run_preflight(1, _config())
    assert result["scientific_result"] is False
    assert result["writes_performed"] is False
    assert result["real_audio_opened"] is False
    assert result["test_audio_opened"] is False
    assert result["optimizer_steps_observed"] == 1
