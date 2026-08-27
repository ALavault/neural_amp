from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from fssr_nam.campaign.r1 import (
    INFRASTRUCTURE_REPLACEMENT_RUN_ID,
    R1AuthorizationError,
    R1ConfigError,
    R1Executor,
    R1RunReuseError,
    RunSpec,
    expand_confirmatory_conditions,
    expand_stage,
    make_run_id,
    parse_run_id,
    validate_confirmatory_lock,
    validate_gate_authorization,
    validate_repository_configs,
)
from fssr_nam.reporting import r1_preflight
from fssr_nam.reporting.ledger import read_runs

ROOT = Path(__file__).resolve().parents[2]
DATA_HASH = "a" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_infrastructure_amendment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    run_id = "r1_competence_bigmuff_lstm64_wright_seed0_v1"
    run_dir = tmp_path / "experiments/runs" / run_id
    run_dir.mkdir(parents=True)
    failure_reason = "RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED"
    status_path = run_dir / "status.json"
    status_path.write_text(
        json.dumps({"status": "failed", "failure_reason": failure_reason}) + "\n",
        encoding="utf-8",
    )
    run_spec_path = run_dir / "run-spec.json"
    run_spec_path.write_text(json.dumps({"run_id": run_id}) + "\n", encoding="utf-8")
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps({"status": "unavailable"}) + "\n", encoding="utf-8"
    )
    audit_path = tmp_path / "experiments/summaries/failure.json"
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(json.dumps({"status": "invalid"}) + "\n", encoding="utf-8")
    reclassification_path = tmp_path / ".codex_campaign/r1/reclassification.json"
    reclassification_path.parent.mkdir(parents=True)
    reclassification_path.write_text(
        json.dumps({"scientific_status": "invalid"}) + "\n", encoding="utf-8"
    )
    entry = {
        "run_id": run_id,
        "status": "failed",
        "failure_reason": failure_reason,
        "commit": "e" * 40,
        "config_sha256": "c" * 64,
        "data_sha256": "d" * 64,
    }
    ledger = tmp_path / ".codex_campaign/RUN_LEDGER.jsonl"
    ledger.parent.mkdir(exist_ok=True)
    ledger.write_text(json.dumps(entry, separators=(",", ":")) + "\n", encoding="utf-8")
    amendment: dict[str, object] = {
        "amendment_number": 2,
        "protocol_revision": "FSSR-R1-v1-a2",
        "post_observation_deviation": True,
        "capacity_accounting_changed": True,
        "scientific_model_data_metrics_changed": False,
        "invalid_run_id": run_id,
        "invalid_scientific_status": "invalid_infrastructure",
        "invalid_run_counts_toward_cap": False,
        "replacement_run_id": INFRASTRUCTURE_REPLACEMENT_RUN_ID,
        "replacement_counts_toward_cap": True,
        "maximum_replacements": 1,
        "further_replacement_authorized": False,
        "evaluation_chunk_samples": 32768,
        "external_report_only_locked": True,
        "failure_signature": "CUDNN_STATUS_NOT_SUPPORTED",
        "declared_condition_run_id": run_id,
        "invalid_run_commit": "e" * 40,
        "invalid_config_sha256": "c" * 64,
        "invalid_data_sha256": "d" * 64,
        "invalid_status_sha256": _sha256(status_path),
        "invalid_failure_audit_path": str(audit_path.relative_to(tmp_path)),
        "invalid_failure_audit_sha256": _sha256(audit_path),
        "invalid_ledger_entry_sha256": _sha256(ledger),
        "invalid_run_spec_sha256": _sha256(run_spec_path),
        "invalid_metrics_sha256": _sha256(metrics_path),
        "reclassification_path": str(reclassification_path.relative_to(tmp_path)),
        "reclassification_sha256": _sha256(reclassification_path),
    }
    monkeypatch.setattr(
        r1_preflight,
        "load_active_infrastructure_amendment",
        lambda root: amendment,
    )
    return amendment


def _stage_configs() -> dict[str, dict[str, object]]:
    return validate_repository_configs(ROOT)


def _valid_lock() -> dict[str, object]:
    return {
        "confirmation_runs_authorized": True,
        "hypothesis": "H1",
        "python_cpp_parity": True,
        "external_report_only_locked": True,
        "candidate": "rf2047",
        "ablation": "rf31",
        "loss_mode": "wright",
        "protocol_sha256": "b" * 64,
    }


def test_repository_declarations_expand_to_exact_caps() -> None:
    configs = _stage_configs()
    assert {stage: len(expand_stage(config)) for stage, config in configs.items()} == {
        "competence": 3,
        "factorial": 8,
        "horizon": 4,
        "cascade": 4,
        "confirm": 76,
    }


def test_confirmatory_expansion_is_exact_and_balanced() -> None:
    specs = expand_confirmatory_conditions(
        _stage_configs()["confirm"], loss_mode="wright"
    )
    assert len(specs) == len({spec.run_id for spec in specs}) == 76
    for device in ("fulltone", "bigmuff", "blackstar", "ua1176"):
        by_family = {
            family: sorted(
                spec.seed
                for spec in specs
                if spec.device == device and spec.model == family
            )
            for family in (
                "a2",
                "lstm_cost_equivalent",
                "nablafx_graybox",
                "r1_ablation",
                "r1_final",
            )
        }
        assert by_family["a2"] == [0, 1, 2, 3, 4]
        assert by_family["r1_final"] == [0, 1, 2, 3, 4]
        assert by_family["lstm_cost_equivalent"] == [0, 1, 2]
        assert by_family["nablafx_graybox"] == [0, 1, 2]
        assert by_family["r1_ablation"] == [0, 1, 2]


def test_run_identifier_is_strict_and_unambiguous() -> None:
    run_id = make_run_id("confirm", "ua1176", "r1_final", "wright", 4)
    assert run_id == "r1_confirm_ua1176_r1-final_wright_seed4_v1"
    assert parse_run_id(run_id).run_id == run_id
    for invalid in (
        "r1_confirm_ua1176_r1_final_wright_seed4_v1",
        "r1_confirm_ua1176_r1-final_wright_seed04_v1",
        "r1_confirm_ua1176_r1-final_wright_seed4_v2",
        "R1_confirm_ua1176_r1-final_wright_seed4_v1",
    ):
        with pytest.raises(ValueError, match="invalid R1 run_id"):
            parse_run_id(invalid)


def test_sequential_gates_are_explicit() -> None:
    configs = _stage_configs()
    competence_seed1 = expand_stage(configs["competence"])[1]
    factorial = expand_stage(configs["factorial"])[0]
    physical_cascade = expand_stage(configs["cascade"], resolved_loss="wright")[2]
    with pytest.raises(R1AuthorizationError, match="competence_seed0"):
        validate_gate_authorization(competence_seed1, {})
    validate_gate_authorization(competence_seed1, {"competence_seed0": "passed"})
    with pytest.raises(R1AuthorizationError, match="competence"):
        validate_gate_authorization(factorial, {})
    with pytest.raises(R1AuthorizationError, match="cascade_synthetic"):
        validate_gate_authorization(physical_cascade, {"horizon": "passed"})
    validate_gate_authorization(
        physical_cascade,
        {"horizon": "passed", "cascade_synthetic": "promoted"},
    )


def test_confirmation_requires_h1_or_h2_and_exact_cpp_parity_flag() -> None:
    lock = _valid_lock()
    validate_confirmatory_lock(lock)
    for update, message in (
        ({"hypothesis": "H3"}, "H1 or H2"),
        ({"python_cpp_parity": False}, "python_cpp_parity"),
        ({"external_report_only_locked": False}, "EXTERNAL_REPORT_ONLY"),
        ({"confirmation_runs_authorized": False}, "does not authorize"),
    ):
        invalid = {**lock, **update}
        with pytest.raises(R1AuthorizationError, match=message):
            validate_confirmatory_lock(invalid)


def test_dry_run_validates_without_creating_state(tmp_path: Path) -> None:
    config = _stage_configs()["competence"]
    executor = R1Executor(tmp_path, stage_configs={"competence": config})
    spec = expand_stage(config)[0]
    result = executor.prepare_run(
        spec,
        data_sha256=DATA_HASH,
        commit="deadbeef",
        dry_run=True,
    )
    assert result["action"] == "would_reserve"
    assert not (tmp_path / "experiments").exists()
    assert not (tmp_path / ".codex_campaign").exists()


def test_existing_run_directory_is_never_reused(tmp_path: Path) -> None:
    config = _stage_configs()["competence"]
    executor = R1Executor(tmp_path, stage_configs={"competence": config})
    spec = expand_stage(config)[0]
    executor.prepare_run(spec, data_sha256=DATA_HASH, commit="deadbeef")
    with pytest.raises(R1RunReuseError, match="already exists"):
        executor.prepare_run(spec, data_sha256=DATA_HASH, commit="deadbeef")


@pytest.mark.parametrize("status", ["failed", "stopped_by_gate"])
def test_failed_and_gate_stopped_runs_are_appended_globally(
    tmp_path: Path, status: str
) -> None:
    config = _stage_configs()["competence"]
    executor = R1Executor(tmp_path, stage_configs={"competence": config})
    spec = expand_stage(config)[0]
    executor.prepare_run(spec, data_sha256=DATA_HASH, commit="deadbeef")
    entry = executor.finalize_run(
        spec, status=status, failure_reason="registered test failure"
    )
    assert entry["status"] == status
    assert read_runs(tmp_path / ".codex_campaign/RUN_LEDGER.jsonl") == [entry]
    persisted = json.loads(
        (tmp_path / "experiments/runs" / spec.run_id / "status.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted["status"] == status
    assert executor.stage_usage("competence") == (1, 3)


def test_stage_cap_counts_all_terminal_statuses(tmp_path: Path) -> None:
    config = _stage_configs()["competence"]
    executor = R1Executor(tmp_path, stage_configs={"competence": config})
    ledger = tmp_path / ".codex_campaign/RUN_LEDGER.jsonl"
    ledger.parent.mkdir(parents=True)
    entries = []
    for seed, status in enumerate(("completed", "failed", "stopped_by_gate")):
        spec = RunSpec("competence", "bigmuff", "lstm64", "wright", seed)
        entries.append(
            {
                "date": "2026-08-27T12:00:00+02:00",
                "run_id": spec.run_id,
                "phase": "R1_COMPETENCE",
                "model": spec.model,
                "device": spec.device,
                "seed": seed,
                "commit": "deadbeef",
                "config_sha256": "b" * 64,
                "data_sha256": DATA_HASH,
                "status": status,
                "failure_reason": "" if status == "completed" else "test",
                "results_path": f"experiments/runs/{spec.run_id}",
            }
        )
    ledger.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )
    assert executor.stage_usage("competence") == (3, 3)


def test_quarantined_noncanonical_attempt_does_not_consume_cap(
    tmp_path: Path,
) -> None:
    config = _stage_configs()["competence"]
    executor = R1Executor(tmp_path, stage_configs={"competence": config})
    ledger = tmp_path / ".codex_campaign/RUN_LEDGER.jsonl"
    ledger.parent.mkdir(parents=True)
    run_id = "r1_competence_bigmuff_wright_lstm64_wright_seed0_v1"
    ledger.write_text(
        json.dumps({"run_id": run_id, "status": "failed"}) + "\n",
        encoding="utf-8",
    )
    assert executor.stage_usage("competence") == (0, 3)
    assert executor.quarantined_noncanonical_attempts()[0]["run_id"] == run_id


def test_amendment_replaces_only_the_invalid_seed_zero_condition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_infrastructure_amendment(tmp_path, monkeypatch)
    config = _stage_configs()["competence"]
    executor = R1Executor(tmp_path, stage_configs={"competence": config})
    assert [spec.run_id for spec in executor.plan("competence")] == [
        INFRASTRUCTURE_REPLACEMENT_RUN_ID,
        "r1_competence_bigmuff_lstm64_wright_seed1_v1",
        "r1_competence_bigmuff_lstm64_wright_seed2_v1",
    ]
    assert executor.stage_usage("competence") == (0, 3)
    replacement = parse_run_id(INFRASTRUCTURE_REPLACEMENT_RUN_ID)
    executor.prepare_run(
        replacement,
        data_sha256=DATA_HASH,
        commit="f" * 40,
    )
    assert executor.stage_usage("competence") == (1, 3)
    with pytest.raises(R1RunReuseError):
        executor.prepare_run(
            replacement,
            data_sha256=DATA_HASH,
            commit="f" * 40,
        )


def test_amendment_does_not_authorize_a_second_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_infrastructure_amendment(tmp_path, monkeypatch)
    executor = R1Executor(
        tmp_path,
        stage_configs={"competence": _stage_configs()["competence"]},
    )
    retry_two = RunSpec("competence", "bigmuff", "lstm64-retry2", "wright", 0)
    with pytest.raises(R1ConfigError, match="outside the declared matrix"):
        executor.prepare_run(
            retry_two,
            data_sha256=DATA_HASH,
            commit="f" * 40,
            dry_run=True,
        )


def test_confirmatory_pending_lock_refuses_reservation(tmp_path: Path) -> None:
    config = _stage_configs()["confirm"]
    executor = R1Executor(tmp_path, stage_configs={"confirm": config})
    spec = expand_confirmatory_conditions(config, loss_mode="wright")[0]
    pending_lock = yaml.safe_load(
        (ROOT / ".codex_campaign/r1/CONFIRMATORY_LOCK.yaml").read_text(encoding="utf-8")
    )
    with pytest.raises(R1AuthorizationError, match="does not authorize"):
        executor.prepare_run(
            spec,
            data_sha256=DATA_HASH,
            commit="deadbeef",
            confirmatory_lock=pending_lock,
            dry_run=True,
        )
