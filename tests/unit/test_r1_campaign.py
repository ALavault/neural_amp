from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from fssr_nam.campaign.r1 import (
    R1AuthorizationError,
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
from fssr_nam.reporting.ledger import read_runs

ROOT = Path(__file__).resolve().parents[2]
DATA_HASH = "a" * 64


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
