#!/usr/bin/env python3
"""Execute the frozen synthetic R2-48K mechanism matrix exactly once."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from fssr_nam.campaign.r2_48k import (
    CAMPAIGN_VERSION,
    make_run_id,
    validate_stage_authorization,
)
from fssr_nam.campaign.r2_48k_registry import append_gate_event, gate_decisions
from fssr_nam.metrics.r2_mechanism import qualify_synthetic_mechanism
from fssr_nam.reporting.ledger import append_run, read_runs

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / ".codex_campaign/r2_48k"
SUMMARY_DIR = ROOT / "experiments/summaries/r2_48k"
GLOBAL_LEDGER = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
RUN_ID = make_run_id("mechanism", "synthetic", "analytic", "matrix", 0)
RUN_DIR = ROOT / "experiments/runs" / RUN_ID
RESULT_PATH = RUN_DIR / "result.json"
SUMMARY_PATH = SUMMARY_DIR / "mechanism.json"
FAILURE_PATH = RUN_DIR / "failure.json"
VERDICT_PATH = CAMPAIGN_DIR / "VERDICT.json"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"unsupported resolved-protocol value: {type(value).__name__}")


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def _write_new_json(path: Path, payload: Any) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing == payload:
            return
        raise RuntimeError(f"refusing to overwrite immutable evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def _replace_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _digest(parts: list[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def _validate_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = ROOT / "configs/r2_48k/protocol.yaml"
    lock_path = CAMPAIGN_DIR / "MECHANISM_IMPLEMENTATION_LOCK.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    if protocol.get("campaign_version") != CAMPAIGN_VERSION:
        raise RuntimeError("mechanism protocol campaign version changed")
    expected_lock = {
        "campaign_version": CAMPAIGN_VERSION,
        "status": "frozen_before_first_mechanism_measurement",
        "decision_results_examined_before_freeze": False,
        "criteria_changed": False,
    }
    for key, expected in expected_lock.items():
        if lock.get(key) != expected:
            raise RuntimeError(f"mechanism implementation lock {key} changed")
    if lock.get("reference", {}).get("hardware_audio_used") is not False:
        raise RuntimeError("mechanism lock permits a hardware high-rate reference")
    return _json_safe(protocol), _json_safe(lock)


def _update_maturity(status: str, *, invalid: bool = False) -> None:
    path = CAMPAIGN_DIR / "MATURITY.json"
    maturity = json.loads(path.read_text(encoding="utf-8"))
    maturity["scientific_runs_launched"] = 1
    maturity["status"] = status
    if invalid:
        maturity["current_stage"] = "terminal"
        maturity["gates"]["mechanism"] = "invalid_instrumentation"
        maturity["verdict"] = "INVALID"
    _replace_json(path, maturity)


def _data_recipe(protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "fixtures": protocol["mechanism"]["fixtures"],
        "modes": protocol["mechanism"]["modes"],
        "synthetic_asr": protocol["synthetic_asr"],
        "synthetic_only": True,
    }


def _finalize_existing_failure() -> int:
    failure = json.loads(FAILURE_PATH.read_text(encoding="utf-8"))
    resolved = json.loads(
        (RUN_DIR / "resolved_protocol.json").read_text(encoding="utf-8")
    )
    protocol = resolved["campaign_protocol"]
    commit, _ = _git_state()
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    failure_reason = str(failure["failure_reason"])
    append_run(
        GLOBAL_LEDGER,
        {
            "date": finished_at,
            "run_id": RUN_ID,
            "phase": "R2-48K-MECHANISM",
            "model": "analytic_fixture_matrix",
            "device": "synthetic",
            "seed": 0,
            "commit": commit,
            "config_sha256": _digest([_json_bytes(resolved)]),
            "data_sha256": _digest([_json_bytes(_data_recipe(protocol))]),
            "status": "failed",
            "failure_reason": failure_reason,
            "results_path": str(RUN_DIR.relative_to(ROOT)),
        },
    )
    status = json.loads((RUN_DIR / "status.json").read_text(encoding="utf-8"))
    _replace_json(
        RUN_DIR / "status.json",
        {
            "status": "failed",
            "started_at": status.get("started_at"),
            "finished_at": finished_at,
            "failure_reason": failure_reason,
        },
    )
    verdict = {
        "campaign_version": CAMPAIGN_VERSION,
        "verdict": "INVALID",
        "valid_scientific_gate_result": False,
        "terminal_stage": "mechanism",
        "reason": failure_reason,
        "x2_gate_evaluated": False,
        "physical_audio_samples_read": 0,
        "evidence_path": str(FAILURE_PATH.relative_to(ROOT)),
    }
    _write_new_json(VERDICT_PATH, verdict)
    append_gate_event(
        CAMPAIGN_DIR / "GATE_LEDGER.jsonl",
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "mechanism_instrumentation",
            "status": "invalid",
            "evidence_path": str(FAILURE_PATH.relative_to(ROOT)),
        },
    )
    _update_maturity("terminal_invalid", invalid=True)
    print(json.dumps(verdict, sort_keys=True))
    return 1


def main() -> int:
    ledger_path = CAMPAIGN_DIR / "GATE_LEDGER.jsonl"
    decisions = gate_decisions(ledger_path)
    validate_stage_authorization("mechanism", decisions)
    if "mechanism_x2" in decisions:
        raise RuntimeError("synthetic mechanism gate is already terminal")
    prior_runs = [
        entry for entry in read_runs(GLOBAL_LEDGER) if entry["run_id"] == RUN_ID
    ]
    if prior_runs:
        if prior_runs[0].get("status") == "failed" and FAILURE_PATH.is_file():
            print(
                json.dumps(
                    {
                        "campaign_version": CAMPAIGN_VERSION,
                        "run_id": RUN_ID,
                        "status": "INVALID",
                        "reason": prior_runs[0].get("failure_reason"),
                    },
                    sort_keys=True,
                )
            )
            return 1
        if RESULT_PATH.is_file() and SUMMARY_PATH.is_file():
            print(
                json.dumps(
                    {
                        "campaign_version": CAMPAIGN_VERSION,
                        "run_id": RUN_ID,
                        "status": "immutable_result_already_present",
                        "summary_path": str(SUMMARY_PATH.relative_to(ROOT)),
                    },
                    sort_keys=True,
                )
            )
            return 0
        raise RuntimeError(
            "mechanism run ID is already registered without full evidence"
        )
    if (
        RUN_DIR.is_dir()
        and FAILURE_PATH.is_file()
        and not SUMMARY_PATH.exists()
        and (RUN_DIR / "resolved_protocol.json").is_file()
    ):
        return _finalize_existing_failure()
    if RUN_DIR.exists() or SUMMARY_PATH.exists():
        raise RuntimeError(
            "unregistered mechanism evidence already exists; refusing retry"
        )

    protocol, implementation_lock = _validate_lock()
    resolved_protocol = {
        "campaign_protocol": protocol,
        "mechanism_implementation_lock": implementation_lock,
    }
    # Validate serialization before reserving the immutable scientific run ID.
    config_bytes = _json_bytes(resolved_protocol)
    commit, worktree_dirty = _git_state()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    _write_new_json(
        RUN_DIR / "resolved_protocol.json",
        resolved_protocol,
    )
    _write_new_json(
        RUN_DIR / "status.json",
        {
            "status": "running",
            "started_at": started_at,
            "finished_at": None,
            "failure_reason": "",
        },
    )

    status = "failed"
    failure_reason = ""
    evidence: dict[str, Any] | None = None
    try:
        evidence = qualify_synthetic_mechanism(
            progress=lambda message: print(message, flush=True)
        )
        evidence.update(
            {
                "run_id": RUN_ID,
                "git_commit": commit,
                "worktree_dirty_at_execution": worktree_dirty,
                "implementation_lock_path": (
                    ".codex_campaign/r2_48k/MECHANISM_IMPLEMENTATION_LOCK.yaml"
                ),
            }
        )
        _write_new_json(RESULT_PATH, evidence)
        _write_new_json(SUMMARY_PATH, evidence)
        status = "completed"
    except Exception as error:  # preserve the terminal instrumentation failure
        failure_reason = f"{type(error).__name__}: {error}"
        _write_new_json(
            RUN_DIR / "failure.json",
            {
                "campaign_version": CAMPAIGN_VERSION,
                "run_id": RUN_ID,
                "status": "INVALID",
                "failure_reason": failure_reason,
            },
        )

    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    data_recipe_bytes = _json_bytes(_data_recipe(protocol))
    append_run(
        GLOBAL_LEDGER,
        {
            "date": finished_at,
            "run_id": RUN_ID,
            "phase": "R2-48K-MECHANISM",
            "model": "analytic_fixture_matrix",
            "device": "synthetic",
            "seed": 0,
            "commit": commit,
            "config_sha256": _digest([config_bytes]),
            "data_sha256": _digest([data_recipe_bytes]),
            "status": status,
            "failure_reason": failure_reason,
            "results_path": str(RUN_DIR.relative_to(ROOT)),
        },
    )
    _replace_json(
        RUN_DIR / "status.json",
        {
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "failure_reason": failure_reason,
        },
    )
    if status != "completed" or evidence is None:
        _update_maturity("invalid_mechanism_instrumentation", invalid=True)
        print(
            json.dumps(
                {
                    "campaign_version": CAMPAIGN_VERSION,
                    "run_id": RUN_ID,
                    "status": "INVALID",
                    "reason": failure_reason,
                },
                sort_keys=True,
            )
        )
        return 1
    _update_maturity("mechanism_measurement_complete_gate_pending")
    print(
        json.dumps(
            {
                "campaign_version": CAMPAIGN_VERSION,
                "run_id": RUN_ID,
                "status": "completed",
                "rows": len(evidence["rows"]),
                "physical_audio_samples_read": 0,
                "summary_path": str(SUMMARY_PATH.relative_to(ROOT)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
