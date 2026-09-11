#!/usr/bin/env python3
"""Execute the single serialization-corrected R2-48K-v2 mechanism matrix."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import date, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

from fssr_nam.campaign.r2_48k_v2 import (
    CAMPAIGN_VERSION,
    make_run_id,
    validate_repository_configs,
    validate_stage_authorization,
)
from fssr_nam.campaign.r2_48k_v2_gates import validate_mechanism_evidence
from fssr_nam.campaign.r2_48k_v2_registry import append_gate_event, gate_decisions
from fssr_nam.metrics.r2_mechanism import qualify_synthetic_mechanism
from fssr_nam.reporting.json_evidence import (
    dumps_strict_evidence,
    loads_strict_json,
)
from fssr_nam.reporting.ledger import append_run, read_runs

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / ".codex_campaign/r2_48k_v2"
SUMMARY_DIR = ROOT / "experiments/summaries/r2_48k_v2"
GLOBAL_LEDGER = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
RUN_ID = make_run_id("mechanism", "synthetic", "analytic", "matrix", 0)
RUN_DIR = ROOT / "experiments/runs" / RUN_ID
RESULT_PATH = RUN_DIR / "result.json"
SUMMARY_PATH = SUMMARY_DIR / "mechanism.json"


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"unsupported resolved v2 protocol type: {type(value).__name__}")


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def _strict_json(payload: Any) -> str:
    return json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _write_new_text(path: Path, serialized: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") == serialized:
            return
        raise RuntimeError(f"refusing to overwrite immutable v2 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)


def _write_new_json(path: Path, payload: Any) -> None:
    _write_new_text(path, _strict_json(payload))


def _replace_json(path: Path, payload: Any) -> None:
    serialized = _strict_json(payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
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


def _load_implementation_lock() -> dict[str, Any]:
    path = CAMPAIGN_DIR / "MECHANISM_IMPLEMENTATION_LOCK.yaml"
    lock = yaml.safe_load(path.read_text(encoding="utf-8"))
    literals = {
        "campaign_version": CAMPAIGN_VERSION,
        "status": "frozen_before_first_v2_mechanism_measurement",
        "parent_numeric_results_used_for_selection": False,
        "criteria_changed": False,
        "renderer_changed": False,
        "metric_formula_changed": False,
        "gate_thresholds_changed": False,
        "retry_policy": "one_v2_mechanism_run_no_result_dependent_retry",
    }
    for name, expected in literals.items():
        if lock.get(name) != expected:
            raise RuntimeError(f"v2 mechanism lock {name} changed")
    return _json_safe(lock)


def _update_maturity(status: str, *, invalid: bool = False) -> None:
    path = CAMPAIGN_DIR / "MATURITY.json"
    maturity = json.loads(path.read_text(encoding="utf-8"))
    maturity["scientific_runs_launched"] = 1
    maturity["status"] = status
    if invalid:
        maturity["current_stage"] = "terminal"
        maturity["gates"]["mechanism"] = "invalid_instrumentation"
        maturity["verdict"] = "INVALID"
    else:
        maturity["gates"]["mechanism"] = "measurement_complete_gate_pending"
    _replace_json(path, maturity)


def _record_invalid(
    *,
    reason: str,
    commit: str,
    config_digest: str,
    data_digest: str,
    started_at: str,
) -> int:
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    failure_path = RUN_DIR / "failure.json"
    _write_new_json(
        failure_path,
        {
            "campaign_version": CAMPAIGN_VERSION,
            "run_id": RUN_ID,
            "status": "INVALID",
            "failure_reason": reason,
            "physical_audio_samples_read": 0,
        },
    )
    append_run(
        GLOBAL_LEDGER,
        {
            "date": finished_at,
            "run_id": RUN_ID,
            "phase": "R2-48K-V2-MECHANISM",
            "model": "analytic_fixture_matrix",
            "device": "synthetic",
            "seed": 0,
            "commit": commit,
            "config_sha256": config_digest,
            "data_sha256": data_digest,
            "status": "failed",
            "failure_reason": reason,
            "results_path": str(RUN_DIR.relative_to(ROOT)),
        },
    )
    _replace_json(
        RUN_DIR / "status.json",
        {
            "status": "failed",
            "started_at": started_at,
            "finished_at": finished_at,
            "failure_reason": reason,
        },
    )
    verdict = {
        "campaign_version": CAMPAIGN_VERSION,
        "verdict": "INVALID",
        "valid_scientific_gate_result": False,
        "terminal_stage": "mechanism",
        "reason": reason,
        "x2_gate_evaluated": False,
        "physical_audio_samples_read": 0,
        "evidence_path": str(failure_path.relative_to(ROOT)),
    }
    _write_new_json(CAMPAIGN_DIR / "VERDICT.json", verdict)
    append_gate_event(
        CAMPAIGN_DIR / "GATE_LEDGER.jsonl",
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "mechanism_instrumentation",
            "status": "invalid",
            "evidence_path": str(failure_path.relative_to(ROOT)),
        },
    )
    _update_maturity("terminal_invalid", invalid=True)
    print(json.dumps(verdict, allow_nan=False, sort_keys=True))
    return 1


def main() -> int:
    ledger = CAMPAIGN_DIR / "GATE_LEDGER.jsonl"
    decisions = gate_decisions(ledger)
    validate_stage_authorization("mechanism", decisions)
    if "mechanism_x2" in decisions or "mechanism_instrumentation" in decisions:
        raise RuntimeError("v2 mechanism stage is already terminal")
    prior = [entry for entry in read_runs(GLOBAL_LEDGER) if entry["run_id"] == RUN_ID]
    if prior or RUN_DIR.exists() or SUMMARY_PATH.exists():
        raise RuntimeError("v2 mechanism run ID or evidence path is already reserved")

    resolved_protocol = validate_repository_configs(ROOT)
    implementation_lock = _load_implementation_lock()
    resolved = {
        "campaign_protocol": resolved_protocol,
        "mechanism_implementation_lock": implementation_lock,
    }
    resolved_serialized = _strict_json(resolved)
    config_digest = _digest([resolved_serialized.encode("utf-8")])
    data_recipe = {
        "fixtures": resolved_protocol["mechanism"]["fixtures"],
        "modes": resolved_protocol["mechanism"]["modes"],
        "synthetic_asr": resolved_protocol["synthetic_asr"],
        "synthetic_only": True,
        "parent_numeric_evidence_reused": False,
    }
    data_digest = _digest([_strict_json(data_recipe).encode("utf-8")])
    commit, worktree_dirty = _git_state()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")

    RUN_DIR.mkdir(parents=True, exist_ok=False)
    _write_new_text(RUN_DIR / "resolved_protocol.json", resolved_serialized)
    _write_new_json(
        RUN_DIR / "command.json",
        {
            "argv": ["uv", "run", "python", "scripts/run_r2_48k_v2_mechanism.py"],
            "cwd": str(ROOT),
        },
    )
    _write_new_json(
        RUN_DIR / "environment.json",
        {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "torch": version("torch"),
            "hypothesis": version("hypothesis"),
        },
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

    try:
        evidence = qualify_synthetic_mechanism(
            progress=lambda message: print(message, flush=True)
        )
        evidence.update(
            {
                "schema_version": 2,
                "campaign_version": CAMPAIGN_VERSION,
                "parent_campaign_version": "FSSR-R2-48K-v1",
                "run_id": RUN_ID,
                "git_commit": commit,
                "worktree_dirty_at_execution": worktree_dirty,
                "evidence_encoding": "fssr-extended-real-json-v1",
                "parent_numeric_evidence_reused": False,
                "implementation_lock_path": (
                    ".codex_campaign/r2_48k_v2/MECHANISM_IMPLEMENTATION_LOCK.yaml"
                ),
            }
        )
        for row in evidence["rows"]:
            row["weights_id"] = f"r2-v2-analytic-{row['fixture']}-v1"
        serialized_evidence = dumps_strict_evidence(evidence, indent=2) + "\n"
        parsed_evidence = loads_strict_json(serialized_evidence)
        validate_mechanism_evidence(parsed_evidence)
        _write_new_text(RESULT_PATH, serialized_evidence)
        _write_new_text(SUMMARY_PATH, serialized_evidence)
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        return _record_invalid(
            reason=reason,
            commit=commit,
            config_digest=config_digest,
            data_digest=data_digest,
            started_at=started_at,
        )

    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    append_run(
        GLOBAL_LEDGER,
        {
            "date": finished_at,
            "run_id": RUN_ID,
            "phase": "R2-48K-V2-MECHANISM",
            "model": "analytic_fixture_matrix",
            "device": "synthetic",
            "seed": 0,
            "commit": commit,
            "config_sha256": config_digest,
            "data_sha256": data_digest,
            "status": "completed",
            "failure_reason": "",
            "results_path": str(RUN_DIR.relative_to(ROOT)),
        },
    )
    _replace_json(
        RUN_DIR / "status.json",
        {
            "status": "completed",
            "started_at": started_at,
            "finished_at": finished_at,
            "failure_reason": "",
        },
    )
    _update_maturity("mechanism_measurement_complete_gate_pending")
    print(
        json.dumps(
            {
                "campaign_version": CAMPAIGN_VERSION,
                "run_id": RUN_ID,
                "status": "completed",
                "rows": len(parsed_evidence["rows"]),
                "extended_real_diagnostics": validate_mechanism_evidence(
                    parsed_evidence
                )["extended_real_diagnostics"],
                "physical_audio_samples_read": 0,
                "summary_path": str(SUMMARY_PATH.relative_to(ROOT)),
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
