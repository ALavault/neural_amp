#!/usr/bin/env python3
"""Execute the metadata-only AMP-QUALITY-TEACHER-v1 preflight once."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import torch

from fssr_nam.campaign.amp_quality_teacher_registry import (
    append_gate_event,
    gate_decisions,
)
from fssr_nam.campaign.amp_quality_teacher_v1 import (
    CAMPAIGN_VERSION,
    validate_repository_state,
    validate_stage_authorization,
)
from fssr_nam.campaign.quality_aa_provenance import replace_json, write_new_json
from fssr_nam.data.quality_teacher import (
    SealedTestAccessError,
    authorize_test_access,
    load_data_contract,
)
from fssr_nam.models.quality_teacher import (
    QUALITY_TEACHER_FAMILY,
    QUALITY_TEACHER_FAST_CONTROL,
    build_quality_teacher_model,
)
from fssr_nam.models.quality_teacher_comparators import DenseWaveNet16x18

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / ".codex_campaign/amp_quality_teacher_v1"
GATE_LEDGER = CAMPAIGN_DIR / "GATE_LEDGER.jsonl"
SUMMARY = CAMPAIGN_DIR / "PREFLIGHT.json"
GLOBAL_LEDGER = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
MINIMUM_GPU_MEMORY_BYTES = 24_000_000_000


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_validation(target: str) -> None:
    subprocess.run(["make", target], cwd=ROOT, check=True)


def _assert_clean_execution_snapshot() -> str:
    if _git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("quality-teacher preflight requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    if _git("rev-parse", "--abbrev-ref", "HEAD") == "main":
        raise RuntimeError("preflight must execute on a dedicated snapshot branch")
    return commit


def _assert_no_scientific_runs() -> None:
    run_root = ROOT / "experiments/runs"
    if run_root.is_dir() and any(run_root.glob("quality_teacher_v1_*")):
        raise RuntimeError("quality-teacher scientific run exists before preflight")
    if GLOBAL_LEDGER.is_file() and "quality_teacher_v1_" in GLOBAL_LEDGER.read_text(
        encoding="utf-8"
    ):
        raise RuntimeError("quality-teacher ledger event exists before preflight")


def _model_contract() -> dict[str, object]:
    candidate = build_quality_teacher_model(QUALITY_TEACHER_FAMILY, seed=0)
    control = build_quality_teacher_model(QUALITY_TEACHER_FAST_CONTROL, seed=0)
    identical = all(
        torch.equal(value, control.state_dict()[name])
        for name, value in candidate.state_dict().items()
    )
    if not identical:
        raise RuntimeError("candidate/control initialization is not identical")
    if candidate.receptive_field_samples != 8_191 or candidate.latency_samples != 32:
        raise RuntimeError("quality-teacher RF or latency drifted")
    if any(parameter.requires_grad for parameter in control.observer.parameters()):
        raise RuntimeError("fast-only observer parameters are not frozen")
    dense = DenseWaveNet16x18()
    dense_parameters = sum(parameter.numel() for parameter in dense.parameters())
    if dense_parameters != 21_913:
        raise RuntimeError("dense WaveNet comparator parameter count drifted")
    return {
        "candidate_control_state_dict_equal": identical,
        "candidate_receptive_field_samples": candidate.receptive_field_samples,
        "candidate_latency_samples": candidate.latency_samples,
        "dense_wavenet_parameters": dense_parameters,
    }


def _write_state_after_pass() -> None:
    (CAMPAIGN_DIR / "STATE.md").write_text(
        """# État

- Lignée : `AMP-QUALITY-TEACHER-v1`.
- Statut : active, préflight passé; `data_audit` autorisé.
- Parent : `AMP-SOTA-PROTOTYPE-v1.2`, supersédé après préflight passé et avant
  tout run scientifique.
- Runs scientifiques : 0.
- Gates évalués : 1 (`preflight=passed`).
- Waveforms de test lus : 0.
- Archives Rodent/Fuzzy Logic téléchargées : non.
- Blackstar, UA1176 et `EXTERNAL_REPORT_ONLY` : fermés.
- Prochaine action autorisée : audit et préparation des seules données de
  développement; aucun chemin `test` ne peut être matérialisé.
""",
        encoding="utf-8",
    )
    (CAMPAIGN_DIR / "HANDOFF.md").write_text(
        """# Handoff

Le préflight a passé depuis un snapshot propre. Exécuter `data_audit` sur les
seules données de développement, sans télécharger Rodent/Fuzzy Logic ni
matérialiser un chemin `test`. Aucun run scientifique n'est autorisé avant ce
gate.
""",
        encoding="utf-8",
    )


def _record_invalid(
    error: BaseException,
    *,
    execution_commit: str,
    failure_stage: str,
    validation_status: dict[str, str],
) -> None:
    path = CAMPAIGN_DIR / "PREFLIGHT_ATTEMPT_001_INVALID.json"
    record = {
        "attempt_status": "INVALID",
        "campaign_version": CAMPAIGN_VERSION,
        "campaign_verdict": None,
        "exception_message": str(error),
        "exception_type": type(error).__name__,
        "execution_commit": execution_commit,
        "execution_worktree": str(ROOT),
        "failure_stage": failure_stage,
        "gate_evaluated": False,
        "physical_audio_samples_read": 0,
        "recorded_at": _now(),
        "scientific_runs_launched": 0,
        "test_waveform_samples_read": 0,
        "validation_status": validation_status,
    }
    write_new_json(path, record)


def main() -> int:
    execution_commit = _assert_clean_execution_snapshot()
    decisions = gate_decisions(GATE_LEDGER)
    validate_stage_authorization("preflight", decisions)
    if decisions or SUMMARY.exists():
        raise RuntimeError("quality-teacher preflight is already frozen")
    _assert_no_scientific_runs()
    validation_status = {"data_audit": "not_run", "test": "not_run", "lint": "not_run"}
    failure_stage = "repository_validation"
    try:
        protocol = validate_repository_state(ROOT)
        data_contract = load_data_contract(ROOT)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        if properties.total_memory < MINIMUM_GPU_MEMORY_BYTES:
            raise RuntimeError("CUDA device has less than the frozen 24 GB minimum")
        try:
            authorize_test_access({})
        except SealedTestAccessError:
            test_access_fail_closed = True
        else:
            raise RuntimeError("test access did not fail closed before locks")
        model_contract = _model_contract()

        failure_stage = "make_data_audit"
        _run_validation("data-audit")
        validation_status["data_audit"] = "passed"
        failure_stage = "make_test"
        _run_validation("test")
        validation_status["test"] = "passed"
        failure_stage = "make_lint"
        _run_validation("lint")
        validation_status["lint"] = "passed"

        evidence = {
            "campaign_version": CAMPAIGN_VERSION,
            "clean_execution_snapshot": True,
            "cuda_device_name": properties.name,
            "cuda_total_memory_bytes": properties.total_memory,
            "data_catalog_commit": data_contract["catalog_commit"],
            "execution_commit": execution_commit,
            "external_report_only_accessed": False,
            "model_contract": model_contract,
            "physical_audio_samples_read": 0,
            "protocol_frozen": protocol["status"].startswith("prospective_"),
            "scientific_runs_launched": 0,
            "stage": "preflight",
            "status": "passed",
            "test_access_fail_closed": test_access_fail_closed,
            "test_waveform_samples_read": 0,
            "validation_status": validation_status,
        }
        failure_stage = "gate_finalization"
        write_new_json(SUMMARY, evidence)
        append_gate_event(
            GATE_LEDGER,
            {
                "campaign_version": CAMPAIGN_VERSION,
                "evidence_path": str(SUMMARY.relative_to(ROOT)),
                "stage": "preflight",
                "status": "passed",
            },
        )
        maturity_path = CAMPAIGN_DIR / "MATURITY.json"
        maturity = json.loads(maturity_path.read_text(encoding="utf-8"))
        maturity.update(
            {
                "current_stage": "data_audit",
                "gates_evaluated": 1,
                "status": "active_preflight_passed",
            }
        )
        replace_json(maturity_path, maturity)
        _write_state_after_pass()
        print(json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        _record_invalid(
            error,
            execution_commit=execution_commit,
            failure_stage=failure_stage,
            validation_status=validation_status,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
