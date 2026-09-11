#!/usr/bin/env python3
"""Validate and advance one evidence-backed R2 stage without fabricating work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from fssr_nam.campaign.r2 import (
    final_condition_keys,
    validate_sealed_test_boundary,
    validate_stage_authorization,
)
from fssr_nam.campaign.r2_gates import (
    R2GateEvidenceError,
    evaluate_final_verdict,
    evaluate_mechanism_gate,
    evaluate_prelisten_gate,
    evaluate_screen_gate,
    evaluate_teacher_gate,
    validate_benchmark_evidence,
    validate_mushra_design,
)
from fssr_nam.campaign.r2_registry import append_gate_event, gate_decisions
from fssr_nam.statistics.r2 import (
    hierarchical_confirmation_bootstrap,
    hierarchical_mushra_bootstrap,
)

STAGES = (
    "mechanism",
    "screen",
    "teacher",
    "distill",
    "lock",
    "confirm",
    "benchmark",
    "listen",
    "audit",
)


class PendingExternalEvidence(RuntimeError):
    pass


def _load_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise PendingExternalEvidence(f"missing {label}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise R2GateEvidenceError(f"invalid {label}: {error}") from error


def _event(root: Path, stage: str, status: str, evidence_path: str) -> None:
    append_gate_event(
        root / ".codex_campaign/r2/GATE_LEDGER.jsonl",
        {
            "campaign_version": "FSSR-R2-v1",
            "stage": stage,
            "status": status,
            "evidence_path": evidence_path,
        },
    )


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _write_new_json(path: Path, payload: Any) -> None:
    if path.exists():
        if _load_json(path, "immutable R2 evidence") == payload:
            return
        raise R2GateEvidenceError(f"refusing to overwrite immutable evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write("\n")


def _confirm_validation(evidence: Any) -> dict[str, Any]:
    if not isinstance(evidence, dict) or not isinstance(
        evidence.get("conditions"), list
    ):
        raise R2GateEvidenceError("confirmation validation needs a conditions list")
    expected = set(final_condition_keys())
    observed = set()
    for condition in evidence["conditions"]:
        if not isinstance(condition, dict):
            raise R2GateEvidenceError("confirmation conditions must be objects")
        key = (condition.get("device"), condition.get("family"), condition.get("seed"))
        if key in observed:
            raise R2GateEvidenceError(f"duplicate confirmation condition: {key}")
        observed.add(key)
        if condition.get("training_status") != "complete":
            raise R2GateEvidenceError(f"incomplete confirmation condition: {key}")
        if condition.get("selection_split") != "validation":
            raise R2GateEvidenceError("confirmation selection must use validation")
        if condition.get("sealed_test_opened") is not False:
            raise R2GateEvidenceError("confirmation training opened a sealed test")
        if (
            key[0] in {"blackstar", "ua1176"}
            and condition.get("tuning_performed") is not False
        ):
            raise R2GateEvidenceError("tuning on Blackstar or UA1176 is forbidden")
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise R2GateEvidenceError(
            f"confirmation condition matrix mismatch; missing={missing}, extra={extra}"
        )
    return {"status": "passed", "conditions": len(observed), "sealed_test_used": False}


def _validate_lock(lock: Any) -> dict[str, Any]:
    if not isinstance(lock, dict):
        raise R2GateEvidenceError("R2 confirmatory lock must be an object")
    if lock.get("campaign_version") != "FSSR-R2-v1" or lock.get("status") != "frozen":
        raise R2GateEvidenceError("R2 confirmatory lock is not frozen")
    if lock.get("external_report_only_locked") is not True:
        raise R2GateEvidenceError("EXTERNAL_REPORT_ONLY must remain locked")
    if lock.get("internal_tests_opened") is not False:
        raise R2GateEvidenceError("R2 lock must precede internal test opening")
    frozen = lock.get("frozen")
    required = {
        "architecture",
        "loss",
        "code",
        "data",
        "checkpoints",
        "metrics",
        "export",
    }
    if not isinstance(frozen, dict) or set(frozen) != required:
        raise R2GateEvidenceError("R2 lock does not freeze every required component")
    if any(not isinstance(value, str) or not value for value in frozen.values()):
        raise R2GateEvidenceError("R2 frozen component references must be non-empty")
    candidate = lock.get("candidate")
    if not isinstance(candidate, dict) or candidate.get("family") not in {
        "aa-nam",
        "aa-fssr",
    }:
        raise R2GateEvidenceError("R2 lock candidate family is invalid")
    if candidate.get("aa_mode") not in {
        "full_island_x2",
        "adaa1",
        "distilled_x2",
        "distilled_adaa1",
    }:
        raise R2GateEvidenceError("R2 lock candidate AA mode is invalid")
    return {"status": "frozen", "candidate": candidate}


def _record_test_open(root: Path) -> None:
    path = root / ".codex_campaign/r2/EXTERNAL_FREEZE.json"
    freeze = _load_json(path, "R2 external freeze")
    if freeze.get("internal_tests_open_count") != 0:
        raise R2GateEvidenceError("R2 internal tests were already opened")
    freeze["internal_tests_opened"] = True
    freeze["internal_tests_open_count"] = 1
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(freeze, stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write("\n")
    temporary.replace(path)


def run_stage(root: Path, stage: str) -> dict[str, Any]:
    summaries = root / "experiments/summaries/r2"
    ledger = root / ".codex_campaign/r2/GATE_LEDGER.jsonl"
    decisions = gate_decisions(ledger)
    if stage == "mechanism":
        validate_stage_authorization("mechanism", decisions)
        path = summaries / "mechanism.json"
        evidence = _load_json(path, "R2 mechanism evidence")
        result = evaluate_mechanism_gate(evidence.get("rows", []))
        _event(
            root,
            "mechanism_x2",
            "passed" if result["r2_continue"] else "failed",
            _relative(root, path),
        )
        if not result["r2_continue"]:
            result["verdict"] = "NO-GO-R2"
        return result
    if stage == "screen":
        validate_stage_authorization("screen", decisions)
        path = summaries / "screen.json"
        evidence = _load_json(path, "R2 screen evidence")
        result = evaluate_screen_gate(evidence.get("rows", []))
        _event(
            root,
            "screen",
            "promoted" if result["passed"] else "failed",
            _relative(root, path),
        )
        if result["passed"]:
            _event(root, "deployable", "passed", _relative(root, path))
        else:
            result["verdict"] = "NO-GO-R2"
        return result
    if stage == "teacher":
        validate_stage_authorization("teacher", decisions)
        screen_path = summaries / "screen.json"
        screen = evaluate_screen_gate(
            _load_json(screen_path, "screen evidence").get("rows", [])
        )
        path = summaries / "teacher.json"
        evidence = _load_json(path, "R2 teacher evidence")
        result = evaluate_teacher_gate(
            evidence.get("rows", []), screen["best_a2_by_device"]
        )
        _event(
            root,
            "teacher",
            "passed" if result["passed"] else "failed",
            _relative(root, path),
        )
        if not result["passed"]:
            result["verdict"] = "NO-GO-R2"
        return result
    if stage == "distill":
        if decisions.get("teacher") != "passed":
            validate_stage_authorization("distill", decisions)
        if decisions.get("deployable") == "passed":
            result = {"status": "skipped_not_required", "final_candidate": "deployable"}
            _event(root, "distill", "not_required", "gate:deployable=passed")
            _event(root, "final_candidate", "selected", "gate:deployable=passed")
            return result
        validate_stage_authorization("distill", decisions)
        path = summaries / "distill.json"
        evidence = _load_json(path, "R2 distillation evidence")
        if (
            evidence.get("teacher_passed") is not True
            or evidence.get("double_gate_passed") is not True
        ):
            return {"status": "failed", "verdict": "NO-GO-R2"}
        if evidence.get("teacher_esr_preemphasis_weight") != 0.25:
            raise R2GateEvidenceError("R2 distillation teacher weight must equal 0.25")
        _event(root, "distill", "passed", _relative(root, path))
        _event(root, "final_candidate", "selected", _relative(root, path))
        return {"status": "passed", "final_candidate": "distilled"}
    if stage == "lock":
        validate_stage_authorization("lock", decisions)
        path = root / ".codex_campaign/r2/CONFIRMATORY_LOCK.yaml"
        if not path.is_file():
            raise PendingExternalEvidence(f"missing R2 confirmatory lock: {path}")
        result = _validate_lock(yaml.safe_load(path.read_text(encoding="utf-8")))
        _event(root, "lock", "frozen", _relative(root, path))
        return result
    if stage == "confirm":
        validate_stage_authorization("confirm", decisions)
        path = summaries / "confirm_validation.json"
        result = _confirm_validation(_load_json(path, "R2 confirmation validation"))
        _event(root, "confirm_validation", "passed", _relative(root, path))
        return result
    if stage == "benchmark":
        validate_stage_authorization("benchmark", decisions)
        path = summaries / "benchmark.json"
        parity_path = summaries / "parity.json"
        parity = _load_json(parity_path, "R2 parity report")
        if (
            parity.get("format") != "fssr-r2-parity-v1"
            or parity.get("passed") is not True
        ):
            raise R2GateEvidenceError("R2 Python/C++ parity report did not pass")
        result = validate_benchmark_evidence(_load_json(path, "R2 benchmark evidence"))
        _event(root, "python_cpp_parity", "passed", _relative(root, parity_path))
        _event(
            root,
            "benchmark",
            "passed" if result["passed"] else "failed",
            _relative(root, path),
        )
        if not result["passed"]:
            result["verdict"] = "NO-GO-R2"
        return result
    if stage == "listen":
        decisions = gate_decisions(ledger)
        raw_path = summaries / "sealed_test_rows.json"
        summary_path = summaries / "confirmation.json"
        if decisions.get("listening") == "complete":
            return {
                "status": "complete",
                "confirmation": _load_json(summary_path, "R2 confirmation summary"),
                "mushra": _load_json(summaries / "mushra.json", "MUSHRA summary"),
            }
        if decisions.get("sealed_test") == "opened_once":
            if not summary_path.is_file():
                raise R2GateEvidenceError(
                    "sealed test is recorded open but its immutable summary is missing"
                )
            confirmation = _load_json(summary_path, "R2 confirmation summary")
        else:
            freeze = _load_json(
                root / ".codex_campaign/r2/EXTERNAL_FREEZE.json", "external freeze"
            )
            validate_sealed_test_boundary(
                decisions=decisions,
                internal_tests_open_count=freeze.get("internal_tests_open_count", -1),
                external_report_only_locked=freeze.get("external_report_only_locked")
                is True,
            )
            rows = _load_json(raw_path, "R2 sealed-test source rows")
            if not isinstance(rows, list):
                raise R2GateEvidenceError("sealed-test rows must be a list")
            confirmation = hierarchical_confirmation_bootstrap(rows)
            _write_new_json(summary_path, confirmation)
            _record_test_open(root)
            _event(
                root,
                "sealed_test",
                "opened_once",
                _relative(root, summary_path),
            )
        interval = confirmation.get("esr_confidence_interval_95", {})
        esr_passed = isinstance(interval, dict) and interval.get("lower", -1.0) >= 0.15
        asr_passed = confirmation.get("asr_reduction_db", -1.0) >= 10.0
        _event(
            root,
            "sealed_esr",
            "passed" if esr_passed else "failed",
            _relative(root, raw_path),
        )
        _event(
            root,
            "sealed_asr",
            "passed" if asr_passed else "failed",
            _relative(root, raw_path),
        )
        benchmark_path = summaries / "benchmark.json"
        benchmark = _load_json(benchmark_path, "R2 benchmark evidence")
        prelisten = evaluate_prelisten_gate(confirmation, benchmark)
        cpu_status = (
            "passed"
            if prelisten["checks"]["cpu"] and prelisten["checks"]["latency"]
            else "failed"
        )
        _event(root, "cpu", cpu_status, _relative(root, benchmark_path))
        if not prelisten["passed"]:
            return {
                "status": "not_launched",
                "verdict": "NO-GO-R2",
                "prelisten": prelisten,
            }
        responses_path = summaries / "mushra_responses.json"
        design_path = summaries / "mushra_protocol.json"
        design = validate_mushra_design(_load_json(design_path, "R2 MUSHRA protocol"))
        responses = _load_json(responses_path, "R2 MUSHRA responses")
        if not isinstance(responses, list):
            raise R2GateEvidenceError("MUSHRA responses must be a list")
        mushra = hierarchical_mushra_bootstrap(responses)
        mushra["design_validated"] = design["passed"]
        _write_new_json(summaries / "mushra.json", mushra)
        _event(root, "listening", "complete", _relative(root, responses_path))
        return {"status": "complete", "prelisten": prelisten, "mushra": mushra}
    if stage == "audit":
        confirmation = _load_json(
            summaries / "confirmation.json", "confirmation summary"
        )
        benchmark = _load_json(summaries / "benchmark.json", "benchmark evidence")
        prelisten = evaluate_prelisten_gate(confirmation, benchmark)
        if not prelisten["passed"]:
            return {"verdict": "NO-GO-R2", "valid": True, "prelisten": prelisten}
        validate_stage_authorization("audit", decisions)
        mushra = _load_json(summaries / "mushra.json", "MUSHRA summary")
        result = evaluate_final_verdict(
            confirmation, benchmark, mushra, instrumentation_valid=True
        )
        _event(root, "audit", result["verdict"], "derived:final-R2-gates")
        return result
    raise ValueError(f"unsupported R2 stage: {stage}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        result = run_stage(root, arguments.stage)
    except PendingExternalEvidence as error:
        print(
            json.dumps(
                {
                    "campaign_version": "FSSR-R2-v1",
                    "stage": arguments.stage,
                    "status": "PENDING_EXTERNAL",
                    "reason": str(error),
                    "scientific_result_invented": False,
                },
                sort_keys=True,
            )
        )
        return 2
    except (R2GateEvidenceError, RuntimeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "campaign_version": "FSSR-R2-v1",
                    "stage": arguments.stage,
                    "status": "INVALID",
                    "reason": str(error),
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if result.get("verdict") not in {"NO-GO-R2", "INVALID"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
