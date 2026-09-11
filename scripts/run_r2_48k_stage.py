#!/usr/bin/env python3
"""Validate one evidence-backed R2-48K stage without fabricating measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from fssr_nam.campaign.r2_48k import (
    CAMPAIGN_VERSION,
    CANDIDATE_FAMILIES,
    DEVELOPMENT_DEVICES,
    PRIMARY_DEVICES,
    evaluation_condition_keys,
    open_internal_validation_tests,
    unlock_internal_validation,
    validate_sealed_test_boundary,
    validate_stage_authorization,
)
from fssr_nam.campaign.r2_48k_gates import (
    R248KGateEvidenceError,
    evaluate_final_verdict,
    evaluate_mechanism_gate,
    evaluate_prelisten_gate,
    evaluate_screen_gate,
    evaluate_teacher_gate,
    validate_benchmark_evidence,
    validate_mushra_design,
)
from fssr_nam.campaign.r2_48k_registry import append_gate_event, gate_decisions
from fssr_nam.statistics.r2_48k import (
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


class PendingEvidence(RuntimeError):
    """Raised when a stage is authorized but its real evidence is not present."""


def _load_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise PendingEvidence(f"missing {label}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise R248KGateEvidenceError(f"invalid {label}: {error}") from error


def _event(root: Path, stage: str, status: str, evidence_path: str) -> None:
    append_gate_event(
        root / ".codex_campaign/r2_48k/GATE_LEDGER.jsonl",
        {
            "campaign_version": CAMPAIGN_VERSION,
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
        if _load_json(path, "immutable R2-48K evidence") == payload:
            return
        raise R248KGateEvidenceError(
            f"refusing to overwrite immutable evidence: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write("\n")


def _confirm_validation(evidence: Any) -> dict[str, Any]:
    if not isinstance(evidence, dict) or not isinstance(
        evidence.get("conditions"), list
    ):
        raise R248KGateEvidenceError("confirmation validation needs conditions")
    expected = set(evaluation_condition_keys())
    observed = set()
    for condition in evidence["conditions"]:
        if not isinstance(condition, dict):
            raise R248KGateEvidenceError("confirmation conditions must be objects")
        key = (condition.get("device"), condition.get("family"), condition.get("seed"))
        if key in observed:
            raise R248KGateEvidenceError(f"duplicate confirmation condition: {key}")
        observed.add(key)
        device = key[0]
        expected_role = (
            "historical_development"
            if device in DEVELOPMENT_DEVICES
            else "prospective_primary"
        )
        if condition.get("evidence_role") != expected_role:
            raise R248KGateEvidenceError(f"invalid evidence role for {key}")
        if condition.get("training_status") != "complete":
            raise R248KGateEvidenceError(f"incomplete confirmation condition: {key}")
        if condition.get("selection_split") != "validation":
            raise R248KGateEvidenceError("confirmation selection must use validation")
        if condition.get("internal_validation_test_opened") is not False:
            raise R248KGateEvidenceError("confirmation opened a primary test")
        if condition.get("physical_asr_used_for_selection") is not False:
            raise R248KGateEvidenceError("confirmation used forbidden physical ASR")
        if device in PRIMARY_DEVICES and condition.get("tuning_performed") is not False:
            raise R248KGateEvidenceError("tuning on Blackstar or UA1176 is forbidden")
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise R248KGateEvidenceError(
            f"confirmation matrix mismatch; missing={missing}, extra={extra}"
        )
    return {
        "status": "passed",
        "conditions": len(observed),
        "prospective_conditions": 20,
        "internal_validation_test_used": False,
    }


def _validate_lock(lock: Any) -> dict[str, Any]:
    if not isinstance(lock, dict):
        raise R248KGateEvidenceError("confirmatory lock must be an object")
    literals = {
        "campaign_version": CAMPAIGN_VERSION,
        "status": "frozen",
        "external_report_only_locked": True,
        "internal_validation_tests_opened": False,
        "development_tests_previously_observed": True,
        "physical_asr_is_a_decision_metric": False,
    }
    for name, expected in literals.items():
        if lock.get(name) != expected:
            raise R248KGateEvidenceError(f"lock {name} must equal {expected!r}")
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
        raise R248KGateEvidenceError("lock does not freeze every required component")
    if any(not isinstance(value, str) or not value for value in frozen.values()):
        raise R248KGateEvidenceError("frozen references must be non-empty")
    candidate = lock.get("candidate")
    if not isinstance(candidate, dict) or candidate.get("family") not in set(
        CANDIDATE_FAMILIES
    ):
        raise R248KGateEvidenceError("lock candidate family is invalid")
    if candidate.get("aa_mode") not in {
        "full_island_x2",
        "adaa1",
        "distilled_x2",
        "distilled_adaa1",
    }:
        raise R248KGateEvidenceError("lock candidate AA mode is invalid")
    return {"status": "frozen", "candidate": candidate}


def _record_test_open(root: Path) -> None:
    path = root / ".codex_campaign/r2_48k/EXTERNAL_FREEZE.json"
    freeze = _load_json(path, "R2-48K external freeze")
    freeze = open_internal_validation_tests(freeze)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(freeze, stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write("\n")
    temporary.replace(path)


def _record_validation_unlock(root: Path) -> None:
    path = root / ".codex_campaign/r2_48k/EXTERNAL_FREEZE.json"
    freeze = _load_json(path, "R2-48K external freeze")
    freeze = unlock_internal_validation(freeze)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(freeze, stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write("\n")
    temporary.replace(path)


def _mechanism_summary(summaries: Path) -> dict[str, Any]:
    value = _load_json(summaries / "mechanism_gate.json", "mechanism gate")
    if not isinstance(value, dict):
        raise R248KGateEvidenceError("mechanism gate must be an object")
    return value


def run_stage(root: Path, stage: str) -> dict[str, Any]:
    summaries = root / "experiments/summaries/r2_48k"
    ledger = root / ".codex_campaign/r2_48k/GATE_LEDGER.jsonl"
    decisions = gate_decisions(ledger)
    if stage == "mechanism":
        validate_stage_authorization("mechanism", decisions)
        path = summaries / "mechanism.json"
        evidence = _load_json(path, "R2-48K mechanism evidence")
        result = evaluate_mechanism_gate(evidence.get("rows", []))
        _write_new_json(summaries / "mechanism_gate.json", result)
        _event(
            root,
            "mechanism_x2",
            "passed" if result["campaign_continue"] else "failed",
            _relative(root, path),
        )
        if not result["campaign_continue"]:
            result["verdict"] = "NO-GO-R2-48K"
        return result
    if stage == "screen":
        validate_stage_authorization("screen", decisions)
        mechanism = _mechanism_summary(summaries)
        adaa_eligible = mechanism.get("adaa_route") == "eligible"
        path = summaries / "screen.json"
        evidence = _load_json(path, "R2-48K screen evidence")
        result = evaluate_screen_gate(
            evidence.get("rows", []), adaa_eligible=adaa_eligible
        )
        _write_new_json(summaries / "screen_gate.json", result)
        _event(root, "screen", "promoted", _relative(root, path))
        deployable_status = (
            "passed" if result["deployable_fidelity_passed"] else "failed_fidelity_gate"
        )
        _event(root, "deployable", deployable_status, _relative(root, path))
        return result
    if stage == "teacher":
        validate_stage_authorization("teacher", decisions)
        screen = _load_json(summaries / "screen_gate.json", "screen gate")
        path = summaries / "teacher.json"
        evidence = _load_json(path, "R2-48K teacher evidence")
        result = evaluate_teacher_gate(
            evidence.get("rows", []), screen["best_a2_by_device"]
        )
        _write_new_json(summaries / "teacher_gate.json", result)
        _event(
            root,
            "teacher",
            "passed" if result["passed"] else "failed",
            _relative(root, path),
        )
        if not result["passed"]:
            result["verdict"] = "NO-GO-R2-48K"
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
        evidence = _load_json(path, "R2-48K distillation evidence")
        literals = {
            "teacher_passed": True,
            "fidelity_gate_passed": True,
            "physical_asr_used_for_selection": False,
            "teacher_role": "model_side_regularizer_not_hardware_reference",
            "teacher_esr_preemphasis_weight": 0.25,
        }
        for name, expected in literals.items():
            if evidence.get(name) != expected:
                raise R248KGateEvidenceError(
                    f"distillation {name} must equal {expected!r}"
                )
        _event(root, "distill", "passed", _relative(root, path))
        _event(root, "final_candidate", "selected", _relative(root, path))
        return {"status": "passed", "final_candidate": "distilled"}
    if stage == "lock":
        validate_stage_authorization("lock", decisions)
        path = root / ".codex_campaign/r2_48k/CONFIRMATORY_LOCK.yaml"
        if not path.is_file():
            raise PendingEvidence(f"missing R2-48K lock: {path}")
        result = _validate_lock(yaml.safe_load(path.read_text(encoding="utf-8")))
        _record_validation_unlock(root)
        _event(root, "lock", "frozen", _relative(root, path))
        return result
    if stage == "confirm":
        validate_stage_authorization("confirm", decisions)
        freeze = _load_json(
            root / ".codex_campaign/r2_48k/EXTERNAL_FREEZE.json",
            "external freeze",
        )
        if freeze.get("internal_validation_train_validation_locked") is not False:
            raise R248KGateEvidenceError(
                "confirmatory lock has not released train/validation outputs"
            )
        if freeze.get("internal_validation_tests_locked") is not True:
            raise R248KGateEvidenceError("confirmation must keep primary tests locked")
        path = summaries / "confirm_validation.json"
        result = _confirm_validation(_load_json(path, "confirmation validation"))
        _event(root, "confirm_validation", "passed", _relative(root, path))
        return result
    if stage == "benchmark":
        validate_stage_authorization("benchmark", decisions)
        benchmark_path = summaries / "benchmark.json"
        parity_path = summaries / "parity.json"
        parity = _load_json(parity_path, "R2-48K parity report")
        if (
            parity.get("format") != "fssr-r2-parity-v1"
            or parity.get("passed") is not True
            or parity.get("scientific_campaign_version") != CAMPAIGN_VERSION
        ):
            raise R248KGateEvidenceError("R2-48K Python/C++ parity did not pass")
        result = validate_benchmark_evidence(
            _load_json(benchmark_path, "R2-48K benchmark evidence")
        )
        _event(root, "python_cpp_parity", "passed", _relative(root, parity_path))
        _event(
            root,
            "benchmark",
            "passed" if result["passed"] else "failed",
            _relative(root, benchmark_path),
        )
        if not result["passed"]:
            result["verdict"] = "NO-GO-R2-48K"
        return result
    if stage == "listen":
        decisions = gate_decisions(ledger)
        rows_path = summaries / "evaluation_rows.json"
        confirmation_path = summaries / "confirmation.json"
        if decisions.get("listening") == "complete":
            return {
                "status": "complete",
                "confirmation": _load_json(confirmation_path, "confirmation summary"),
                "mushra": _load_json(summaries / "mushra.json", "MUSHRA summary"),
            }
        if decisions.get("sealed_test") == "opened_once":
            confirmation = _load_json(confirmation_path, "confirmation summary")
        else:
            freeze = _load_json(
                root / ".codex_campaign/r2_48k/EXTERNAL_FREEZE.json",
                "external freeze",
            )
            validate_sealed_test_boundary(
                decisions=decisions,
                internal_validation_tests_open_count=freeze.get(
                    "internal_validation_test_open_count", -1
                ),
                external_report_only_locked=freeze.get("external_report_only_locked")
                is True,
                development_tests_previously_observed=freeze.get(
                    "development_tests_previously_observed"
                )
                is True,
            )
            rows = _load_json(rows_path, "R2-48K paired evaluation rows")
            if not isinstance(rows, list):
                raise R248KGateEvidenceError("evaluation rows must be a list")
            confirmation = hierarchical_confirmation_bootstrap(rows)
            _write_new_json(confirmation_path, confirmation)
            _record_test_open(root)
            _event(
                root, "sealed_test", "opened_once", _relative(root, confirmation_path)
            )
        interval = confirmation.get("heldout_esr_confidence_interval_95", {})
        won = confirmation.get("devices_won", [])
        esr_passed = (
            isinstance(interval, dict)
            and interval.get("lower", -1.0) >= 0.15
            and isinstance(won, list)
            and len(won) >= 3
            and set(PRIMARY_DEVICES) <= set(won)
        )
        _event(
            root,
            "sealed_esr",
            "passed" if esr_passed else "failed",
            _relative(root, rows_path),
        )
        benchmark_path = summaries / "benchmark.json"
        benchmark = _load_json(benchmark_path, "R2-48K benchmark evidence")
        mechanism = _mechanism_summary(summaries)
        prelisten = evaluate_prelisten_gate(confirmation, benchmark, mechanism)
        cpu_status = (
            "passed"
            if prelisten["checks"]["cpu"] and prelisten["checks"]["latency"]
            else "failed"
        )
        _event(root, "cpu", cpu_status, _relative(root, benchmark_path))
        if not prelisten["passed"]:
            return {
                "status": "not_launched",
                "verdict": "NO-GO-R2-48K",
                "prelisten": prelisten,
            }
        validate_stage_authorization("listen", gate_decisions(ledger))
        design_path = summaries / "mushra_protocol.json"
        responses_path = summaries / "mushra_responses.json"
        design = validate_mushra_design(
            _load_json(design_path, "R2-48K MUSHRA protocol")
        )
        responses = _load_json(responses_path, "R2-48K MUSHRA responses")
        if not isinstance(responses, list):
            raise R248KGateEvidenceError("MUSHRA responses must be a list")
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
        mechanism = _mechanism_summary(summaries)
        prelisten = evaluate_prelisten_gate(confirmation, benchmark, mechanism)
        if not prelisten["passed"]:
            return {
                "verdict": "NO-GO-R2-48K",
                "valid": True,
                "prelisten": prelisten,
            }
        validate_stage_authorization("audit", decisions)
        mushra = _load_json(summaries / "mushra.json", "MUSHRA summary")
        result = evaluate_final_verdict(
            confirmation,
            benchmark,
            mechanism,
            mushra,
            instrumentation_valid=True,
        )
        _event(root, "audit", result["verdict"], "derived:final-R2-48K-gates")
        return result
    raise ValueError(f"unsupported R2-48K stage: {stage}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        result = run_stage(root, arguments.stage)
    except PendingEvidence as error:
        print(
            json.dumps(
                {
                    "campaign_version": CAMPAIGN_VERSION,
                    "stage": arguments.stage,
                    "status": "PENDING_EVIDENCE",
                    "reason": str(error),
                    "scientific_result_invented": False,
                },
                sort_keys=True,
            )
        )
        return 2
    except (R248KGateEvidenceError, RuntimeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "campaign_version": CAMPAIGN_VERSION,
                    "stage": arguments.stage,
                    "status": "INVALID",
                    "reason": str(error),
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if result.get("verdict") not in {"NO-GO-R2-48K", "INVALID"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
