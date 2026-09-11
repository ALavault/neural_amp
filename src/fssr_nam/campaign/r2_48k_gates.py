"""Pure fail-closed gates for the scoped 48 kHz R2 campaign."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from fssr_nam.campaign.r2 import R2ConfigError
from fssr_nam.campaign.r2_48k import (
    CANDIDATE_FAMILIES,
    DEVELOPMENT_DEVICES,
    EVALUATION_CONDITIONS,
    PRIMARY_DEVICES,
    PROSPECTIVE_CONDITIONS,
)
from fssr_nam.campaign.r2_gates import (
    R2GateEvidenceError,
)
from fssr_nam.campaign.r2_gates import (
    evaluate_mechanism_gate as evaluate_r2_mechanism_gate,
)
from fssr_nam.campaign.r2_gates import (
    validate_benchmark_evidence as validate_r2_benchmark_evidence,
)

MUSHRA_CONDITIONS = (
    "hidden_hardware_reference",
    "candidate",
    "a2",
    "aa_off_ablation",
    "lowpass_anchor_3p5khz",
    "aliasing_anchor",
)


class R248KGateEvidenceError(ValueError):
    """Raised for malformed, incomplete, or scope-contaminated evidence."""


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R248KGateEvidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise R248KGateEvidenceError(f"{label} must be finite")
    return result


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R248KGateEvidenceError(f"{label} must be a mapping")
    return value


def evaluate_mechanism_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Qualify AA on synthetic references without implying a hardware ASR claim."""
    for row in rows:
        if row.get("reference_kind") != "synthetic_192khz":
            raise R248KGateEvidenceError(
                "R2-48K mechanism rows require a synthetic_192khz reference"
            )
        if row.get("physical_hardware_reference_used") is not False:
            raise R248KGateEvidenceError(
                "R2-48K mechanism must not claim a physical 192 kHz reference"
            )
    try:
        base = evaluate_r2_mechanism_gate(rows)
    except (R2GateEvidenceError, R2ConfigError) as error:
        raise R248KGateEvidenceError(str(error)) from error
    x2_passed = bool(base["x2"]["passed"])
    return {
        "format": "fssr-r2-48k-mechanism-gate-v1",
        "valid": True,
        "campaign_continue": x2_passed,
        "terminal_if_stopped": None if x2_passed else "NO-GO-R2-48K",
        "x2": base["x2"],
        "adaa": base["adaa"],
        "adaa_route": base["adaa_route"],
        "same_weights_verified": True,
        "reference_kind": "synthetic_192khz",
        "supports_synthetic_aliasing_mechanism_claim": True,
        "supports_physical_hardware_aliasing_claim": False,
        "sealed_test_used": False,
    }


def _validate_screen_row(row: Mapping[str, Any]) -> None:
    if row.get("status") != "complete":
        raise R248KGateEvidenceError("screen trajectory status must be complete")
    if row.get("seed") != 0:
        raise R248KGateEvidenceError("screen trajectories must use seed 0")
    if row.get("selection_split") != "validation":
        raise R248KGateEvidenceError("screen selection must use validation only")
    if row.get("sealed_test_opened") is not False:
        raise R248KGateEvidenceError("screen evidence opened a sealed test")
    if row.get("physical_asr_used_for_selection") is not False:
        raise R248KGateEvidenceError("physical ASR cannot be a 48 kHz selection metric")
    optimizer_steps = row.get("optimizer_steps")
    if optimizer_steps not in {5000, 15000}:
        raise R248KGateEvidenceError("screen updates must be 5000 or 15000")
    checkpoints = row.get("checkpoint_steps")
    expected = [200, 1000, 5000] + ([15000] if optimizer_steps == 15000 else [])
    if checkpoints != expected:
        raise R248KGateEvidenceError("screen checkpoint set violates the stop rule")
    snapshots = _mapping(row.get("snapshots"), "screen snapshots")
    if {int(step) for step in snapshots} != set(expected):
        raise R248KGateEvidenceError("screen snapshots do not match checkpoints")
    for step in expected:
        metrics = _mapping(
            snapshots.get(str(step), snapshots.get(step)), f"screen snapshot {step}"
        )
        if _finite(metrics.get("esr"), f"snapshot {step} ESR") <= 0.0:
            raise R248KGateEvidenceError("screen ESR must be positive")
        if metrics.get("output_guard_passed") is not True:
            raise R248KGateEvidenceError("screen finite/nonconstant guard failed")
    selected = row.get("selected_checkpoint")
    if selected not in expected:
        raise R248KGateEvidenceError("selected screen checkpoint is not common")
    selected_metrics = _mapping(
        snapshots.get(str(selected), snapshots.get(selected)), "selected snapshot"
    )
    validation = _mapping(row.get("selected_validation"), "selected validation")
    if _finite(validation.get("esr"), "selected ESR") != _finite(
        selected_metrics.get("esr"), "snapshot ESR"
    ):
        raise R248KGateEvidenceError("selected ESR differs from its snapshot")


def evaluate_screen_gate(
    rows: Sequence[Mapping[str, Any]], *, adaa_eligible: bool
) -> dict[str, Any]:
    """Rank three families by dev ESR; physical ASR is deliberately absent."""
    indexed: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        _validate_screen_row(row)
        key = (
            str(row.get("device")),
            str(row.get("family")),
            str(row.get("loss")),
            str(row.get("aa_mode")),
        )
        if key in indexed:
            raise R248KGateEvidenceError(f"duplicate screen trajectory: {key}")
        indexed[key] = row

    expected_base = {
        (device, family, loss, "full_island_x2")
        for device in DEVELOPMENT_DEVICES
        for family in CANDIDATE_FAMILIES
        for loss in ("m4", "wright")
    } | {
        (device, "a2", loss, "off")
        for device in DEVELOPMENT_DEVICES
        for loss in ("m4", "wright")
    }
    if not expected_base <= set(indexed):
        raise R248KGateEvidenceError(
            f"initial screen matrix missing={sorted(expected_base - set(indexed))}"
        )
    loss_medians = {}
    for loss in ("m4", "wright"):
        values = []
        for device in DEVELOPMENT_DEVICES:
            for family in CANDIDATE_FAMILIES:
                snapshots = indexed[(device, family, loss, "full_island_x2")][
                    "snapshots"
                ]
                at_5000 = _mapping(
                    snapshots.get("5000", snapshots.get(5000)), "5000 snapshot"
                )
                values.append(_finite(at_5000.get("esr"), "5000 ESR"))
        loss_medians[loss] = float(np.median(values))
    promoted_loss = min(
        ("m4", "wright"),
        key=lambda loss: (loss_medians[loss], 0 if loss == "m4" else 1),
    )
    expected = set(expected_base)
    modes = ["full_island_x2"]
    if adaa_eligible:
        modes.append("adaa1")
        expected |= {
            (device, family, promoted_loss, "adaa1")
            for device in DEVELOPMENT_DEVICES
            for family in CANDIDATE_FAMILIES
        }
    if set(indexed) != expected:
        raise R248KGateEvidenceError(
            f"screen matrix mismatch; missing={sorted(expected - set(indexed))}, "
            f"extra={sorted(set(indexed) - expected)}"
        )
    for (device, family, loss, mode), row in indexed.items():
        expected_steps = 15_000 if loss == promoted_loss else 5_000
        if mode == "adaa1" and (not adaa_eligible or loss != promoted_loss):
            raise R248KGateEvidenceError("ADAA ran without mechanism eligibility")
        if row.get("optimizer_steps") != expected_steps:
            raise R248KGateEvidenceError(
                f"{device}/{family}/{loss}/{mode} violates the loss stop rule"
            )

    baselines = {}
    for device in DEVELOPMENT_DEVICES:
        options = [indexed[(device, "a2", loss, "off")] for loss in ("m4", "wright")]
        selected = min(
            options,
            key=lambda row: _finite(row["selected_validation"].get("esr"), "A2 ESR"),
        )
        baselines[device] = {
            "loss": selected["loss"],
            "esr": _finite(selected["selected_validation"].get("esr"), "A2 ESR"),
        }

    candidates = []
    for family in CANDIDATE_FAMILIES:
        for mode in modes:
            device_checks = {}
            improvements = []
            cpu_ratios = []
            for device in DEVELOPMENT_DEVICES:
                row = indexed[(device, family, promoted_loss, mode)]
                candidate_esr = _finite(
                    row["selected_validation"].get("esr"), "candidate ESR"
                )
                baseline_esr = baselines[device]["esr"]
                improvement = (baseline_esr - candidate_esr) / baseline_esr
                cpu = _finite(row.get("projected_cpu_ratio_a2"), "projected CPU")
                if cpu <= 0.0:
                    raise R248KGateEvidenceError("projected CPU ratio must be positive")
                improvements.append(improvement)
                cpu_ratios.append(cpu)
                device_checks[device] = {
                    "esr_relative_improvement": improvement,
                    "deployable_fidelity_passed": improvement >= 0.15,
                }
            candidates.append(
                {
                    "family": family,
                    "aa_mode": mode,
                    "median_esr_relative_improvement": float(np.median(improvements)),
                    "projected_cpu_ratio_a2": float(np.median(cpu_ratios)),
                    "deployable_fidelity_passed": all(
                        check["deployable_fidelity_passed"]
                        for check in device_checks.values()
                    ),
                    "devices": device_checks,
                }
            )
    selected_candidate = min(
        candidates,
        key=lambda candidate: (
            -candidate["median_esr_relative_improvement"],
            candidate["projected_cpu_ratio_a2"],
            candidate["family"],
            candidate["aa_mode"],
        ),
    )
    return {
        "format": "fssr-r2-48k-screen-gate-v1",
        "valid": True,
        "passed": True,
        "promoted_loss": promoted_loss,
        "loss_median_validation_esr_at_5000": loss_medians,
        "best_a2_by_device": baselines,
        "candidates": candidates,
        "selected_candidate": selected_candidate,
        "deployable_fidelity_passed": selected_candidate["deployable_fidelity_passed"],
        "physical_asr_used_for_selection": False,
        "sealed_test_used": False,
    }


def evaluate_teacher_gate(
    rows: Sequence[Mapping[str, Any]], baselines: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Gate the x4 teacher on dev ESR while keeping its role model-side only."""
    if len(rows) != 2:
        raise R248KGateEvidenceError("teacher gate requires exactly two rows")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        device = row.get("device")
        if device not in DEVELOPMENT_DEVICES or device in indexed:
            raise R248KGateEvidenceError("teacher device matrix is invalid")
        literals = {
            "seed": 0,
            "aa_mode": "teacher_x4",
            "selection_split": "validation",
            "sealed_test_opened": False,
            "physical_target_sample_rate_hz": 48_000,
            "internal_sample_rate_hz": 192_000,
            "teacher_role": "model_side_regularizer_not_hardware_reference",
            "physical_asr_used_for_selection": False,
        }
        for name, expected in literals.items():
            if row.get(name) != expected:
                raise R248KGateEvidenceError(f"teacher {name} must equal {expected!r}")
        indexed[str(device)] = row
    checks = {}
    for device in DEVELOPMENT_DEVICES:
        baseline = _mapping(baselines.get(device), f"{device} baseline")
        validation = _mapping(indexed[device].get("validation"), "teacher validation")
        baseline_esr = _finite(baseline.get("esr"), "baseline ESR")
        improvement = (
            baseline_esr - _finite(validation.get("esr"), "teacher ESR")
        ) / baseline_esr
        checks[device] = {
            "esr_relative_improvement": improvement,
            "passed": improvement >= 0.15,
        }
    passed = all(check["passed"] for check in checks.values())
    return {
        "format": "fssr-r2-48k-teacher-gate-v1",
        "passed": passed,
        "checks": checks,
        "terminal_if_failed": None if passed else "NO-GO-R2-48K",
        "teacher_role": "model_side_regularizer_not_hardware_reference",
        "supports_physical_hardware_aliasing_claim": False,
        "sealed_test_used": False,
    }


def validate_benchmark_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse the identical native benchmark contract under the new claim scope."""
    try:
        result = validate_r2_benchmark_evidence(evidence)
    except (R2GateEvidenceError, R2ConfigError) as error:
        raise R248KGateEvidenceError(str(error)) from error
    return {
        **result,
        "format": "fssr-r2-48k-benchmark-gate-v1",
        "physical_asr_evaluated": False,
    }


def validate_mushra_design(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Validate eight trials while limiting the primary analysis to heldouts."""
    literal = {
        "standard": "ITU-R BS.1534-3",
        "recruited_participants": 24,
        "double_blind": True,
        "randomized_order": True,
        "one_global_gain_per_trial": True,
        "hidden_hardware_reference": True,
        "primary_devices": list(PRIMARY_DEVICES),
        "development_secondary_devices": list(DEVELOPMENT_DEVICES),
    }
    for name, expected in literal.items():
        if evidence.get(name) != expected:
            raise R248KGateEvidenceError(
                f"MUSHRA design {name} must equal {expected!r}"
            )
    if evidence.get("conditions") != list(MUSHRA_CONDITIONS):
        raise R248KGateEvidenceError("MUSHRA condition set or order changed")
    excerpts = evidence.get("excerpts")
    if not isinstance(excerpts, list) or len(excerpts) != 8:
        raise R248KGateEvidenceError("MUSHRA design requires eight excerpts")
    observed_ids: set[str] = set()
    counts: dict[str, int] = defaultdict(int)
    for excerpt in excerpts:
        item = _mapping(excerpt, "MUSHRA excerpt")
        excerpt_id = item.get("excerpt")
        device = item.get("device")
        if (
            not isinstance(excerpt_id, str)
            or not excerpt_id
            or excerpt_id in observed_ids
        ):
            raise R248KGateEvidenceError("MUSHRA excerpt IDs must be unique")
        if device not in {*DEVELOPMENT_DEVICES, *PRIMARY_DEVICES}:
            raise R248KGateEvidenceError("MUSHRA excerpt device is invalid")
        if item.get("duration_seconds") != 10:
            raise R248KGateEvidenceError("MUSHRA excerpts must be ten seconds")
        if device in PRIMARY_DEVICES:
            expected_role = "prospective_primary"
            expected_sealed = True
        else:
            expected_role = "development_secondary"
            expected_sealed = False
        if (
            item.get("evidence_role") != expected_role
            or item.get("sealed") is not expected_sealed
        ):
            raise R248KGateEvidenceError("MUSHRA excerpt evidence role is invalid")
        observed_ids.add(excerpt_id)
        counts[str(device)] += 1
    if counts != {device: 2 for device in (*DEVELOPMENT_DEVICES, *PRIMARY_DEVICES)}:
        raise R248KGateEvidenceError("MUSHRA requires two excerpts per device")
    if (
        evidence.get("preregistered_exclusion_rule")
        != "configs/r2_48k/mushra_exclusion.yaml"
    ):
        raise R248KGateEvidenceError("MUSHRA exclusion rule changed")
    return {
        "passed": True,
        "standard": literal["standard"],
        "excerpts": 8,
        "primary_excerpts": 4,
        "conditions": list(MUSHRA_CONDITIONS),
    }


def _validate_confirmation_summary(summary: Mapping[str, Any]) -> None:
    literals = {
        "method": "paired-hierarchical-r2-48k-confirmation-v1",
        "replicates": 10_000,
        "seed": 20_260_828,
        "hierarchy": ["seed", "source"],
        "primary_devices": list(PRIMARY_DEVICES),
        "development_devices": list(DEVELOPMENT_DEVICES),
        "development_excluded_from_primary_interval": True,
        "windows_resampled": False,
        "probes_resampled": False,
        "evaluation_conditions": EVALUATION_CONDITIONS,
        "prospective_conditions": PROSPECTIVE_CONDITIONS,
        "physical_asr_in_primary_decision": False,
    }
    for name, expected in literals.items():
        if summary.get(name) != expected:
            raise R248KGateEvidenceError(f"confirmation {name} must equal {expected!r}")


def _validate_devices_won(confirmation: Mapping[str, Any]) -> set[str]:
    won = confirmation.get("devices_won")
    allowed = {*DEVELOPMENT_DEVICES, *PRIMARY_DEVICES}
    if not isinstance(won, list) or len(set(won)) != len(won):
        raise R248KGateEvidenceError("devices_won must be a unique list")
    if not set(won) <= allowed:
        raise R248KGateEvidenceError("devices_won contains an unknown device")
    return set(won)


def _mechanism_x2_passed(mechanism: Mapping[str, Any]) -> bool:
    if mechanism.get("format") != "fssr-r2-48k-mechanism-gate-v1":
        raise R248KGateEvidenceError("synthetic mechanism summary format changed")
    if mechanism.get("reference_kind") != "synthetic_192khz":
        raise R248KGateEvidenceError("mechanism reference scope changed")
    if mechanism.get("supports_physical_hardware_aliasing_claim") is not False:
        raise R248KGateEvidenceError("mechanism overstates hardware evidence")
    x2 = _mapping(mechanism.get("x2"), "mechanism x2")
    return x2.get("passed") is True


def evaluate_prelisten_gate(
    confirmation: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    mechanism: Mapping[str, Any],
) -> dict[str, Any]:
    """Launch listening only after heldout ESR, mechanism, and native cost pass."""
    _validate_confirmation_summary(confirmation)
    benchmark_result = validate_benchmark_evidence(benchmark)
    interval = _mapping(
        confirmation.get("heldout_esr_confidence_interval_95"),
        "heldout ESR interval",
    )
    won = _validate_devices_won(confirmation)
    checks = {
        "heldout_esr": _finite(interval.get("lower"), "heldout ESR lower bound")
        >= 0.15,
        "devices_won": len(won) >= 3,
        "mandatory_devices_won": set(PRIMARY_DEVICES) <= won,
        "synthetic_mechanism": _mechanism_x2_passed(mechanism),
        "cpu": benchmark_result["checks"]["cpu_ratio"],
        "latency": benchmark_result["checks"]["latency"],
        "python_cpp_parity": benchmark_result["checks"]["python_cpp_parity"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "benchmark": benchmark_result,
        "physical_asr_gate_present": False,
    }


def _validate_mushra_summary(summary: Mapping[str, Any]) -> tuple[float, float]:
    literals = {
        "method": "paired-hierarchical-r2-48k-mushra-v1",
        "recruited_participants": 24,
        "excerpts_per_participant": 8,
        "primary_excerpts_per_participant": 4,
        "replicates": 10_000,
        "seed": 20_260_829,
        "hierarchy": ["participant", "primary_excerpt"],
        "primary_devices": list(PRIMARY_DEVICES),
        "development_excluded_from_primary_interval": True,
        "design_validated": True,
    }
    for name, expected in literals.items():
        if summary.get(name) != expected:
            raise R248KGateEvidenceError(f"MUSHRA {name} must equal {expected!r}")
    retained = summary.get("retained_participants")
    if isinstance(retained, bool) or not isinstance(retained, int) or retained < 20:
        raise R248KGateEvidenceError("MUSHRA requires at least 20 retained listeners")
    advantage = _finite(
        summary.get("primary_candidate_minus_a2_points"), "primary MUSHRA advantage"
    )
    interval = _mapping(
        summary.get("primary_confidence_interval_95"), "primary MUSHRA interval"
    )
    lower = _finite(interval.get("lower"), "primary MUSHRA lower bound")
    return advantage, lower


def evaluate_final_verdict(
    confirmation: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    mechanism: Mapping[str, Any],
    mushra: Mapping[str, Any],
    *,
    instrumentation_valid: bool,
) -> dict[str, Any]:
    """Return GO only for the narrower, explicitly scoped 48 kHz claim."""
    if not instrumentation_valid:
        return {
            "verdict": "INVALID",
            "valid": False,
            "reason": "instrumentation_or_protocol_failure",
        }
    prelisten = evaluate_prelisten_gate(confirmation, benchmark, mechanism)
    advantage, mushra_lower = _validate_mushra_summary(mushra)
    interval = _mapping(
        confirmation.get("heldout_esr_confidence_interval_95"),
        "heldout ESR interval",
    )
    won = _validate_devices_won(confirmation)
    checks = {
        "heldout_esr_lower_bound": _finite(
            interval.get("lower"), "heldout ESR lower bound"
        )
        >= 0.15,
        "devices_won": len(won) >= 3,
        "mandatory_devices_won": set(PRIMARY_DEVICES) <= won,
        "synthetic_mechanism": prelisten["checks"]["synthetic_mechanism"],
        "cpu_ratio": prelisten["checks"]["cpu"],
        "latency": prelisten["checks"]["latency"],
        "python_cpp_parity": prelisten["checks"]["python_cpp_parity"],
        "primary_mushra_advantage": advantage > 10.0,
        "primary_mushra_lower_bound": mushra_lower > 0.0,
    }
    passed = all(checks.values())
    return {
        "verdict": "GO-R2-48K" if passed else "NO-GO-R2-48K",
        "valid": True,
        "passed": passed,
        "checks": checks,
        "prelisten": prelisten,
        "claim_scope": {
            "physical": "48khz_fidelity_efficiency_and_listening",
            "aliasing": "synthetic_mechanism_only",
            "global_state_of_the_art": False,
        },
        "physical_asr_used_for_decision": False,
    }
