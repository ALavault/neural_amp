"""Pure fail-closed mechanism, cost, listening, and verdict gates for R2."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from fssr_nam.campaign.r2 import FINAL_CONDITIONS

FIXTURES = (
    "tanh",
    "asymmetric_clipping",
    "two_clippers",
    "short_memory",
    "slow_sag",
    "rf2047_residual",
)
MECHANISM_MODES = ("off", "full_island_x2", "adaa1", "teacher_x4")
MUSHRA_CONDITIONS = (
    "hidden_hardware_reference",
    "candidate",
    "a2",
    "aa_off_ablation",
    "lowpass_anchor_3p5khz",
    "aliasing_anchor",
)


class R2GateEvidenceError(ValueError):
    """Raised when evidence is missing, malformed, non-finite, or contaminated."""


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R2GateEvidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise R2GateEvidenceError(f"{label} must be finite")
    return result


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R2GateEvidenceError(f"{label} must be a mapping")
    return value


def evaluate_mechanism_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate the exact same-weight six-fixture AA mechanism matrix."""
    expected = {(fixture, mode) for fixture in FIXTURES for mode in MECHANISM_MODES}
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    weights: dict[str, set[str]] = defaultdict(set)
    measured = []
    reference = []
    for row in rows:
        fixture = row.get("fixture")
        mode = row.get("mode")
        key = (str(fixture), str(mode))
        if key in indexed:
            raise R2GateEvidenceError(f"duplicate mechanism condition: {key}")
        indexed[key] = row
        weight_id = row.get("weights_id")
        if not isinstance(weight_id, str) or not weight_id:
            raise R2GateEvidenceError("mechanism weights_id must be non-empty")
        weights[str(fixture)].add(weight_id)
        measured.append(_finite(row.get("asr_db"), f"{key} asr_db"))
        reference.append(
            _finite(row.get("reference_192khz_asr_db"), f"{key} reference ASR")
        )
        if row.get("guard_passed") is not True:
            raise R2GateEvidenceError(f"{key} failed the anti-silence guard")
    if set(indexed) != expected:
        raise R2GateEvidenceError(
            f"mechanism matrix mismatch; missing={sorted(expected - set(indexed))}, "
            f"extra={sorted(set(indexed) - expected)}"
        )
    if any(len(ids) != 1 for ids in weights.values()):
        raise R2GateEvidenceError("AA modes must use identical weights per fixture")
    residual_ratio = _finite(
        indexed[("rf2047_residual", "off")].get("residual_energy_ratio"),
        "RF2047 residual energy ratio",
    )
    if residual_ratio < 0.10:
        raise R2GateEvidenceError("RF2047 residual energy ratio must be at least 10%")
    rho_result = spearmanr(measured, reference)
    rho = float(rho_result.statistic)
    if not math.isfinite(rho):
        raise R2GateEvidenceError("mechanism Spearman rho is undefined")

    def route(mode: str) -> dict[str, Any]:
        gains = []
        fundamental_errors = []
        latencies = []
        for fixture in FIXTURES:
            off = _finite(indexed[(fixture, "off")].get("asr_db"), "off ASR")
            selected = indexed[(fixture, mode)]
            gains.append(off - _finite(selected.get("asr_db"), f"{mode} ASR"))
            fundamental_errors.append(
                _finite(
                    selected.get("fundamental_complex_error"),
                    f"{fixture} {mode} fundamental complex error",
                )
            )
            latencies.append(
                int(_finite(selected.get("latency_samples"), f"{mode} latency"))
            )
        checks = {
            "spearman": rho >= 0.90,
            "median_gain": float(np.median(gains)) >= 10.0,
            "each_fixture_gain": min(gains) >= 6.0,
            "fundamental_complex_error": max(fundamental_errors) <= 1.0e-5,
            "phase_alignment": max(latencies) <= 48,
        }
        return {
            "mode": mode,
            "passed": all(checks.values()),
            "checks": checks,
            "spearman_rho": rho,
            "median_asr_gain_db": float(np.median(gains)),
            "minimum_fixture_asr_gain_db": min(gains),
            "maximum_fundamental_complex_error": max(fundamental_errors),
            "maximum_latency_samples": max(latencies),
        }

    x2 = route("full_island_x2")
    adaa = route("adaa1")
    return {
        "format": "fssr-r2-mechanism-gate-v1",
        "valid": True,
        "r2_continue": x2["passed"],
        "terminal_if_stopped": None if x2["passed"] else "NO-GO-R2",
        "x2": x2,
        "adaa": adaa,
        "adaa_route": "eligible" if adaa["passed"] else "rejected_only",
        "same_weights_verified": True,
        "sealed_test_used": False,
    }


def _validate_screen_row(row: Mapping[str, Any]) -> None:
    if row.get("status") != "complete":
        raise R2GateEvidenceError("screen trajectory status must be complete")
    if row.get("seed") != 0:
        raise R2GateEvidenceError("screen trajectories must use seed 0")
    if row.get("selection_split") != "validation":
        raise R2GateEvidenceError("screen selection must use validation only")
    if row.get("sealed_test_opened") is not False:
        raise R2GateEvidenceError("screen evidence opened a sealed test")
    optimizer_steps = row.get("optimizer_steps")
    if optimizer_steps not in {5000, 15000}:
        raise R2GateEvidenceError("screen optimizer steps must be 5000 or 15000")
    checkpoints = row.get("checkpoint_steps")
    expected = [200, 1000, 5000] + ([15000] if optimizer_steps == 15000 else [])
    if checkpoints != expected:
        raise R2GateEvidenceError("screen checkpoint set does not match stop rule")
    snapshots = _mapping(row.get("snapshots"), "screen snapshots")
    if {int(step) for step in snapshots} != set(expected):
        raise R2GateEvidenceError("screen snapshots do not match checkpoint set")
    for step in expected:
        metrics = _mapping(
            snapshots.get(str(step), snapshots.get(step)), f"screen snapshot {step}"
        )
        esr = _finite(metrics.get("esr"), f"snapshot {step} ESR")
        _finite(metrics.get("asr_db"), f"snapshot {step} ASR")
        if esr <= 0.0 or metrics.get("asr_guard_passed") is not True:
            raise R2GateEvidenceError("screen snapshot ESR/ASR guard is invalid")
    selected = row.get("selected_checkpoint")
    if selected not in expected:
        raise R2GateEvidenceError("selected screen checkpoint is not common")
    validation = _mapping(row.get("selected_validation"), "selected validation")
    selected_metrics = _mapping(
        snapshots.get(str(selected), snapshots.get(selected)), "selected snapshot"
    )
    for metric in ("esr", "asr_db"):
        value = _finite(validation.get(metric), f"selected {metric}")
        snapshot_value = _finite(selected_metrics.get(metric), f"snapshot {metric}")
        if value != snapshot_value:
            raise R2GateEvidenceError(
                f"selected validation {metric} differs from its snapshot"
            )


def evaluate_screen_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select one loss and one deployable family/mode using validation only."""
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
            raise R2GateEvidenceError(f"duplicate screen trajectory: {key}")
        indexed[key] = row
    expected_base = set()
    for device in ("fulltone", "bigmuff"):
        for loss in ("m4", "wright"):
            expected_base.add((device, "a2", loss, "off"))
            for family in ("aa-nam", "aa-fssr"):
                expected_base.add((device, family, loss, "full_island_x2"))
    if not expected_base <= set(indexed):
        raise R2GateEvidenceError(
            f"initial screen matrix missing={sorted(expected_base - set(indexed))}"
        )
    loss_medians = {}
    for loss in ("m4", "wright"):
        esr_at_5000 = []
        for device in ("fulltone", "bigmuff"):
            for family in ("aa-nam", "aa-fssr"):
                row = indexed[(device, family, loss, "full_island_x2")]
                metrics = _mapping(
                    row["snapshots"].get("5000", row["snapshots"].get(5000)),
                    "5000 snapshot",
                )
                esr_at_5000.append(_finite(metrics.get("esr"), "5000 ESR"))
        loss_medians[loss] = float(np.median(esr_at_5000))
    promoted_loss = min(("m4", "wright"), key=lambda loss: (loss_medians[loss], loss))
    expected = expected_base | {
        (device, family, promoted_loss, "adaa1")
        for device in ("fulltone", "bigmuff")
        for family in ("aa-nam", "aa-fssr")
    }
    if set(indexed) != expected:
        raise R2GateEvidenceError(
            f"screen matrix mismatch; missing={sorted(expected - set(indexed))}, "
            f"extra={sorted(set(indexed) - expected)}"
        )
    for (device, family, loss, mode), row in indexed.items():
        expected_steps = 15000 if loss == promoted_loss else 5000
        if mode == "adaa1" and loss != promoted_loss:
            raise R2GateEvidenceError("ADAA may run only under the promoted loss")
        if row.get("optimizer_steps") != expected_steps:
            raise R2GateEvidenceError(
                f"{device}/{family}/{loss}/{mode} violates the loss stop rule"
            )
    baselines = {}
    for device in ("fulltone", "bigmuff"):
        options = [indexed[(device, "a2", loss, "off")] for loss in ("m4", "wright")]
        selected = min(
            options,
            key=lambda row: _finite(row["selected_validation"].get("esr"), "A2 ESR"),
        )
        baselines[device] = {
            "loss": selected["loss"],
            "esr": _finite(selected["selected_validation"].get("esr"), "A2 ESR"),
            "asr_db": _finite(selected["selected_validation"].get("asr_db"), "A2 ASR"),
        }
    candidates = []
    for family in ("aa-nam", "aa-fssr"):
        for mode in ("full_island_x2", "adaa1"):
            device_checks = {}
            improvements = []
            cpu_ratios = []
            for device in ("fulltone", "bigmuff"):
                row = indexed[(device, family, promoted_loss, mode)]
                metrics = row["selected_validation"]
                candidate_esr = _finite(metrics.get("esr"), "candidate ESR")
                improvement = (baselines[device]["esr"] - candidate_esr) / baselines[
                    device
                ]["esr"]
                asr_gain = baselines[device]["asr_db"] - _finite(
                    metrics.get("asr_db"), "candidate ASR"
                )
                cpu = _finite(row.get("projected_cpu_ratio_a2"), "projected CPU")
                if cpu <= 0.0:
                    raise R2GateEvidenceError("projected CPU ratio must be positive")
                improvements.append(improvement)
                cpu_ratios.append(cpu)
                device_checks[device] = {
                    "esr_relative_improvement": improvement,
                    "asr_gain_db": asr_gain,
                    "passed": improvement >= 0.15 and asr_gain >= 10.0,
                }
            candidates.append(
                {
                    "family": family,
                    "aa_mode": mode,
                    "passed": all(check["passed"] for check in device_checks.values()),
                    "median_esr_relative_improvement": float(np.median(improvements)),
                    "projected_cpu_ratio_a2": float(np.median(cpu_ratios)),
                    "devices": device_checks,
                }
            )
    eligible = [candidate for candidate in candidates if candidate["passed"]]
    selected_candidate = (
        min(
            eligible,
            key=lambda candidate: (
                -candidate["median_esr_relative_improvement"],
                candidate["projected_cpu_ratio_a2"],
                candidate["family"],
                candidate["aa_mode"],
            ),
        )
        if eligible
        else None
    )
    return {
        "format": "fssr-r2-screen-gate-v1",
        "valid": True,
        "passed": selected_candidate is not None,
        "terminal_if_failed": None if selected_candidate else "NO-GO-R2",
        "promoted_loss": promoted_loss,
        "loss_median_validation_esr_at_5000": loss_medians,
        "best_a2_by_device": baselines,
        "candidates": candidates,
        "selected_candidate": selected_candidate,
        "sealed_test_used": False,
    }


def evaluate_teacher_gate(
    rows: Sequence[Mapping[str, Any]], baselines: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Require the x4 teacher to pass ESR and ASR gates on both dev devices."""
    if len(rows) != 2:
        raise R2GateEvidenceError("teacher gate requires exactly two device rows")
    indexed = {}
    for row in rows:
        device = row.get("device")
        if device not in {"fulltone", "bigmuff"} or device in indexed:
            raise R2GateEvidenceError("teacher device matrix is invalid")
        if row.get("seed") != 0 or row.get("aa_mode") != "teacher_x4":
            raise R2GateEvidenceError("teacher must be x4 seed 0")
        if row.get("selection_split") != "validation":
            raise R2GateEvidenceError("teacher selection must use validation")
        if row.get("sealed_test_opened") is not False:
            raise R2GateEvidenceError("teacher evidence opened a sealed test")
        indexed[str(device)] = row
    checks = {}
    for device in ("fulltone", "bigmuff"):
        baseline = _mapping(baselines.get(device), f"{device} baseline")
        validation = _mapping(indexed[device].get("validation"), "teacher validation")
        baseline_esr = _finite(baseline.get("esr"), "baseline ESR")
        improvement = (
            baseline_esr - _finite(validation.get("esr"), "teacher ESR")
        ) / baseline_esr
        asr_gain = _finite(baseline.get("asr_db"), "baseline ASR") - _finite(
            validation.get("asr_db"), "teacher ASR"
        )
        checks[device] = {
            "esr_relative_improvement": improvement,
            "asr_gain_db": asr_gain,
            "passed": improvement >= 0.15 and asr_gain >= 10.0,
        }
    passed = all(check["passed"] for check in checks.values())
    return {
        "format": "fssr-r2-teacher-gate-v1",
        "passed": passed,
        "checks": checks,
        "terminal_if_failed": None if passed else "NO-GO-R2",
        "sealed_test_used": False,
    }


def validate_benchmark_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the interleaved same-binary benchmark and compute block-64 cost."""
    literal = {
        "same_binary": True,
        "abi": "float32",
        "compiler_optimization": "-Ofast",
        "lto_ipo": True,
        "same_isa": True,
        "repetitions": 30,
        "schedule": "interleaved_ab",
        "allocations_outside_timing": True,
    }
    for name, expected in literal.items():
        if evidence.get(name) != expected:
            raise R2GateEvidenceError(f"benchmark {name} must equal {expected!r}")
    parity_error = _finite(evidence.get("python_cpp_max_abs_error"), "parity error")
    if parity_error > 2.0e-5:
        raise R2GateEvidenceError("Python/C++ parity exceeds 2e-5")
    models = _mapping(evidence.get("models"), "benchmark models")
    if set(models) != {"a2", "candidate"}:
        raise R2GateEvidenceError("benchmark requires exactly A2 and candidate")
    summaries: dict[str, dict[str, Any]] = {}
    for model_name in ("a2", "candidate"):
        model = _mapping(models[model_name], model_name)
        blocks = _mapping(model.get("blocks"), f"{model_name} blocks")
        if {int(block) for block in blocks} != {1, 16, 64, 128}:
            raise R2GateEvidenceError(
                f"{model_name} benchmark block matrix is incomplete"
            )
        validated_blocks = {}
        for block in (1, 16, 64, 128):
            result = _mapping(
                blocks.get(str(block), blocks.get(block)), f"block {block}"
            )
            median = _finite(result.get("median_ns_per_sample"), "benchmark median")
            p95 = _finite(result.get("p95_ns_per_sample"), "benchmark p95")
            rtf = _finite(result.get("rtf"), "benchmark RTF")
            if median <= 0.0 or p95 < median or rtf <= 0.0:
                raise R2GateEvidenceError("benchmark timing values are inconsistent")
            validated_blocks[block] = {"median": median, "p95": p95, "rtf": rtf}
        sizes = {}
        for field in (
            "parameters",
            "weight_bytes",
            "persistent_state_bytes",
            "scratch_bytes",
            "latency_samples",
        ):
            value = _finite(model.get(field), f"{model_name}.{field}")
            minimum = 0.0 if field == "latency_samples" else 1.0
            if value < minimum or int(value) != value:
                raise R2GateEvidenceError(
                    f"{model_name}.{field} is not a valid reported integer size"
                )
            sizes[field] = int(value)
        summaries[model_name] = {"blocks": validated_blocks, **sizes}
    ratio = (
        summaries["candidate"]["blocks"][64]["median"]
        / summaries["a2"]["blocks"][64]["median"]
    )
    checks = {
        "cpu_ratio": ratio <= 1.25,
        "latency": summaries["candidate"]["latency_samples"] <= 48,
        "python_cpp_parity": True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "candidate_to_a2_block64_cpu_ratio": ratio,
        "candidate_latency_samples": summaries["candidate"]["latency_samples"],
        "models": summaries,
    }


def validate_mushra_design(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the locked BS.1534-3 session before reading listener scores."""
    literal = {
        "standard": "ITU-R BS.1534-3",
        "recruited_participants": 24,
        "double_blind": True,
        "randomized_order": True,
        "one_global_gain_per_trial": True,
        "hidden_hardware_reference": True,
    }
    for name, expected in literal.items():
        if evidence.get(name) != expected:
            raise R2GateEvidenceError(f"MUSHRA design {name} must equal {expected!r}")
    if evidence.get("conditions") != list(MUSHRA_CONDITIONS):
        raise R2GateEvidenceError("MUSHRA design condition set or order changed")
    excerpts = evidence.get("excerpts")
    if not isinstance(excerpts, list) or len(excerpts) != 8:
        raise R2GateEvidenceError("MUSHRA design requires eight excerpts")
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
            raise R2GateEvidenceError("MUSHRA excerpt IDs must be unique")
        if device not in {"fulltone", "bigmuff", "blackstar", "ua1176"}:
            raise R2GateEvidenceError("MUSHRA excerpt device is invalid")
        if item.get("duration_seconds") != 10 or item.get("sealed") is not True:
            raise R2GateEvidenceError("MUSHRA excerpts must be sealed ten-second audio")
        observed_ids.add(excerpt_id)
        counts[str(device)] += 1
    if counts != {
        device: 2 for device in ("fulltone", "bigmuff", "blackstar", "ua1176")
    }:
        raise R2GateEvidenceError("MUSHRA requires exactly two excerpts per device")
    exclusion_rule = evidence.get("preregistered_exclusion_rule")
    if exclusion_rule != "configs/r2/mushra_exclusion.yaml":
        raise R2GateEvidenceError("MUSHRA exclusion rule must use the frozen file")
    return {
        "passed": True,
        "standard": literal["standard"],
        "excerpts": 8,
        "conditions": list(MUSHRA_CONDITIONS),
    }


def _validate_confirmation_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("method") != "paired-hierarchical-r2-confirmation-v1":
        raise R2GateEvidenceError("confirmation bootstrap method is not frozen R2")
    if summary.get("replicates") != 10_000 or summary.get("seed") != 20_260_828:
        raise R2GateEvidenceError("confirmation bootstrap replicates/seed changed")
    if summary.get("hierarchy") != ["seed", "source"]:
        raise R2GateEvidenceError("confirmation bootstrap hierarchy changed")
    if summary.get("windows_resampled") is not False:
        raise R2GateEvidenceError("windows cannot be confirmation observations")
    if summary.get("probes_resampled") is not False:
        raise R2GateEvidenceError("probes cannot be confirmation observations")
    if summary.get("final_conditions") != FINAL_CONDITIONS:
        raise R2GateEvidenceError("confirmation must contain exactly 40 conditions")


def evaluate_prelisten_gate(
    confirmation: Mapping[str, Any], benchmark: Mapping[str, Any]
) -> dict[str, Any]:
    """Allow MUSHRA only after fidelity, aliasing, native cost, and parity pass."""
    _validate_confirmation_summary(confirmation)
    benchmark_result = validate_benchmark_evidence(benchmark)
    interval = _mapping(
        confirmation.get("esr_confidence_interval_95"), "ESR confidence interval"
    )
    lower = _finite(interval.get("lower"), "ESR lower bound")
    asr = _finite(confirmation.get("asr_reduction_db"), "ASR reduction")
    checks = {
        "esr": lower >= 0.15,
        "asr": asr >= 10.0,
        "cpu": benchmark_result["checks"]["cpu_ratio"],
        "latency": benchmark_result["checks"]["latency"],
        "python_cpp_parity": benchmark_result["checks"]["python_cpp_parity"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "benchmark": benchmark_result,
    }


def _validate_mushra_summary(summary: Mapping[str, Any]) -> tuple[float, float]:
    literal = {
        "method": "paired-hierarchical-r2-mushra-v1",
        "recruited_participants": 24,
        "excerpts_per_participant": 8,
        "replicates": 10_000,
        "seed": 20_260_829,
        "hierarchy": ["participant", "excerpt"],
        "design_validated": True,
    }
    for name, expected in literal.items():
        if summary.get(name) != expected:
            raise R2GateEvidenceError(f"MUSHRA {name} must equal {expected!r}")
    retained = summary.get("retained_participants")
    if isinstance(retained, bool) or not isinstance(retained, int) or retained < 20:
        raise R2GateEvidenceError("MUSHRA retained participants must be at least 20")
    advantage = _finite(summary.get("candidate_minus_a2_points"), "MUSHRA advantage")
    interval = _mapping(summary.get("confidence_interval_95"), "MUSHRA interval")
    lower = _finite(interval.get("lower"), "MUSHRA lower bound")
    return advantage, lower


def evaluate_final_verdict(
    confirmation: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    mushra: Mapping[str, Any],
    *,
    instrumentation_valid: bool,
) -> dict[str, Any]:
    """Return GO only when every simultaneous frozen R2 criterion succeeds."""
    if not instrumentation_valid:
        return {
            "verdict": "INVALID",
            "valid": False,
            "reason": "instrumentation_or_protocol_failure",
        }
    prelisten = evaluate_prelisten_gate(confirmation, benchmark)
    advantage, mushra_lower = _validate_mushra_summary(mushra)
    interval = _mapping(
        confirmation.get("esr_confidence_interval_95"), "ESR confidence interval"
    )
    won = confirmation.get("devices_won")
    if not isinstance(won, list) or len(set(won)) != len(won):
        raise R2GateEvidenceError("devices_won must be a unique list")
    if not set(won) <= {"fulltone", "bigmuff", "blackstar", "ua1176"}:
        raise R2GateEvidenceError("devices_won contains an unknown device")
    checks = {
        "esr_lower_bound": _finite(interval.get("lower"), "ESR lower bound") >= 0.15,
        "asr_reduction": _finite(confirmation.get("asr_reduction_db"), "ASR reduction")
        >= 10.0,
        "devices_won": len(won) >= 3,
        "mandatory_devices_won": {"blackstar", "ua1176"} <= set(won),
        "cpu_ratio": prelisten["checks"]["cpu"],
        "latency": prelisten["checks"]["latency"],
        "python_cpp_parity": prelisten["checks"]["python_cpp_parity"],
        "mushra_advantage": advantage > 10.0,
        "mushra_lower_bound": mushra_lower > 0.0,
    }
    passed = all(checks.values())
    return {
        "verdict": "GO-R2" if passed else "NO-GO-R2",
        "valid": True,
        "passed": passed,
        "checks": checks,
        "prelisten": prelisten,
    }
