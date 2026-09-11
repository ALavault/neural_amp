"""Pure fail-closed gates for QUALITY-AA-v1."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from fssr_nam.data.r2_fixtures import R2_FIXTURES

from .quality_aa_v1 import CAMPAIGN_VERSION

ROUTES = ("full_island_x2", "teacher_x4")


class QualityAAGateEvidenceError(ValueError):
    """Raised when evidence is malformed or incomplete."""


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualityAAGateEvidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise QualityAAGateEvidenceError(f"{label} must be finite")
    return result


def evaluate_preflight_gate(
    evidence: Mapping[str, Any], *, campaign_version: str = CAMPAIGN_VERSION
) -> dict[str, Any]:
    """Validate independent floor, filter, reference, and ADAA checks."""
    if evidence.get("campaign_version") != campaign_version:
        raise QualityAAGateEvidenceError("preflight campaign version changed")
    if evidence.get("candidate_route_outputs_observed") is not False:
        raise QualityAAGateEvidenceError("preflight observed x2/x4 outputs")
    if evidence.get("physical_audio_samples_read") != 0:
        raise QualityAAGateEvidenceError("preflight read physical audio")
    floor = evidence.get("floor_calibration")
    if not isinstance(floor, Mapping) or floor.get("passed") is not True:
        raise QualityAAGateEvidenceError("identity floor calibration failed")
    floor_db = _finite(floor.get("locked_floor_db"), "locked floor")
    floor_passed = floor_db <= -120.0
    filters = evidence.get("filters")
    if not isinstance(filters, Mapping) or set(filters) != set(ROUTES):
        raise QualityAAGateEvidenceError("preflight filter routes are incomplete")
    filter_passed = all(
        isinstance(filters[route], Mapping) and filters[route].get("passed") is True
        for route in ROUTES
    )
    reference = evidence.get("reference_convergence")
    if not isinstance(reference, Mapping):
        raise QualityAAGateEvidenceError("reference convergence is missing")
    reference_passed = (
        reference.get("conditions") == 54
        and reference.get("all_conditions_passed") is True
    )
    adaa = evidence.get("adaa_alignment")
    if not isinstance(adaa, Mapping):
        raise QualityAAGateEvidenceError("ADAA alignment evidence is missing")
    if adaa.get("eligible") is True:
        residual = _finite(
            adaa.get("maximum_global_alignment_residual_samples"), "ADAA residual"
        )
        if residual > 0.01 or adaa.get("native_compensation_parity") is not True:
            raise QualityAAGateEvidenceError("eligible ADAA route is not aligned")
        adaa_status = "eligible"
    else:
        if adaa.get("route_action") != "excluded_without_route_poisoning":
            raise QualityAAGateEvidenceError("unalignable ADAA action is invalid")
        adaa_status = "excluded_unalignable"
    passed = floor_passed and filter_passed and reference_passed
    return {
        "format": "fssr-quality-aa-preflight-gate-v1",
        "valid": True,
        "passed": passed,
        "checks": {
            "identity_floor": floor_passed,
            "filter_response_and_delay": filter_passed,
            "reference_x8_x16_convergence": reference_passed,
            "candidate_outputs_unobserved": True,
            "physical_audio_absent": True,
        },
        "locked_floor_db": floor_db,
        "adaa_status": adaa_status,
        "campaign_continue": passed,
    }


def _mechanism_rows(
    evidence: Mapping[str, Any], campaign_version: str
) -> list[Mapping[str, Any]]:
    if evidence.get("campaign_version") != campaign_version:
        raise QualityAAGateEvidenceError("mechanism campaign version changed")
    if evidence.get("reference_kind") != "direct_synthetic_x8_x16":
        raise QualityAAGateEvidenceError("mechanism reference kind changed")
    if evidence.get("physical_audio_samples_read") != 0:
        raise QualityAAGateEvidenceError("mechanism read physical audio")
    if evidence.get("same_weights_across_modes") is not True:
        raise QualityAAGateEvidenceError("mechanism weights differ across modes")
    rows = evidence.get("rows")
    if not isinstance(rows, list) or len(rows) != 18:
        raise QualityAAGateEvidenceError("mechanism requires exactly 18 rows")
    return rows


def evaluate_mechanism_gate(
    evidence: Mapping[str, Any], *, campaign_version: str = CAMPAIGN_VERSION
) -> dict[str, Any]:
    """Decide x2 and x4 independently using uncensored paired conditions."""
    rows = _mechanism_rows(evidence, campaign_version)
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        fixture = str(row.get("fixture"))
        mode = str(row.get("mode"))
        key = (fixture, mode)
        if key in indexed:
            raise QualityAAGateEvidenceError(f"duplicate mechanism row: {key}")
        conditions = row.get("conditions")
        if not isinstance(conditions, list) or len(conditions) != 9:
            raise QualityAAGateEvidenceError(f"row {key} lacks the exact grid")
        indexed[key] = row
    expected = {(fixture, mode) for fixture in R2_FIXTURES for mode in ("off", *ROUTES)}
    if set(indexed) != expected:
        raise QualityAAGateEvidenceError("mechanism row matrix is incomplete")

    route_results = {}
    for route in ROUTES:
        gains = []
        measured_asr = []
        reference_residual = []
        fixture_gains = {}
        guards = []
        for fixture in R2_FIXTURES:
            row = indexed[(fixture, route)]
            guards.append(row.get("guard_passed") is True)
            current_gains = []
            for condition in row["conditions"]:
                if not isinstance(condition, Mapping):
                    raise QualityAAGateEvidenceError("condition must be an object")
                if condition.get("off_asr_floor_censored") is True:
                    continue
                gain = _finite(condition.get("paired_asr_gain_db"), "ASR gain")
                gains.append(gain)
                current_gains.append(gain)
                measured_asr.append(_finite(condition.get("asr_linear"), "linear ASR"))
                reference_residual.append(
                    _finite(
                        condition.get("known_reference_alias_residual_linear"),
                        "known-reference residual",
                    )
                )
            if current_gains:
                fixture_gains[fixture] = float(np.median(current_gains))
        if not gains or set(fixture_gains) != set(R2_FIXTURES):
            raise QualityAAGateEvidenceError(
                f"route {route} lacks uncensored evidence for every fixture"
            )
        rho_result = spearmanr(measured_asr, reference_residual)
        rho = _finite(rho_result.statistic, f"{route} Spearman rho")
        median_gain = float(np.median(gains))
        reference_medians = [
            _finite(
                indexed[(fixture, route)]["known_reference_alias_residual"]["median"],
                "reference residual median",
            )
            for fixture in R2_FIXTURES
        ]
        checks = {
            "spearman": rho >= 0.90,
            "median_gain": median_gain >= 10.0,
            "every_fixture_gain": all(value >= 6.0 for value in fixture_gains.values()),
            "guards": all(guards),
            "rf2047_residual": _finite(
                indexed[("rf2047_residual", route)].get("residual_energy_ratio"),
                "RF2047 residual ratio",
            )
            >= 0.10,
            "latency": indexed[("tanh", route)].get("latency_samples", 999) <= 48,
        }
        route_results[route] = {
            "passed": all(checks.values()),
            "checks": checks,
            "spearman_rho": rho,
            "median_asr_gain_db": median_gain,
            "fixture_median_asr_gain_db": fixture_gains,
            "median_known_reference_alias_residual_db": float(
                np.median(reference_medians)
            ),
            "uncensored_conditions": len(gains),
        }
    passing = [route for route, result in route_results.items() if result["passed"]]
    return {
        "format": "fssr-quality-aa-mechanism-gate-v1",
        "valid": True,
        "passed": bool(passing),
        "campaign_continue": bool(passing),
        "route_decisions_independent": True,
        "routes": route_results,
        "passing_routes": passing,
        "terminal_if_stopped": None if passing else "NO-GO-QUALITY-AA-v1",
    }


def evaluate_native_gate(
    evidence: Mapping[str, Any],
    mechanism_gate: Mapping[str, Any],
    *,
    campaign_version: str = CAMPAIGN_VERSION,
) -> dict[str, Any]:
    """Require parity, latency, allocation discipline, and block-64 headroom."""
    if evidence.get("campaign_version") != campaign_version:
        raise QualityAAGateEvidenceError("native campaign version changed")
    routes = evidence.get("routes")
    if not isinstance(routes, Mapping):
        raise QualityAAGateEvidenceError("native route evidence is missing")
    required = set(mechanism_gate.get("passing_routes", ()))
    if set(routes) != required:
        raise QualityAAGateEvidenceError("native routes differ from mechanism pass set")
    results = {}
    for route in sorted(required):
        item = routes[route]
        if not isinstance(item, Mapping):
            raise QualityAAGateEvidenceError("native route must be an object")
        parity = item.get("parity")
        benchmark = item.get("benchmark")
        if not isinstance(parity, Mapping) or not isinstance(benchmark, Mapping):
            raise QualityAAGateEvidenceError("parity or benchmark evidence is missing")
        models = benchmark.get("models")
        if not isinstance(models, Mapping):
            raise QualityAAGateEvidenceError("benchmark models are missing")
        candidate = models.get("candidate")
        if not isinstance(candidate, Mapping):
            raise QualityAAGateEvidenceError("candidate benchmark is missing")
        blocks = candidate.get("blocks")
        if not isinstance(blocks, Mapping) or set(blocks) != {"1", "16", "64", "128"}:
            raise QualityAAGateEvidenceError("benchmark block matrix is incomplete")
        block64 = blocks["64"]
        block64_p95 = _finite(block64.get("p95_ns_per_sample"), "block64 p95")
        block64_p95_rtf = block64_p95 / (1.0e9 / 48_000.0)
        checks = {
            "python_cpp_parity": parity.get("passed") is True
            and _finite(parity.get("max_abs_error"), "parity error") <= 2.0e-5,
            "reset": parity.get("reset_verified") is True,
            "latency": _finite(candidate.get("latency_samples"), "latency") <= 48,
            "same_binary": benchmark.get("same_binary") is True,
            "scientific_repetitions": benchmark.get("repetitions") == 30
            and benchmark.get("scientific_eligible") is True,
            "compiler": benchmark.get("compiler_optimization") == "-Ofast"
            and benchmark.get("lto_ipo") is True
            and benchmark.get("same_isa") is True,
            "allocations": benchmark.get("allocations_outside_timing") is True,
            "block64_realtime": block64_p95_rtf <= 0.80,
        }
        results[route] = {
            "passed": all(checks.values()),
            "checks": checks,
            "block64_p95_ns_per_sample": block64_p95,
            "block64_p95_rtf": block64_p95_rtf,
            "latency_samples": candidate.get("latency_samples"),
            "sizes": {
                name: candidate.get(name)
                for name in (
                    "parameters",
                    "weight_bytes",
                    "persistent_state_bytes",
                    "scratch_bytes",
                    "native_state_allocation_bytes",
                )
            },
        }
    passing = [route for route, result in results.items() if result["passed"]]
    return {
        "format": "fssr-quality-aa-native-gate-v1",
        "valid": True,
        "passed": bool(passing),
        "routes": results,
        "passing_routes": passing,
    }


def final_backend_decision(
    mechanism_gate: Mapping[str, Any],
    native_gate: Mapping[str, Any],
    *,
    campaign_version: str = CAMPAIGN_VERSION,
    go_verdict: str = "GO-QUALITY-AA-v1",
    no_go_verdict: str = "NO-GO-QUALITY-AA-v1",
) -> dict[str, Any]:
    """Select the admissible quality-cost route or return a valid no-go."""
    admissible = list(native_gate.get("passing_routes", ()))
    if not admissible:
        return {
            "campaign_version": campaign_version,
            "verdict": no_go_verdict,
            "selected_backend": None,
            "reason": "no route passed mechanism, parity, latency, and runtime",
        }
    qualities = {
        route: float(
            mechanism_gate["routes"][route]["median_known_reference_alias_residual_db"]
        )
        for route in admissible
    }
    best_quality = min(qualities.values())
    contenders = [
        route for route, value in qualities.items() if value <= best_quality + 0.5
    ]
    selected = min(
        contenders,
        key=lambda route: native_gate["routes"][route]["block64_p95_ns_per_sample"],
    )
    return {
        "campaign_version": campaign_version,
        "verdict": go_verdict,
        "selected_backend": selected,
        "admissible_routes": admissible,
        "quality_db": qualities,
        "quality_indifference_db": 0.5,
        "tiebreaker": "lowest_block64_p95_ns_per_sample",
        "claim_boundary": "synthetic AA backend only",
    }
