from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from fssr_nam.campaign.amp_sota_gates import (
    SotaPrototypeGateError,
    evaluate_confirmation_gate,
    evaluate_mechanism_promotion_gate,
    evaluate_preflight_gate,
)
from fssr_nam.campaign.amp_sota_prototype_v1 import CAMPAIGN_VERSION

ROOT = Path(__file__).resolve().parents[2]


def _protocol() -> dict[str, object]:
    return yaml.safe_load(
        (ROOT / "configs/amp_sota_prototype_v1/protocol.yaml").read_text(
            encoding="utf-8"
        )
    )


def _preflight() -> dict[str, object]:
    return {
        "campaign_version": CAMPAIGN_VERSION,
        "parent_campaign": "AMP-QUALITY-ARCH-v3",
        "parent_verdict": "INVALID",
        "protocol_frozen": True,
        "data_audit_passed": True,
        "tests_passed": True,
        "lint_passed": True,
        "public_license_audit_passed": True,
        "source_file_disjoint_splits_verified": True,
        "physical_audio_samples_read": 0,
        "confirmation_audio_samples_read": 0,
        "fm9_audio_samples_read": 0,
        "failed_or_invalid_run_resumed": False,
    }


def test_preflight_requires_all_clean_boundaries() -> None:
    gate = evaluate_preflight_gate(_preflight(), _protocol())
    assert gate["valid"] is True
    assert gate["passed"] is True
    assert all(gate["checks"].values())

    failed = _preflight()
    failed["data_audit_passed"] = False
    gate = evaluate_preflight_gate(failed, _protocol())
    assert gate["passed"] is False
    assert gate["checks"]["data_audit_passed"] is False


def test_preflight_rejects_missing_or_malformed_evidence() -> None:
    missing = _preflight()
    missing.pop("tests_passed")
    with pytest.raises(SotaPrototypeGateError, match="must be boolean"):
        evaluate_preflight_gate(missing, _protocol())

    crossed = _preflight()
    crossed["confirmation_audio_samples_read"] = 1
    gate = evaluate_preflight_gate(crossed, _protocol())
    assert gate["passed"] is False
    assert gate["checks"]["confirmation_locked"] is False


def _mechanism() -> dict[str, object]:
    return {
        "campaign_version": CAMPAIGN_VERSION,
        "evidence_tier": "SYNTHETIC",
        "seed": 20_260_830,
        "physical_audio_samples_read": 0,
        "axes": {
            "approximant": {
                "other_axes_at_control": True,
                "control": {
                    "name": "cubic_hermite_c1",
                    "metrics": {
                        "asr_db": 50.0,
                        "approximation_error": 0.10,
                        "cost": 100.0,
                    },
                },
                "candidates": [
                    {
                        "name": "quintic_hermite_c2",
                        "metrics": {
                            "asr_db": 53.0,
                            "approximation_error": 0.101,
                            "cost": 110.0,
                        },
                    },
                    {
                        "name": "safe_rational_4_3",
                        "metrics": {
                            "asr_db": 48.0,
                            "approximation_error": 0.08,
                            "cost": 90.0,
                        },
                    },
                ],
            },
            "slow_control": {
                "other_axes_at_control": True,
                "control": {
                    "name": "causal_zero_order_hold",
                    "metrics": {"dynamic_esr": 1.0, "parasite_db": -60.0},
                },
                "candidates": [
                    {
                        "name": "causal_exponential_hold",
                        "causal": True,
                        "metrics": {
                            "dynamic_esr": 0.95,
                            "parasite_db": -59.5,
                            "lookahead_samples": 0,
                        },
                    },
                    {
                        "name": "causal_slope_limited_hold",
                        "causal": True,
                        "metrics": {
                            "dynamic_esr": 0.96,
                            "parasite_db": -60.0,
                            "lookahead_samples": 0,
                        },
                    },
                ],
            },
            "resampler": {
                "other_axes_at_control": True,
                "control": {
                    "name": "kaiser_windowed_sinc",
                    "metrics": {"stopband_attenuation_db": 90.0, "cost": 100.0},
                },
                "candidates": [
                    {
                        "name": "equiripple_halfband_polyphase",
                        "metrics": {
                            "stopband_attenuation_db": 89.5,
                            "cost": 75.0,
                        },
                    }
                ],
            },
        },
    }


def test_mechanism_gate_applies_inclusive_thresholds_and_one_promotion_per_axis() -> (
    None
):
    gate = evaluate_mechanism_promotion_gate(_mechanism(), _protocol())
    assert gate["valid"] is True
    assert gate["passed"] is True
    assert gate["promoted_components"] == {
        "approximant": "quintic_hermite_c2",
        "slow_control": "causal_exponential_hold",
        "resampler": "equiripple_halfband_polyphase",
    }
    assert gate["maximum_promotions_per_axis"] == 1
    approximant = gate["axis_results"]["approximant"]
    assert approximant["candidate_results"]["quintic_hermite_c2"]["passed"] is True
    assert approximant["candidate_results"]["safe_rational_4_3"]["passed"] is False


def test_approximant_cost_route_accepts_only_asr_noninferiority() -> None:
    evidence = _mechanism()
    rational = evidence["axes"]["approximant"]["candidates"][1]
    rational["metrics"].update({"asr_db": 49.5, "cost": 75.0})
    gate = evaluate_mechanism_promotion_gate(evidence, _protocol())
    result = gate["axis_results"]["approximant"]["candidate_results"][
        "safe_rational_4_3"
    ]
    assert result["checks"]["quality_route"] is False
    assert result["checks"]["cost_route"] is True


def test_mechanism_gate_never_infers_unavailable_native_cost() -> None:
    evidence = _mechanism()
    approximant = evidence["axes"]["approximant"]
    approximant["cost_measurement_available"] = False
    approximant["control"]["metrics"].pop("cost")
    for row in approximant["candidates"]:
        row["metrics"].pop("cost")
    resampler = evidence["axes"]["resampler"]
    resampler["cost_measurement_available"] = False
    resampler["control"]["metrics"].pop("cost")
    resampler["candidates"][0]["metrics"].pop("cost")

    gate = evaluate_mechanism_promotion_gate(evidence, _protocol())

    assert gate["axis_results"]["resampler"]["promoted_candidate"] is None
    rational = gate["axis_results"]["approximant"]["candidate_results"][
        "safe_rational_4_3"
    ]
    assert rational["checks"]["cost_route"] is False
    assert rational["cost_reduction"] is None


def test_mechanism_gate_rejects_nonisolated_incomplete_or_nonfinite_evidence() -> None:
    nonisolated = _mechanism()
    nonisolated["axes"]["slow_control"]["other_axes_at_control"] = False
    with pytest.raises(SotaPrototypeGateError, match="not isolated"):
        evaluate_mechanism_promotion_gate(nonisolated, _protocol())

    incomplete = _mechanism()
    incomplete["axes"]["approximant"]["candidates"].pop()
    with pytest.raises(SotaPrototypeGateError, match="incomplete"):
        evaluate_mechanism_promotion_gate(incomplete, _protocol())

    nonfinite = _mechanism()
    nonfinite["axes"]["resampler"]["candidates"][0]["metrics"]["cost"] = float("nan")
    with pytest.raises(SotaPrototypeGateError, match="finite"):
        evaluate_mechanism_promotion_gate(nonfinite, _protocol())


def _confirmation() -> dict[str, object]:
    rows = []
    for device in ("blackstar", "ua1176"):
        for seed in range(5):
            for source in ("source_a", "source_b"):
                rows.append(
                    {
                        "device": device,
                        "seed": seed,
                        "source": source,
                        "baseline": {
                            "esr": 1.0,
                            "mae": 1.0,
                            "log_mel": 1.0,
                            "mrstft": 1.0,
                            "alias_residual_db": -70.0,
                        },
                        "candidate": {
                            "esr": 0.90,
                            "mae": 1.05,
                            "log_mel": 1.05,
                            "mrstft": 1.05,
                            "alias_residual_db": -70.0,
                        },
                    }
                )
    return {
        "campaign_version": CAMPAIGN_VERSION,
        "candidate_locked_before_confirmation": True,
        "baseline_selected_on_development_only": True,
        "same_pairs_and_splits": True,
        "confirmation_excluded_from_selection": True,
        "baseline_name": "nam_a2_full",
        "candidate_name": "slow_long_tcn_x2_quintic",
        "rows": rows,
        "bootstrap": {
            "method": "paired-hierarchical-sota-confirmation-v1",
            "replicates": 10_000,
            "seed": 20_260_830,
            "hierarchy": ["seed", "source"],
            "devices_fixed_strata": ["blackstar", "ua1176"],
            "windows_resampled": False,
            "esr_relative_improvement_lower_95_bound": 0.001,
        },
        "runtime": {
            "finite_outputs": True,
            "deterministic_reset": True,
            "block_parity_max_absolute_error": 2.0e-5,
            "latency_samples": 64,
            "benchmark_block_size": 128,
            "p95_realtime_factor": 0.999,
        },
    }


def test_confirmation_gate_accepts_all_inclusive_bounds_except_strict_ones() -> None:
    gate = evaluate_confirmation_gate(_confirmation(), _protocol())
    assert gate["valid"] is True
    assert gate["passed"] is True
    assert gate["observation_count"] == 20
    assert gate["median_esr_relative_improvement"] == pytest.approx(0.10)
    assert gate["maximum_per_device_metric_relative_regressions"] == {
        "mae": pytest.approx(0.05),
        "log_mel": pytest.approx(0.05),
        "mrstft": pytest.approx(0.05),
    }
    assert gate["checks"]["block_parity"] is True
    assert gate["checks"]["latency"] is True


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("bootstrap_zero", "bootstrap_lower_95_bound"),
        ("rtf_one", "p95_realtime_factor"),
        ("mae_regression", "mae_noninferiority"),
        ("alias_regression", "aliasing_no_regression"),
    ],
)
def test_confirmation_gate_stops_on_each_scientific_or_runtime_failure(
    mutation: str, failed_check: str
) -> None:
    evidence = _confirmation()
    if mutation == "bootstrap_zero":
        evidence["bootstrap"]["esr_relative_improvement_lower_95_bound"] = 0.0
    elif mutation == "rtf_one":
        evidence["runtime"]["p95_realtime_factor"] = 1.0
    elif mutation == "mae_regression":
        for row in evidence["rows"]:
            if row["device"] == "ua1176":
                row["candidate"]["mae"] = 1.051
    else:
        evidence["rows"][0]["candidate"]["alias_residual_db"] = -69.999
    gate = evaluate_confirmation_gate(evidence, _protocol())
    assert gate["passed"] is False
    assert gate["checks"][failed_check] is False


def test_confirmation_gate_rejects_missing_seed_duplicate_source_and_nan() -> None:
    missing_seed = _confirmation()
    missing_seed["rows"] = [row for row in missing_seed["rows"] if row["seed"] != 4]
    with pytest.raises(SotaPrototypeGateError, match="every frozen seed"):
        evaluate_confirmation_gate(missing_seed, _protocol())

    duplicate = _confirmation()
    duplicate["rows"].append(copy.deepcopy(duplicate["rows"][0]))
    with pytest.raises(SotaPrototypeGateError, match="duplicate"):
        evaluate_confirmation_gate(duplicate, _protocol())

    nonfinite = _confirmation()
    nonfinite["rows"][0]["candidate"]["esr"] = float("nan")
    with pytest.raises(SotaPrototypeGateError, match="finite"):
        evaluate_confirmation_gate(nonfinite, _protocol())


def test_gate_rejects_decision_threshold_drift_not_covered_by_base_validator() -> None:
    protocol = _protocol()
    protocol["prototype"]["p95_realtime_factor_strictly_less_than"] = 1.01
    with pytest.raises(SotaPrototypeGateError, match="prototype"):
        evaluate_preflight_gate(_preflight(), protocol)
