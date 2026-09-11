from __future__ import annotations

from copy import deepcopy

import pytest

from fssr_nam.campaign.r2_gates import (
    FIXTURES,
    MECHANISM_MODES,
    R2GateEvidenceError,
    evaluate_final_verdict,
    evaluate_mechanism_gate,
    evaluate_prelisten_gate,
    evaluate_screen_gate,
    evaluate_teacher_gate,
    validate_benchmark_evidence,
    validate_mushra_design,
)


def _mechanism_rows(*, x2_gain: float = 12.0, adaa_gain: float = 7.0) -> list[dict]:
    rows = []
    mode_gain = {
        "off": 0.0,
        "full_island_x2": x2_gain,
        "adaa1": adaa_gain,
        "teacher_x4": 15.0,
    }
    for fixture_index, fixture in enumerate(FIXTURES):
        off_asr = -5.0 + fixture_index
        for mode in MECHANISM_MODES:
            asr = off_asr - mode_gain[mode]
            rows.append(
                {
                    "fixture": fixture,
                    "mode": mode,
                    "weights_id": f"weights-{fixture}",
                    "asr_db": asr,
                    "reference_192khz_asr_db": asr,
                    "guard_passed": True,
                    "fundamental_complex_error": 5.0e-6,
                    "latency_samples": 16 if "x" in mode else 1,
                    "residual_energy_ratio": (
                        0.10 if fixture == "rf2047_residual" else 0.0
                    ),
                }
            )
    return rows


def _benchmark() -> dict:
    def model(median: float, latency: int) -> dict:
        return {
            "parameters": 100,
            "weight_bytes": 400,
            "persistent_state_bytes": 128,
            "scratch_bytes": 256,
            "latency_samples": latency,
            "blocks": {
                str(block): {
                    "median_ns_per_sample": median,
                    "p95_ns_per_sample": 1.1 * median,
                    "rtf": 0.1,
                }
                for block in (1, 16, 64, 128)
            },
        }

    return {
        "same_binary": True,
        "abi": "float32",
        "compiler_optimization": "-Ofast",
        "lto_ipo": True,
        "same_isa": True,
        "repetitions": 30,
        "schedule": "interleaved_ab",
        "allocations_outside_timing": True,
        "python_cpp_max_abs_error": 1.0e-5,
        "models": {"a2": model(100.0, 0), "candidate": model(120.0, 16)},
    }


def _confirmation() -> dict:
    return {
        "method": "paired-hierarchical-r2-confirmation-v1",
        "replicates": 10000,
        "seed": 20260828,
        "hierarchy": ["seed", "source"],
        "windows_resampled": False,
        "probes_resampled": False,
        "final_conditions": 40,
        "esr_confidence_interval_95": {"lower": 0.16, "upper": 0.22},
        "asr_reduction_db": 11.0,
        "devices_won": ["fulltone", "blackstar", "ua1176"],
    }


def _mushra() -> dict:
    return {
        "method": "paired-hierarchical-r2-mushra-v1",
        "recruited_participants": 24,
        "retained_participants": 20,
        "excerpts_per_participant": 8,
        "replicates": 10000,
        "seed": 20260829,
        "hierarchy": ["participant", "excerpt"],
        "design_validated": True,
        "candidate_minus_a2_points": 11.0,
        "confidence_interval_95": {"lower": 1.0, "upper": 18.0},
    }


def _mushra_design() -> dict:
    return {
        "standard": "ITU-R BS.1534-3",
        "recruited_participants": 24,
        "double_blind": True,
        "randomized_order": True,
        "one_global_gain_per_trial": True,
        "hidden_hardware_reference": True,
        "conditions": [
            "hidden_hardware_reference",
            "candidate",
            "a2",
            "aa_off_ablation",
            "lowpass_anchor_3p5khz",
            "aliasing_anchor",
        ],
        "excerpts": [
            {
                "excerpt": f"{device}-{index}",
                "device": device,
                "duration_seconds": 10,
                "sealed": True,
            }
            for device in ("fulltone", "bigmuff", "blackstar", "ua1176")
            for index in range(2)
        ],
        "preregistered_exclusion_rule": "configs/r2/mushra_exclusion.yaml",
    }


def _screen_row(
    device: str,
    family: str,
    loss: str,
    mode: str,
    *,
    optimizer_steps: int,
    final_esr: float,
    asr_db: float,
    esr_at_5000: float | None = None,
    cpu: float = 1.0,
) -> dict:
    checkpoint_steps = [200, 1000, 5000]
    if optimizer_steps == 15000:
        checkpoint_steps.append(15000)
    snapshots = {
        str(step): {
            "esr": (
                esr_at_5000 if step == 5000 and esr_at_5000 is not None else final_esr
            ),
            "asr_db": asr_db,
            "asr_guard_passed": True,
        }
        for step in checkpoint_steps
    }
    selected = checkpoint_steps[-1]
    return {
        "status": "complete",
        "device": device,
        "family": family,
        "loss": loss,
        "aa_mode": mode,
        "seed": 0,
        "selection_split": "validation",
        "sealed_test_opened": False,
        "optimizer_steps": optimizer_steps,
        "checkpoint_steps": checkpoint_steps,
        "snapshots": snapshots,
        "selected_checkpoint": selected,
        "selected_validation": dict(snapshots[str(selected)]),
        "projected_cpu_ratio_a2": cpu,
    }


def _screen_rows() -> list[dict]:
    rows = []
    for device in ("fulltone", "bigmuff"):
        rows.extend(
            [
                _screen_row(
                    device,
                    "a2",
                    "m4",
                    "off",
                    optimizer_steps=15000,
                    final_esr=0.2,
                    asr_db=-5.0,
                ),
                _screen_row(
                    device,
                    "a2",
                    "wright",
                    "off",
                    optimizer_steps=5000,
                    final_esr=0.3,
                    asr_db=-5.0,
                ),
            ]
        )
        for family, final_esr in (("aa-nam", 0.16), ("aa-fssr", 0.14)):
            rows.append(
                _screen_row(
                    device,
                    family,
                    "m4",
                    "full_island_x2",
                    optimizer_steps=15000,
                    final_esr=final_esr,
                    esr_at_5000=final_esr + 0.02,
                    asr_db=-17.0,
                    cpu=1.1 if family == "aa-nam" else 1.2,
                )
            )
            rows.append(
                _screen_row(
                    device,
                    family,
                    "wright",
                    "full_island_x2",
                    optimizer_steps=5000,
                    final_esr=0.4,
                    asr_db=-17.0,
                    cpu=1.1,
                )
            )
            rows.append(
                _screen_row(
                    device,
                    family,
                    "m4",
                    "adaa1",
                    optimizer_steps=15000,
                    final_esr=final_esr + 0.01,
                    asr_db=-16.0,
                    cpu=1.0,
                )
            )
    return rows


def test_mechanism_x2_is_mandatory_and_adaa_can_be_rejected_alone() -> None:
    result = evaluate_mechanism_gate(_mechanism_rows())
    assert result["r2_continue"] is True
    assert result["x2"]["passed"] is True
    assert result["adaa"]["passed"] is False
    assert result["adaa_route"] == "rejected_only"
    assert result["sealed_test_used"] is False
    stopped = evaluate_mechanism_gate(_mechanism_rows(x2_gain=5.0))
    assert stopped["r2_continue"] is False
    assert stopped["terminal_if_stopped"] == "NO-GO-R2"


def test_mechanism_rejects_different_weights_between_aa_modes() -> None:
    rows = _mechanism_rows()
    rows[1]["weights_id"] = "different"
    with pytest.raises(R2GateEvidenceError, match="identical weights"):
        evaluate_mechanism_gate(rows)


def test_screen_selects_loss_best_a2_per_device_and_promoted_family() -> None:
    result = evaluate_screen_gate(_screen_rows())
    assert result["passed"] is True
    assert result["promoted_loss"] == "m4"
    assert result["best_a2_by_device"]["fulltone"]["loss"] == "m4"
    assert result["selected_candidate"]["family"] == "aa-fssr"
    assert result["selected_candidate"]["aa_mode"] == "full_island_x2"
    assert result["sealed_test_used"] is False


def test_screen_rejects_nonpromoted_loss_continuation() -> None:
    rows = _screen_rows()
    wright = next(
        row for row in rows if row["family"] == "a2" and row["loss"] == "wright"
    )
    wright["optimizer_steps"] = 15000
    wright["checkpoint_steps"].append(15000)
    wright["snapshots"]["15000"] = dict(wright["snapshots"]["5000"])
    wright["selected_checkpoint"] = 15000
    wright["selected_validation"] = dict(wright["snapshots"]["15000"])
    with pytest.raises(R2GateEvidenceError, match="loss stop rule"):
        evaluate_screen_gate(rows)


def test_teacher_failure_is_terminal_no_go() -> None:
    baselines = {
        device: {"esr": 0.2, "asr_db": -5.0} for device in ("fulltone", "bigmuff")
    }
    rows = [
        {
            "device": device,
            "seed": 0,
            "aa_mode": "teacher_x4",
            "selection_split": "validation",
            "sealed_test_opened": False,
            "validation": {"esr": 0.18, "asr_db": -17.0},
        }
        for device in ("fulltone", "bigmuff")
    ]
    result = evaluate_teacher_gate(rows, baselines)
    assert result["passed"] is False
    assert result["terminal_if_failed"] == "NO-GO-R2"


def test_benchmark_requires_all_native_controls_and_computes_primary_ratio() -> None:
    result = validate_benchmark_evidence(_benchmark())
    assert result["passed"] is True
    assert result["candidate_to_a2_block64_cpu_ratio"] == 1.2
    invalid = _benchmark()
    invalid["python_cpp_max_abs_error"] = 2.1e-5
    with pytest.raises(R2GateEvidenceError, match="parity"):
        validate_benchmark_evidence(invalid)


def test_listening_launch_and_final_go_require_every_simultaneous_gate() -> None:
    assert validate_mushra_design(_mushra_design())["passed"] is True
    prelisten = evaluate_prelisten_gate(_confirmation(), _benchmark())
    assert prelisten["passed"] is True
    result = evaluate_final_verdict(
        _confirmation(), _benchmark(), _mushra(), instrumentation_valid=True
    )
    assert result["verdict"] == "GO-R2"
    assert all(result["checks"].values())
    weak_mushra = deepcopy(_mushra())
    weak_mushra["candidate_minus_a2_points"] = 10.0
    no_go = evaluate_final_verdict(
        _confirmation(), _benchmark(), weak_mushra, instrumentation_valid=True
    )
    assert no_go["verdict"] == "NO-GO-R2"
    assert no_go["checks"]["mushra_advantage"] is False


def test_instrumentation_failure_is_invalid_not_no_go() -> None:
    result = evaluate_final_verdict({}, {}, {}, instrumentation_valid=False)
    assert result == {
        "verdict": "INVALID",
        "valid": False,
        "reason": "instrumentation_or_protocol_failure",
    }
