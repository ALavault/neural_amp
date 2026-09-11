from __future__ import annotations

from copy import deepcopy

import pytest

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
from fssr_nam.campaign.r2_gates import FIXTURES, MECHANISM_MODES


def _mechanism_rows(*, x2_gain: float = 12.0) -> list[dict]:
    rows = []
    gains = {
        "off": 0.0,
        "full_island_x2": x2_gain,
        "adaa1": 11.0,
        "teacher_x4": 15.0,
    }
    for fixture_index, fixture in enumerate(FIXTURES):
        off_asr = -5.0 + fixture_index
        for mode in MECHANISM_MODES:
            asr = off_asr - gains[mode]
            rows.append(
                {
                    "fixture": fixture,
                    "mode": mode,
                    "weights_id": f"weights-{fixture}",
                    "asr_db": asr,
                    "reference_192khz_asr_db": asr,
                    "reference_kind": "synthetic_192khz",
                    "physical_hardware_reference_used": False,
                    "guard_passed": True,
                    "fundamental_complex_error": 5.0e-6,
                    "latency_samples": 16 if "x" in mode else 1,
                    "residual_energy_ratio": (
                        0.10 if fixture == "rf2047_residual" else 0.0
                    ),
                }
            )
    return rows


def _screen_row(
    device: str,
    family: str,
    loss: str,
    mode: str,
    *,
    optimizer_steps: int,
    final_esr: float,
    esr_at_5000: float | None = None,
    cpu: float = 1.0,
) -> dict:
    checkpoint_steps = [200, 1000, 5000]
    if optimizer_steps == 15_000:
        checkpoint_steps.append(15_000)
    snapshots = {
        str(step): {
            "esr": (
                esr_at_5000 if step == 5000 and esr_at_5000 is not None else final_esr
            ),
            "output_guard_passed": True,
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
        "physical_asr_used_for_selection": False,
        "optimizer_steps": optimizer_steps,
        "checkpoint_steps": checkpoint_steps,
        "snapshots": snapshots,
        "selected_checkpoint": selected,
        "selected_validation": dict(snapshots[str(selected)]),
        "projected_cpu_ratio_a2": cpu,
    }


def _screen_rows() -> list[dict]:
    rows = []
    final = {"aa-nam": 0.17, "aa-fssr": 0.16, "aa-fssr-xl": 0.13}
    for device in ("fulltone", "bigmuff"):
        rows.extend(
            [
                _screen_row(
                    device,
                    "a2",
                    "m4",
                    "off",
                    optimizer_steps=15_000,
                    final_esr=0.20,
                ),
                _screen_row(
                    device,
                    "a2",
                    "wright",
                    "off",
                    optimizer_steps=5000,
                    final_esr=0.25,
                ),
            ]
        )
        for family in ("aa-nam", "aa-fssr", "aa-fssr-xl"):
            rows.append(
                _screen_row(
                    device,
                    family,
                    "m4",
                    "full_island_x2",
                    optimizer_steps=15_000,
                    final_esr=final[family],
                    esr_at_5000=final[family] + 0.02,
                    cpu=1.0 if family == "aa-nam" else 1.2,
                )
            )
            rows.append(
                _screen_row(
                    device,
                    family,
                    "wright",
                    "full_island_x2",
                    optimizer_steps=5000,
                    final_esr=0.30,
                    cpu=1.0,
                )
            )
            rows.append(
                _screen_row(
                    device,
                    family,
                    "m4",
                    "adaa1",
                    optimizer_steps=15_000,
                    final_esr=final[family] + 0.01,
                    cpu=0.95 if family == "aa-nam" else 1.1,
                )
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
        "method": "paired-hierarchical-r2-48k-confirmation-v1",
        "replicates": 10_000,
        "seed": 20_260_828,
        "hierarchy": ["seed", "source"],
        "primary_devices": ["blackstar", "ua1176"],
        "development_devices": ["fulltone", "bigmuff"],
        "development_excluded_from_primary_interval": True,
        "windows_resampled": False,
        "probes_resampled": False,
        "evaluation_conditions": 40,
        "prospective_conditions": 20,
        "physical_asr_in_primary_decision": False,
        "heldout_esr_confidence_interval_95": {"lower": 0.16, "upper": 0.22},
        "devices_won": ["fulltone", "blackstar", "ua1176"],
    }


def _mushra() -> dict:
    return {
        "method": "paired-hierarchical-r2-48k-mushra-v1",
        "recruited_participants": 24,
        "retained_participants": 20,
        "excerpts_per_participant": 8,
        "primary_excerpts_per_participant": 4,
        "replicates": 10_000,
        "seed": 20_260_829,
        "hierarchy": ["participant", "primary_excerpt"],
        "primary_devices": ["blackstar", "ua1176"],
        "development_excluded_from_primary_interval": True,
        "design_validated": True,
        "primary_candidate_minus_a2_points": 11.0,
        "primary_confidence_interval_95": {"lower": 1.0, "upper": 18.0},
    }


def _mushra_design() -> dict:
    devices = ("fulltone", "bigmuff", "blackstar", "ua1176")
    return {
        "standard": "ITU-R BS.1534-3",
        "recruited_participants": 24,
        "double_blind": True,
        "randomized_order": True,
        "one_global_gain_per_trial": True,
        "hidden_hardware_reference": True,
        "primary_devices": ["blackstar", "ua1176"],
        "development_secondary_devices": ["fulltone", "bigmuff"],
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
                "sealed": device in {"blackstar", "ua1176"},
                "evidence_role": (
                    "prospective_primary"
                    if device in {"blackstar", "ua1176"}
                    else "development_secondary"
                ),
            }
            for device in devices
            for index in range(2)
        ],
        "preregistered_exclusion_rule": "configs/r2_48k/mushra_exclusion.yaml",
    }


def test_mechanism_is_synthetic_only_and_x2_remains_mandatory() -> None:
    result = evaluate_mechanism_gate(_mechanism_rows())
    assert result["campaign_continue"] is True
    assert result["x2"]["passed"] is True
    assert result["supports_physical_hardware_aliasing_claim"] is False
    stopped = evaluate_mechanism_gate(_mechanism_rows(x2_gain=5.0))
    assert stopped["campaign_continue"] is False
    assert stopped["terminal_if_stopped"] == "NO-GO-R2-48K"
    contaminated = _mechanism_rows()
    contaminated[0]["physical_hardware_reference_used"] = True
    with pytest.raises(R248KGateEvidenceError, match="must not claim"):
        evaluate_mechanism_gate(contaminated)


def test_screen_admits_xl_and_uses_only_validation_esr() -> None:
    result = evaluate_screen_gate(_screen_rows(), adaa_eligible=True)
    assert result["passed"] is True
    assert result["promoted_loss"] == "m4"
    assert result["selected_candidate"]["family"] == "aa-fssr-xl"
    assert result["selected_candidate"]["aa_mode"] == "full_island_x2"
    assert result["deployable_fidelity_passed"] is True
    assert result["physical_asr_used_for_selection"] is False
    contaminated = _screen_rows()
    contaminated[0]["physical_asr_used_for_selection"] = True
    with pytest.raises(R248KGateEvidenceError, match="physical ASR"):
        evaluate_screen_gate(contaminated, adaa_eligible=True)


def test_screen_can_promote_teacher_rescue_without_relabeling_failure() -> None:
    rows = _screen_rows()
    for row in rows:
        if row["family"] != "a2" and row["loss"] == "m4":
            for metrics in row["snapshots"].values():
                metrics["esr"] = 0.18
            row["selected_validation"]["esr"] = 0.18
    result = evaluate_screen_gate(rows, adaa_eligible=True)
    assert result["passed"] is True
    assert result["deployable_fidelity_passed"] is False


def test_teacher_is_esr_only_model_side_regularization() -> None:
    baselines = {device: {"esr": 0.20} for device in ("fulltone", "bigmuff")}
    rows = [
        {
            "device": device,
            "seed": 0,
            "aa_mode": "teacher_x4",
            "selection_split": "validation",
            "sealed_test_opened": False,
            "physical_target_sample_rate_hz": 48_000,
            "internal_sample_rate_hz": 192_000,
            "teacher_role": "model_side_regularizer_not_hardware_reference",
            "physical_asr_used_for_selection": False,
            "validation": {"esr": 0.16},
        }
        for device in ("fulltone", "bigmuff")
    ]
    result = evaluate_teacher_gate(rows, baselines)
    assert result["passed"] is True
    assert result["supports_physical_hardware_aliasing_claim"] is False


def test_prelisten_and_final_verdict_have_no_physical_asr_gate() -> None:
    mechanism = evaluate_mechanism_gate(_mechanism_rows())
    prelisten = evaluate_prelisten_gate(_confirmation(), _benchmark(), mechanism)
    assert prelisten["passed"] is True
    assert prelisten["physical_asr_gate_present"] is False
    assert validate_benchmark_evidence(_benchmark())["passed"] is True
    assert validate_mushra_design(_mushra_design())["passed"] is True
    result = evaluate_final_verdict(
        _confirmation(),
        _benchmark(),
        mechanism,
        _mushra(),
        instrumentation_valid=True,
    )
    assert result["verdict"] == "GO-R2-48K"
    assert result["physical_asr_used_for_decision"] is False
    assert result["claim_scope"]["global_state_of_the_art"] is False
    weak = deepcopy(_mushra())
    weak["primary_candidate_minus_a2_points"] = 10.0
    assert (
        evaluate_final_verdict(
            _confirmation(),
            _benchmark(),
            mechanism,
            weak,
            instrumentation_valid=True,
        )["verdict"]
        == "NO-GO-R2-48K"
    )


def test_instrumentation_failure_is_invalid() -> None:
    result = evaluate_final_verdict({}, {}, {}, {}, instrumentation_valid=False)
    assert result["verdict"] == "INVALID"
    assert result["valid"] is False
