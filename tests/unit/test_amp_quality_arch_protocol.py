from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from fssr_nam.campaign.amp_quality_arch_v1 import (
    CANDIDATES,
    COMPARATORS,
    TEACHER_CANDIDATES,
    TRAINING_CANDIDATES,
    TRAINING_FAMILIES,
    ArchAuthorizationError,
    ArchConfigError,
    loss_qualification_specs,
    make_run_id,
    parse_run_id,
    terminal_verdict,
    validate_distillation_authorization,
    validate_protocol_config,
    validate_sealed_test_boundary,
    validate_stage_authorization,
)

ROOT = Path(__file__).resolve().parents[2]


def _protocol() -> dict[str, object]:
    return yaml.safe_load(
        (ROOT / "configs/amp_quality_arch_v1/protocol.yaml").read_text(encoding="utf-8")
    )


def test_architecture_campaign_lock_is_preserved_and_sealed() -> None:
    protocol = _protocol()
    validate_protocol_config(protocol)
    lock = yaml.safe_load(
        (ROOT / ".codex_campaign/amp_quality_arch_v1/PROTOCOL_LOCK.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert lock == protocol
    assert protocol["campaign_version"] == "AMP-QUALITY-ARCH-v1"
    assert protocol["status"] == "frozen_before_first_scientific_run"
    assert protocol["comparators"]["families"] == list(COMPARATORS)
    assert protocol["candidates"]["frozen_families"] == list(CANDIDATES)
    assert protocol["claim_boundary"]["physical_192khz_dataset_assumed"] is False
    assert protocol["claim_boundary"]["fm9_capture_allowed"] is False


def test_frozen_preflight_reads_metadata_but_no_audio_samples() -> None:
    report = json.loads(
        (ROOT / "experiments/summaries/amp_quality_arch_v1/preflight.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "passed"
    assert report["data"]["audio_samples_read"] == 0
    assert report["loss_qualification_trajectories"] == 36
    assert set(report["candidate_skeletons"]) == set(CANDIDATES)
    assert report["sealed_outputs_accessed"] == {
        "blackstar": False,
        "ua1176": False,
    }


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("runtime_gate", "p95_rtf_maximum"), 0.81, "runtime.rtf"),
        (("final_gate", "devices_won_minimum"), 2, "final.devices"),
        (("resource_budget", "gpu_hours_total_maximum"), 241, "resources.gpu_hours"),
        (("claim_boundary", "physical_192khz_dataset_assumed"), True, "claim.192khz"),
        (("candidates", "maximum_families"), 13, "candidates.maximum"),
    ],
)
def test_decision_bearing_protocol_changes_fail_closed(
    path: tuple[str, str], value: object, message: str
) -> None:
    protocol = deepcopy(_protocol())
    protocol[path[0]][path[1]] = value
    with pytest.raises(ArchConfigError, match=message):
        validate_protocol_config(protocol)


def test_target_derived_features_cannot_enter_inference() -> None:
    protocol = deepcopy(_protocol())
    protocol["physical_state_bus"]["inference_inputs"].append("wet_fast_envelope")
    with pytest.raises(ArchConfigError, match="wet descriptors"):
        validate_protocol_config(protocol)


def test_loss_qualification_matrix_is_exact_and_bounded() -> None:
    protocol = _protocol()
    specs = loss_qualification_specs(protocol)
    assert len(specs) == 36
    assert len({spec.run_id for spec in specs}) == 36
    assert {spec.device for spec in specs} == {"fulltone", "bigmuff"}
    for family in TRAINING_FAMILIES:
        family_specs = [spec for spec in specs if spec.family == family]
        assert len(family_specs) == 4
        assert {spec.loss for spec in family_specs} == set(
            protocol["losses"]["per_family_allowed_pairs"][family]
        )


@st.composite
def valid_run_components(draw: st.DrawFn) -> tuple[str, str, str, str, int]:
    stage = draw(
        st.sampled_from(
            [
                "preflight",
                "mechanism",
                "loss_qualify",
                "screen",
                "teacher",
                "distill",
                "robustness",
                "lock",
                "native",
                "confirm_train",
                "confirm_test",
                "listen",
                "audit",
            ]
        )
    )
    if stage == "preflight":
        return stage, "all", "protocol", "none", 0
    if stage == "mechanism":
        return (
            stage,
            "synthetic",
            draw(st.sampled_from(TRAINING_CANDIDATES)),
            "none",
            0,
        )
    if stage in {"loss_qualify", "screen"}:
        return (
            stage,
            draw(st.sampled_from(["fulltone", "bigmuff"])),
            draw(st.sampled_from(TRAINING_FAMILIES)),
            draw(st.sampled_from(["m4", "wright", "nablafx"])),
            0,
        )
    if stage in {"teacher", "distill"}:
        return (
            stage,
            draw(st.sampled_from(["fulltone", "bigmuff"])),
            draw(st.sampled_from(TEACHER_CANDIDATES)),
            draw(st.sampled_from(["m4", "wright", "nablafx"])),
            0,
        )
    if stage == "robustness":
        return (
            stage,
            draw(st.sampled_from(["fulltone", "bigmuff"])),
            draw(st.sampled_from(TRAINING_FAMILIES)),
            draw(st.sampled_from(["m4", "wright", "nablafx"])),
            draw(st.integers(min_value=1, max_value=4)),
        )
    if stage == "lock":
        return stage, "all", "all", "none", 0
    if stage == "native":
        return (
            stage,
            "host_cpu",
            draw(st.sampled_from(TRAINING_FAMILIES)),
            "none",
            0,
        )
    if stage == "confirm_train":
        return (
            stage,
            draw(st.sampled_from(["blackstar", "ua1176"])),
            draw(st.sampled_from(TRAINING_FAMILIES)),
            draw(st.sampled_from(["m4", "wright", "nablafx"])),
            draw(st.integers(min_value=0, max_value=4)),
        )
    return stage, "all", "all", "none", 0


@given(components=valid_run_components())
@settings(max_examples=60, deadline=None)
def test_run_identifier_roundtrip_is_canonical(
    components: tuple[str, str, str, str, int],
) -> None:
    run_id = make_run_id(*components)
    parsed = parse_run_id(run_id)
    assert (parsed.stage, parsed.device, parsed.family, parsed.loss, parsed.seed) == (
        components
    )
    assert parsed.run_id == run_id


@pytest.mark.parametrize(
    "run_id",
    [
        "arch_v1_screen_bigmuff_micro_tcn_x2_wright_seed00_v1",
        "arch_v1_screen_bigmuff_micro-tcn-x2_wright_seed0_v1",
        "arch_v1_screen_bigmuff_micro_tcn_x2_wright_seed0_v2",
        "arch_v1_screen_bigmuff_micro_tcn_x2_wright_seed0_retry_v1",
        "arch_v1_confirm_train_fulltone_micro_tcn_x2_wright_seed0_v1",
    ],
)
def test_aliases_retries_and_invalid_stage_shapes_are_rejected(run_id: str) -> None:
    with pytest.raises(ValueError):
        parse_run_id(run_id)


def test_sequential_stage_and_distillation_gates_are_literal() -> None:
    validate_stage_authorization("preflight", {})
    with pytest.raises(ArchAuthorizationError, match="preflight=passed"):
        validate_stage_authorization("mechanism", {"preflight": "failed"})
    validate_stage_authorization("mechanism", {"preflight": "passed"})
    with pytest.raises(ArchAuthorizationError, match="teacher=passed"):
        validate_distillation_authorization({"teacher": "failed"})
    with pytest.raises(ArchAuthorizationError, match="failed deployable"):
        validate_distillation_authorization(
            {"teacher": "passed", "deployable": "passed"}
        )
    validate_distillation_authorization(
        {"teacher": "passed", "deployable": "failed_fidelity_gate"}
    )


def test_sealed_boundary_requires_lock_parity_benchmark_and_zero_openings() -> None:
    decisions = {
        "lock": "passed",
        "python_cpp_parity": "passed",
        "benchmark": "passed",
        "confirm_train": "passed",
    }
    validate_sealed_test_boundary(
        decisions=decisions,
        blackstar_open_count=0,
        ua1176_open_count=0,
        external_report_only_locked=True,
    )
    with pytest.raises(ArchAuthorizationError, match="only once"):
        validate_sealed_test_boundary(
            decisions=decisions,
            blackstar_open_count=1,
            ua1176_open_count=0,
            external_report_only_locked=True,
        )
    with pytest.raises(ArchAuthorizationError, match="benchmark=passed"):
        validate_sealed_test_boundary(
            decisions={**decisions, "benchmark": "failed"},
            blackstar_open_count=0,
            ua1176_open_count=0,
            external_report_only_locked=True,
        )


def test_terminal_verdict_is_fail_closed_and_simultaneous() -> None:
    gates = {
        "esr": True,
        "device_wins": True,
        "asr": True,
        "runtime": True,
        "listening": True,
        "tests": True,
        "lint": True,
    }
    assert terminal_verdict(valid=True, gates=gates) == "GO-ARCH"
    assert (
        terminal_verdict(valid=True, gates={**gates, "runtime": False}) == "NO-GO-ARCH"
    )
    assert terminal_verdict(valid=False, gates=gates) == "INVALID"
    with pytest.raises(ValueError, match="exactly"):
        terminal_verdict(valid=True, gates={"esr": True})
