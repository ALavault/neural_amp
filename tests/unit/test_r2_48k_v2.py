from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from fssr_nam.campaign.r2_48k_v2 import (
    CAMPAIGN_VERSION,
    make_run_id,
    resolve_protocol,
    validate_repository_configs,
)
from fssr_nam.campaign.r2_48k_v2_gates import (
    R248KV2EvidenceError,
    evaluate_mechanism_gate,
    validate_mechanism_evidence,
)
from fssr_nam.campaign.r2_gates import FIXTURES, MECHANISM_MODES
from fssr_nam.reporting.json_evidence import encode_extended_reals

ROOT = Path(__file__).resolve().parents[2]
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_r2_48k_v2_stage", ROOT / "scripts/run_r2_48k_v2_stage.py"
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)


def _mechanism_evidence() -> dict:
    rows = []
    gains = {"off": 0.0, "full_island_x2": 12.0, "adaa1": 11.0, "teacher_x4": 15.0}
    for fixture_index, fixture in enumerate(FIXTURES):
        off_asr = -5.0 + fixture_index
        for mode in MECHANISM_MODES:
            asr = off_asr - gains[mode]
            conditions = [
                {
                    "k0": k0,
                    "amplitude": amplitude,
                    "periodicity_error_db": float("-inf"),
                    "reference_periodicity_error_db": float("-inf"),
                }
                for k0 in (1705, 8191, 12287)
                for amplitude in (0.10, 0.25, 0.48)
            ]
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
                    "internal_sample_rate_hz": 96_000
                    if mode == "full_island_x2"
                    else 48_000,
                    "residual_energy_ratio": 0.10
                    if fixture == "rf2047_residual"
                    else 0.0,
                    "conditions": conditions,
                }
            )
    return encode_extended_reals(
        {
            "campaign_version": CAMPAIGN_VERSION,
            "evidence_encoding": "fssr-extended-real-json-v1",
            "reference_kind": "synthetic_192khz",
            "physical_hardware_reference_used": False,
            "physical_audio_samples_read": 0,
            "parent_numeric_evidence_reused": False,
            "rows": rows,
        }
    )


def test_v2_overlay_changes_only_version_and_evidence_transport() -> None:
    base = yaml.safe_load((ROOT / "configs/r2_48k/protocol.yaml").read_text())
    resolved = resolve_protocol(ROOT)
    assert resolved["campaign_version"] == CAMPAIGN_VERSION
    for section in (
        "claim_scope",
        "baseline",
        "architectures",
        "data",
        "mechanism",
        "screening",
        "teacher_and_distillation",
        "confirmation",
        "physical_asr",
        "synthetic_asr",
        "benchmark",
        "listening",
        "export",
    ):
        assert resolved[section] == base[section]
    assert resolved["version_amendment"]["scientific_thresholds_changed"] is False


def test_v2_repository_contract_and_run_id_are_versioned() -> None:
    resolved = validate_repository_configs(ROOT)
    assert resolved["campaign_version"] == CAMPAIGN_VERSION
    assert make_run_id("mechanism", "synthetic", "analytic", "matrix", 0) == (
        "r2_48k_v2_mechanism_synthetic_analytic_matrix_seed0_v1"
    )


def test_v2_transport_accepts_tagged_exact_zero_before_unchanged_gate() -> None:
    evidence = _mechanism_evidence()
    validated = validate_mechanism_evidence(evidence)
    assert validated["extended_real_diagnostics"] == 48 * 9
    result = evaluate_mechanism_gate(evidence)
    assert result["campaign_continue"] is True
    assert result["scientific_thresholds_changed_from_v1"] is False


def test_v2_transport_rejects_nan_and_nonfinite_decision_aggregate() -> None:
    evidence = _mechanism_evidence()
    evidence["rows"][0]["conditions"][0]["periodicity_error_db"]["value"] = "nan"
    with pytest.raises(R248KV2EvidenceError, match="NaN diagnostics"):
        validate_mechanism_evidence(evidence)
    invalid_decision = _mechanism_evidence()
    invalid_decision["rows"][0]["asr_db"] = deepcopy(
        invalid_decision["rows"][0]["conditions"][0]["periodicity_error_db"]
    )
    with pytest.raises(R248KV2EvidenceError, match="decision field asr_db"):
        validate_mechanism_evidence(invalid_decision)


def test_v2_invalid_gate_terminalization_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_dir = tmp_path / ".codex_campaign/r2_48k_v2"
    summary_dir = tmp_path / "experiments/summaries/r2_48k_v2"
    campaign_dir.mkdir(parents=True)
    summary_dir.mkdir(parents=True)
    (campaign_dir / "MATURITY.json").write_text(
        json.dumps(
            {
                "current_stage": "mechanism",
                "gates": {"mechanism": "measurement_complete_gate_pending"},
                "status": "mechanism_measurement_complete_gate_pending",
            }
        ),
        encoding="utf-8",
    )
    evidence_path = summary_dir / "mechanism.json"
    evidence_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "CAMPAIGN_DIR", campaign_dir)
    monkeypatch.setattr(runner, "SUMMARY_DIR", summary_dir)

    first = runner._terminalize_invalid("synthetic guard failed", evidence_path)
    second = runner._terminalize_invalid("synthetic guard failed", evidence_path)

    assert first == second
    assert json.loads((campaign_dir / "VERDICT.json").read_text())["verdict"] == (
        "INVALID"
    )
    maturity = json.loads((campaign_dir / "MATURITY.json").read_text())
    assert maturity["current_stage"] == "terminal"
    assert maturity["gates"]["mechanism"] == "invalid"
    assert (campaign_dir / "GATE_LEDGER.jsonl").read_text().count("\n") == 1
    monkeypatch.setattr(runner, "validate_repository_configs", lambda _: {})
    monkeypatch.setattr(runner, "validate_stage_authorization", lambda *_: None)
    assert runner.run_mechanism() == first
