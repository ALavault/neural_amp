from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from fssr_nam.campaign.r1 import R1Executor, R1RunReuseError, RunSpec
from fssr_nam.campaign.r1_gates import (
    R1GateEvidenceError,
    evaluate_cascade_physical_gate,
    evaluate_cascade_synthetic_gate,
    evaluate_factorial_gate,
    evaluate_horizon_gate,
    validate_counted_metrics,
)
from fssr_nam.training import r1_diagnostic
from fssr_nam.training.r1 import PhaseScheduler
from fssr_nam.training.r1_diagnostic import (
    ExecutionProfile,
    execution_profile,
    load_m4_dataset,
    load_synthetic_cascade_dataset,
)

ROOT = Path(__file__).resolve().parents[2]


def _yaml(relative: str) -> dict:
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


def _profile() -> ExecutionProfile:
    return ExecutionProfile(
        optimizer_steps=2,
        checkpoint_steps=(1, 2),
        context_samples=6_346,
        output_samples=4_096,
        batch_size=1,
        validation_samples=8_192,
        evaluation_block_samples=1_024,
        preflight=True,
    )


def _metrics(
    stage: str,
    device: str,
    model: str,
    loss: str,
    esr: float,
    *,
    gain_error: float = -0.1,
    residual_ratio: float = 0.05,
    cost: float = 900.0,
) -> dict:
    validation = {
        "esr": esr,
        "gain_error": gain_error,
        "output_energy": 10.0,
        "target_energy": 12.0,
        "residual_energy_ratio": residual_ratio,
    }
    snapshots = {
        str(step): {
            "step": step,
            "validation": dict(validation),
            "gradient_norms_by_block": {"core": 0.25},
            "gradients_finite": True,
        }
        for step in (200, 1_000, 5_000)
    }
    return {
        "schema_version": 1,
        "run_id": f"fixture-{stage}-{device}-{model}-{loss}",
        "stage": stage,
        "device": device,
        "model": model,
        "loss": loss,
        "seed": 0,
        "preflight": False,
        "optimizer_steps": 5_000,
        "checkpoint_steps": [200, 1_000, 5_000],
        "snapshots": snapshots,
        "selected_checkpoint": {"step": 5_000, "validation_esr": esr},
        "best_validation": dict(validation),
        "selected_validation": dict(validation),
        "checks": {
            "all_gradients_finite": True,
            "finite_selected_validation_prediction": True,
            "sealed_test_opened": False,
        },
        "theoretical_macs_per_sample": cost,
    }


def _factorial_rows() -> list[dict]:
    values = {
        ("fulltone", "a2", "m4"): 0.20,
        ("fulltone", "a2", "wright"): 0.20,
        ("fulltone", "s3", "m4"): 0.40,
        ("fulltone", "s3", "wright"): 0.41,
        ("bigmuff", "a2", "m4"): 0.30,
        ("bigmuff", "a2", "wright"): 0.30,
        ("bigmuff", "s3", "m4"): 0.80,
        ("bigmuff", "s3", "wright"): 0.50,
    }
    return [
        _metrics(
            "factorial",
            device,
            model,
            loss,
            esr,
            gain_error=-0.2 if device == "bigmuff" and model == "s3" else -0.1,
        )
        for (device, model, loss), esr in values.items()
    ]


def test_execution_profile_keeps_full_budget_and_exact_phase_lengths() -> None:
    stage = _yaml("configs/training/r1_horizon.yaml")
    m4 = _yaml("configs/training/m4_smoke.yaml")
    profile = execution_profile(stage, m4, preflight=False)
    assert profile.optimizer_steps == 5_000
    assert profile.checkpoint_steps == (200, 1_000, 5_000)
    assert (profile.context_samples, profile.output_samples, profile.batch_size) == (
        6_346,
        8_192,
        2,
    )
    scheduler = PhaseScheduler.from_config(stage["phase_schedule"])
    assert [phase.end_step - phase.start_step + 1 for phase in scheduler.phases] == [
        500,
        1_000,
        2_000,
        1_500,
    ]
    assert scheduler.phase_at(1_501).name == "residual_on_frozen_core_error"


def test_m4_loader_never_opens_the_sealed_test_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / r1_diagnostic.M4_MANIFEST_PATH
    split_path = tmp_path / r1_diagnostic.M4_SPLIT_PATH
    manifest_path.parent.mkdir(parents=True)
    split_path.parent.mkdir(parents=True)
    entries = [
        {
            "device": "fulltone_full_drive_2",
            "split": split,
            "input_path": f"{split}_input.wav",
            "target_path": f"{split}_target.wav",
            "input_sha256": "0" * 64,
            "target_sha256": "1" * 64,
        }
        for split in ("train", "validation", "test")
    ]
    manifest_path.write_text(
        json.dumps({"name": "m4_internal", "tier": "INTERNAL_DEV", "files": entries}),
        encoding="utf-8",
    )
    split_path.write_text(
        json.dumps(
            {
                "dataset": "m4_internal",
                "path_overlap": [],
                "groups": {
                    "fulltone_full_drive_2": {
                        "train": ["a"],
                        "validation": ["b"],
                        "test": ["c"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    opened: list[str] = []

    def fake_pair(root, entry, *, frame_limit):
        del root, frame_limit
        opened.append(entry["split"])
        samples = np.zeros(16_000, dtype=np.float32)
        return samples, samples.copy()

    monkeypatch.setattr(r1_diagnostic, "_read_pair", fake_pair)
    bundle = load_m4_dataset(tmp_path, "fulltone", _profile())
    assert opened == ["train", "validation"]
    assert not hasattr(bundle, "test_input")


def test_synthetic_loader_declares_but_does_not_materialize_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called_seeds: list[int] = []

    def fake_excitation(name, *, sample_rate, duration_seconds, seed):
        del name, sample_rate, duration_seconds
        called_seeds.append(seed)
        return np.linspace(-0.2, 0.2, 48_000, dtype=np.float32)

    monkeypatch.setattr(r1_diagnostic, "generate_excitation", fake_excitation)
    bundle = load_synthetic_cascade_dataset(ROOT, _profile())
    assert called_seeds == [0, 1]
    assert bundle.manifest["splits"]["test"] == {
        "seed": 2,
        "status": "sealed_not_materialized",
    }


def test_factorial_gate_selects_wright_and_applies_all_checks() -> None:
    result = evaluate_factorial_gate(
        _factorial_rows(), _yaml("configs/training/r1_factorial.yaml")
    )
    assert result["passed"] is True
    assert result["selected_loss"] == "wright"
    improvement = result["checks"]["bigmuff_s3_esr_improvement_over_m4"]["value"]
    assert improvement == pytest.approx(0.375)


def test_horizon_and_cascade_gates_use_validation_only() -> None:
    factorial = [row for row in _factorial_rows() if row["loss"] == "wright"]
    horizon = [
        _metrics("horizon", "fulltone", "rf31", "wright", 0.60),
        _metrics("horizon", "fulltone", "rf2047", "wright", 0.35),
        _metrics("horizon", "bigmuff", "rf31", "wright", 0.70, gain_error=-0.8),
        _metrics("horizon", "bigmuff", "rf2047", "wright", 0.55, gain_error=-0.8),
    ]
    horizon_result = evaluate_horizon_gate(
        horizon,
        factorial,
        selected_loss="wright",
        config=_yaml("configs/training/r1_horizon.yaml"),
        promoted_residuals={
            "fulltone": {"run_id": "f", "path": "f", "sha256": "0" * 64},
            "bigmuff": {"run_id": "b", "path": "b", "sha256": "1" * 64},
        },
    )
    assert horizon_result["passed"] is True
    assert horizon_result["devices"]["fulltone"]["gap_closure"] == pytest.approx(0.625)

    synthetic = [
        _metrics("cascade", "synthetic", "mono", "wright", 0.40),
        _metrics("cascade", "synthetic", "cascade", "wright", 0.19),
    ]
    synthetic_result = evaluate_cascade_synthetic_gate(
        synthetic,
        selected_loss="wright",
        config=_yaml("configs/training/r1_cascade.yaml"),
    )
    assert synthetic_result["passed"] is True

    physical = [
        _metrics("cascade", "bigmuff", "cascade", "wright", 0.35, gain_error=-0.3),
        _metrics("cascade", "fulltone", "cascade", "wright", 0.36),
    ]
    baselines = [
        _metrics("horizon", "bigmuff", "rf2047", "wright", 0.55, gain_error=-0.8),
        _metrics("horizon", "fulltone", "rf2047", "wright", 0.35),
    ]
    physical_result = evaluate_cascade_physical_gate(
        physical,
        baselines,
        selected_loss="wright",
        config=_yaml("configs/training/r1_cascade.yaml"),
    )
    assert physical_result["passed"] is True
    assert physical_result["sealed_test_used"] is False


def test_gate_evidence_rejects_missing_common_snapshot() -> None:
    metrics = _metrics("horizon", "fulltone", "rf2047", "wright", 0.4)
    del metrics["snapshots"]["1000"]
    with pytest.raises(R1GateEvidenceError, match="snapshot set"):
        validate_counted_metrics(
            metrics,
            stage="horizon",
            device="fulltone",
            model="rf2047",
            loss="wright",
        )


def test_campaign_records_explicit_gate_stop_immutably(tmp_path: Path) -> None:
    config = _yaml("configs/training/r1_horizon.yaml")
    executor = R1Executor(
        tmp_path,
        stage_configs={"horizon": config},
        gate_decisions={"factorial": {"passed": False, "decision": "failed"}},
    )
    spec = RunSpec("horizon", "fulltone", "rf31", "wright", 0)
    entry = executor.record_gate_stop(
        spec,
        gate="factorial",
        reason="factorial gate failed",
        data_sha256="0" * 64,
        commit="fixture",
    )
    assert entry["status"] == "stopped_by_gate"
    status = json.loads(
        (tmp_path / "experiments/runs" / spec.run_id / "status.json").read_text()
    )
    assert status["status"] == "stopped_by_gate"
    with pytest.raises(R1RunReuseError, match="already"):
        executor.record_gate_stop(
            spec,
            gate="factorial",
            reason="repeat",
            data_sha256="0" * 64,
            commit="fixture",
        )
