from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fssr_nam.campaign.r1_gates import (
    R1GateEvidenceError,
    evaluate_factorial_gate,
    validate_counted_metrics,
)

ROOT = Path(__file__).resolve().parents[2]


def _metrics(device: str, model: str, loss: str, esr: float) -> dict[str, object]:
    validation = {
        "esr": esr,
        "gain_error": 0.0,
        "output_energy": 1.0,
        "target_energy": 1.0,
        "residual_energy_ratio": 0.1,
    }
    snapshots = {
        str(step): {
            "validation": dict(validation),
            "gradients_finite": True,
            "gradient_norms_by_block": {"all": 1.0},
        }
        for step in (200, 1000, 5000)
    }
    return {
        "run_id": f"r1_factorial_{device}_{model}_{loss}_seed0_v1",
        "stage": "factorial",
        "device": device,
        "model": model,
        "loss": loss,
        "seed": 0,
        "preflight": False,
        "optimizer_steps": 5000,
        "checkpoint_steps": [200, 1000, 5000],
        "snapshots": snapshots,
        "selected_checkpoint": {"step": 5000, "validation_esr": esr},
        "best_validation": dict(validation),
        "selected_validation": dict(validation),
        "checks": {
            "all_gradients_finite": True,
            "finite_selected_validation_prediction": True,
            "sealed_test_opened": False,
        },
        "theoretical_macs_per_sample": 100.0,
    }


def test_counted_gate_evidence_rejects_preflight_and_unsealed_test() -> None:
    metrics = _metrics("fulltone", "a2", "m4", 0.1)
    metrics["preflight"] = True
    with pytest.raises(R1GateEvidenceError, match="preflight"):
        validate_counted_metrics(
            metrics, stage="factorial", device="fulltone", model="a2", loss="m4"
        )
    metrics["preflight"] = False
    metrics["checks"]["sealed_test_opened"] = True
    with pytest.raises(R1GateEvidenceError, match="sealed test"):
        validate_counted_metrics(
            metrics, stage="factorial", device="fulltone", model="a2", loss="m4"
        )


def test_factorial_gate_selects_wright_and_applies_all_thresholds() -> None:
    s3_esr = {
        ("fulltone", "m4"): 0.20,
        ("bigmuff", "m4"): 0.40,
        ("fulltone", "wright"): 0.208,
        ("bigmuff", "wright"): 0.28,
    }
    rows = []
    for device in ("fulltone", "bigmuff"):
        for model in ("a2", "s3"):
            for loss in ("m4", "wright"):
                esr = 0.1 if model == "a2" else s3_esr[(device, loss)]
                rows.append(_metrics(device, model, loss, esr))
    config = yaml.safe_load(
        (ROOT / "configs/training/r1_factorial.yaml").read_text(encoding="utf-8")
    )
    decision = evaluate_factorial_gate(rows, config)
    assert decision["passed"] is True
    assert decision["selected_loss"] == "wright"
    assert all(check["passed"] for check in decision["checks"].values())
    assert decision["sealed_test_used"] is False


def test_factorial_gate_fails_closed_on_incomplete_matrix() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/training/r1_factorial.yaml").read_text(encoding="utf-8")
    )
    with pytest.raises(R1GateEvidenceError, match="matrix mismatch"):
        evaluate_factorial_gate([_metrics("fulltone", "a2", "m4", 0.1)], config)
