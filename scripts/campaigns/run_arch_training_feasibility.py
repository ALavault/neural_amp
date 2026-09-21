#!/usr/bin/env python3
"""Measure pre-freeze training feasibility without reading physical audio."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from fssr_nam.campaign.amp_quality_arch_v1 import (
    CAMPAIGN_PATH,
    CANDIDATES,
    validate_repository_state,
)
from fssr_nam.campaign.quality_aa_provenance import write_new_json
from fssr_nam.models import build_arch_v1_candidate

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / CAMPAIGN_PATH / "TRAINING_FEASIBILITY.json"


def _iteration(model: torch.nn.Module, signal: torch.Tensor) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    signal.grad = None
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = model(signal)
    loss = output.square().mean()
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    finite_output = bool(torch.isfinite(output).all().item())
    finite_loss = bool(torch.isfinite(loss).item())
    finite_input_gradient = signal.grad is not None and bool(
        torch.isfinite(signal.grad).all().item()
    )
    finite_parameter_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )
    return {
        "elapsed_seconds": elapsed,
        "finite_output": finite_output,
        "finite_loss": finite_loss,
        "finite_input_gradient": finite_input_gradient,
        "finite_parameter_gradients": finite_parameter_gradients,
    }


def main() -> int:
    if EVIDENCE.exists():
        raise RuntimeError(f"training feasibility evidence already exists: {EVIDENCE}")
    protocol = validate_repository_state(
        ROOT, require_frozen=False, require_native_evidence=False
    )
    gate = protocol["training_feasibility_gate"]
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen CUDA device is unavailable")
    torch.manual_seed(20_260_828)
    torch.cuda.manual_seed_all(20_260_828)
    device = torch.device("cuda")
    results: dict[str, dict[str, Any]] = {}
    for family in CANDIDATES:
        print(f"training_feasibility_started:{family}", flush=True)
        torch.cuda.empty_cache()
        model = build_arch_v1_candidate(
            family, profile=str(gate["selected_profile"])
        ).to(device)
        model.train()
        signal = torch.randn(
            int(gate["batch_size"]),
            int(gate["segment_samples"]),
            device=device,
            requires_grad=True,
        )
        for _ in range(int(gate["warmup_iterations"])):
            warmup = _iteration(model, signal)
            if not all(
                value for key, value in warmup.items() if key != "elapsed_seconds"
            ):
                raise RuntimeError(f"non-finite warmup for {family}: {warmup}")
        torch.cuda.reset_peak_memory_stats()
        measurements = [
            _iteration(model, signal) for _ in range(int(gate["measured_iterations"]))
        ]
        median_seconds = statistics.median(
            float(measurement["elapsed_seconds"]) for measurement in measurements
        )
        projected_hours = median_seconds * int(gate["projected_updates"]) / 3600.0
        finite = all(
            all(value for key, value in measurement.items() if key != "elapsed_seconds")
            for measurement in measurements
        )
        checks = {
            "finite_output_and_gradients": finite,
            "median_seconds_per_update": median_seconds
            <= float(gate["maximum_median_seconds_per_update"]),
            "projected_hours_per_trajectory": projected_hours
            <= float(gate["maximum_projected_hours_per_trajectory"]),
        }
        results[family] = {
            "passed": all(checks.values()),
            "checks": checks,
            "median_seconds_per_update": median_seconds,
            "projected_5000_update_hours": projected_hours,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "measurements": measurements,
        }
        print(
            f"training_feasibility_completed:{family}:"
            f"passed={results[family]['passed']}:median={median_seconds:.6f}",
            flush=True,
        )
        del model, signal
        torch.cuda.empty_cache()
    passed_families = [family for family in CANDIDATES if results[family]["passed"]]
    rejected_families = [
        family for family in CANDIDATES if not results[family]["passed"]
    ]
    report = {
        "schema_version": 1,
        "campaign_version": "AMP-QUALITY-ARCH-v1",
        "stage": "training_feasibility_preflight",
        "status": "passed_with_rejections" if passed_families else "failed",
        "scientific_run": False,
        "input_source": gate["input_source"],
        "physical_audio_samples_read": 0,
        "sealed_outputs_accessed": {"blackstar": False, "ua1176": False},
        "selection_uses_audio_or_esr": False,
        "gate": gate,
        "cuda_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "passed_families": passed_families,
        "rejected_families": rejected_families,
        "results": results,
    }
    write_new_json(EVIDENCE, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if passed_families else 1


if __name__ == "__main__":
    raise SystemExit(main())
