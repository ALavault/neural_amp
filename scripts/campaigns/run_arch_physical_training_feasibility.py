#!/usr/bin/env python3
"""Benchmark the frozen physical-training shapes without reading audio."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from fssr_nam.campaign.amp_quality_arch_v1 import (
    CAMPAIGN_PATH,
    PROTOCOL_PATH,
    TRAINING_CANDIDATES,
    validate_protocol_config,
)
from fssr_nam.campaign.quality_aa_provenance import write_new_json
from fssr_nam.losses import NablafxLoss
from fssr_nam.models import build_arch_v1_candidate
from fssr_nam.models.sota_comparators import build_sota_comparator

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / CAMPAIGN_PATH / "PHYSICAL_TRAINING_FEASIBILITY.json"


def _workloads(config: dict[str, Any]) -> tuple[tuple[str, str, int], ...]:
    contexts = config["context_samples"]
    output = int(config["output_samples_per_update"])
    rows = [
        ("nam_a2_full", "official", contexts["nam_a2_full"]["official"]),
        ("wright_lstm64", "official", contexts["wright_lstm64"]["official"]),
    ]
    for family in ("nablafx_tcn_tfilm", "nablafx_s4_tfilm"):
        for variant in ("small", "large"):
            rows.append((family, variant, contexts[family][variant]))
    rows.extend(
        (family, "max", int(contexts["candidates"])) for family in TRAINING_CANDIDATES
    )
    return tuple(
        (family, variant, int(context) + output) for family, variant, context in rows
    )


def _build(family: str, variant: str) -> torch.nn.Module:
    if family in TRAINING_CANDIDATES:
        return build_arch_v1_candidate(family, profile=variant)
    return build_sota_comparator(family, variant=variant, root=ROOT)


def _prediction(
    model: torch.nn.Module, family: str, signal: torch.Tensor
) -> torch.Tensor:
    if family == "nam_a2_full":
        output = model(signal, pad_start=False)
    else:
        output = model(signal)
    return output[..., -8_192:]


def _iteration(
    model: torch.nn.Module,
    family: str,
    signal: torch.Tensor,
    target: torch.Tensor,
    loss_function: NablafxLoss,
) -> dict[str, float | bool]:
    model.zero_grad(set_to_none=True)
    signal.grad = None
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = _prediction(model, family, signal)
    loss = loss_function(output, target)
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )
    return {
        "elapsed_seconds": elapsed,
        "finite_output": bool(torch.isfinite(output).all().item()),
        "finite_loss": bool(torch.isfinite(loss).item()),
        "finite_input_gradient": signal.grad is not None
        and bool(torch.isfinite(signal.grad).all().item()),
        "finite_parameter_gradients": gradients_finite,
    }


def main() -> int:
    if EVIDENCE.exists():
        raise RuntimeError(f"physical feasibility evidence already exists: {EVIDENCE}")
    protocol = yaml.safe_load((ROOT / PROTOCOL_PATH).read_text(encoding="utf-8"))
    validate_protocol_config(protocol)
    config = protocol["physical_training"]
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen CUDA device is unavailable")
    device = torch.device("cuda")
    results: dict[str, dict[str, Any]] = {}
    for index, (family, variant, segment_samples) in enumerate(_workloads(config)):
        workload = f"{family}:{variant}"
        print(f"physical_training_feasibility_started:{workload}", flush=True)
        torch.manual_seed(20_260_828 + index)
        torch.cuda.manual_seed_all(20_260_828 + index)
        torch.cuda.empty_cache()
        model = _build(family, variant).to(device).train()
        loss_function = NablafxLoss().to(device)
        signal = torch.randn(
            int(config["batch_size"]),
            segment_samples,
            device=device,
            requires_grad=True,
        )
        target = torch.randn(
            int(config["batch_size"]),
            int(config["output_samples_per_update"]),
            device=device,
        )
        for _ in range(int(config["feasibility_warmup_iterations"])):
            warmup = _iteration(model, family, signal, target, loss_function)
            if not all(
                value for key, value in warmup.items() if key != "elapsed_seconds"
            ):
                raise RuntimeError(f"non-finite warmup for {workload}: {warmup}")
        torch.cuda.reset_peak_memory_stats()
        measurements = [
            _iteration(model, family, signal, target, loss_function)
            for _ in range(int(config["feasibility_measured_iterations"]))
        ]
        median_seconds = statistics.median(
            float(measurement["elapsed_seconds"]) for measurement in measurements
        )
        projected_hours = median_seconds * 5_000 / 3_600.0
        finite = all(
            all(value for key, value in row.items() if key != "elapsed_seconds")
            for row in measurements
        )
        checks = {
            "finite_output_loss_and_gradients": finite,
            "median_seconds_per_update": median_seconds
            <= float(config["maximum_median_seconds_per_update"]),
            "projected_hours_per_5000_updates": projected_hours
            <= float(config["maximum_projected_hours_per_5000_update_trajectory"]),
        }
        results[workload] = {
            "passed": all(checks.values()),
            "checks": checks,
            "family": family,
            "variant": variant,
            "segment_samples": segment_samples,
            "output_samples": int(config["output_samples_per_update"]),
            "median_seconds_per_update": median_seconds,
            "projected_5000_update_hours": projected_hours,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "measurements": measurements,
        }
        print(
            f"physical_training_feasibility_completed:{workload}:"
            f"passed={results[workload]['passed']}:median={median_seconds:.6f}",
            flush=True,
        )
        del model, loss_function, signal, target
        torch.cuda.empty_cache()
    passed = all(result["passed"] for result in results.values())
    report = {
        "schema_version": 1,
        "campaign_version": "AMP-QUALITY-ARCH-v1",
        "stage": "physical_training_feasibility_preflight",
        "status": "passed" if passed else "failed",
        "scientific_run": False,
        "input_source": "deterministic_synthetic_probe",
        "physical_audio_samples_read": 0,
        "sealed_outputs_accessed": {"blackstar": False, "ua1176": False},
        "selection_uses_audio_or_esr": False,
        "loss_cost_probe": "exact_nablafx_0.5_l1_plus_0.5_mrstft",
        "gate": config,
        "cuda_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "workload_count": len(results),
        "results": results,
    }
    write_new_json(EVIDENCE, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
