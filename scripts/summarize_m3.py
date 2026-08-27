#!/usr/bin/env python3
"""Aggregate M3 evidence and evaluate the implementation gate."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from fssr_nam.data.systems import apply_system
from fssr_nam.dsp.multirate import derive_reference_rates
from fssr_nam.metrics.nonlinear import (
    complex_harmonic_error,
    known_reference_parasite_db,
)
from fssr_nam.models import LocalOversampledSpline2x
from fssr_nam.reporting.provenance import git_state

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments/summaries/m3_validation"
RUNS = {
    "s0_tanh": "m3_s0_tanh_seed0_v1",
    "s3_tanh": "m3_s3_tanh_seed0_v1",
    "s4_tanh": "m3_s4_tanh_seed0_v1",
    "s0_slow": "m3_s0_slow_sag_seed0_v1",
    "s1_slow": "m3_s1_slow_sag_seed0_v1",
    "s3_slow": "m3_s3_slow_sag_seed0_v1",
    "s0_short": "m3_s0_short_memory_seed0_v1",
    "s2_short": "m3_s2_short_memory_seed0_v1",
}


def load_metrics(run_id: str) -> dict:
    path = ROOT / "experiments/runs" / run_id / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def improvement(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline


def controlled_alias_result() -> dict[str, float]:
    master_rate = 192_000
    index = np.arange(master_rate, dtype=np.float64)
    master_input = 0.48 * np.sin(2.0 * np.pi * 9_000.0 * index / master_rate)
    reference = derive_reference_rates(apply_system("tanh", master_input, master_rate))[
        48_000
    ].astype(np.float32)
    low_rate_input = derive_reference_rates(master_input)[48_000].astype(np.float32)
    naive = apply_system("tanh", low_rate_input, 48_000)
    shaper = LocalOversampledSpline2x(filter_taps=33)
    with torch.no_grad():
        denominator = torch.tanh(torch.tensor(2.8))
        shaper.spline.values.copy_(torch.tanh(2.8 * shaper.spline.knots) / denominator)
        shaper.spline.slopes.copy_(
            2.8 * (1.0 - torch.tanh(2.8 * shaper.spline.knots).square()) / denominator
        )
        antialiased = shaper(torch.from_numpy(low_rate_input)).numpy()
    latency = shaper.latency_samples
    delayed_reference = np.pad(reference, (latency, 0))[:-latency]
    delayed_naive = np.pad(naive, (latency, 0))[:-latency]
    stable = slice(512, None)
    naive_db = known_reference_parasite_db(
        delayed_naive[stable], delayed_reference[stable]
    )
    antialiased_db = known_reference_parasite_db(
        antialiased[stable], delayed_reference[stable]
    )
    return {
        "naive_known_reference_parasite_db": naive_db,
        "s4_known_reference_parasite_db": antialiased_db,
        "reduction_db": naive_db - antialiased_db,
        "s4_complex_fundamental_error": complex_harmonic_error(
            antialiased[stable],
            delayed_reference[stable],
            fundamental_hz=9_000.0,
            sample_rate=48_000,
        ),
    }


def main() -> None:
    metrics = {name: load_metrics(run_id) for name, run_id in RUNS.items()}
    final_esr = {name: value["final"]["esr"] for name, value in metrics.items()}
    s1_gain = improvement(final_esr["s0_slow"], final_esr["s1_slow"])
    s2_gain = improvement(final_esr["s0_short"], final_esr["s2_short"])
    residual_ratios = {
        name: value["final"]["residual_energy_ratio"]
        for name, value in metrics.items()
        if name.startswith(("s2_", "s3_", "s4_"))
    }
    reload_and_stream_checks = {
        name: max(
            value["checks"]["reload_max_abs"],
            value["checks"]["irregular_stream_max_abs"],
            value["checks"]["reset_max_abs"],
            value["checks"]["causal_prefix_max_abs"],
        )
        for name, value in metrics.items()
    }
    theoretical = {
        "S0": {
            "parameters": metrics["s0_tanh"]["parameters"],
            "linear_macs_per_sample": 34,
        },
        "S1": {
            "parameters": metrics["s1_slow"]["parameters"],
            "linear_macs_per_sample_approx": 38.5,
        },
        "S2": {
            "parameters": metrics["s2_short"]["parameters"],
            "linear_macs_per_sample_approx": 826,
        },
        "S3": {
            "parameters": metrics["s3_tanh"]["parameters"],
            "linear_macs_per_sample_approx": 830.5,
        },
        "S4": {
            "parameters": metrics["s4_tanh"]["parameters"],
            "linear_macs_per_sample_approx": 962.5,
            "declared_latency_samples": metrics["s4_tanh"]["latency_samples"],
        },
        "A2_Full": {
            "parameters": 12_145,
            "linear_macs_per_sample": 11_777,
        },
    }
    alias = controlled_alias_result()
    checks = {
        "s3_learns_tanh": final_esr["s3_tanh"] < 1.0e-3,
        "s3_learns_slow_sag": final_esr["s3_slow"] < 1.0e-3,
        "s1_improves_slow_sag": s1_gain > 0.05,
        "s2_improves_short_memory": s2_gain > 0.05,
        "residual_not_dominant": max(residual_ratios.values()) < 0.1,
        "trained_reload_stream_reset_causal": max(reload_and_stream_checks.values())
        < 2.0e-5,
        "s4_declares_expected_latency": metrics["s4_tanh"]["latency_samples"] == 16,
        "s3_theoretical_cost_below_a2": theoretical["S3"][
            "linear_macs_per_sample_approx"
        ]
        < theoretical["A2_Full"]["linear_macs_per_sample"],
        "s4_reduces_controlled_alias": alias["reduction_db"] > 3.0,
        "s4_preserves_controlled_fundamental": alias["s4_complex_fundamental_error"]
        < 1.0e-5,
    }
    result = {
        "schema_version": 1,
        "provenance": git_state(
            ignored_generated_paths=("experiments/summaries/m3_validation",)
        ),
        "runs": RUNS,
        "final_esr": final_esr,
        "s1_relative_improvement_over_s0_slow_sag": s1_gain,
        "s2_relative_improvement_over_s0_short_memory": s2_gain,
        "residual_energy_ratios": residual_ratios,
        "maximum_reload_stream_reset_causal_error": reload_and_stream_checks,
        "theoretical_cost": theoretical,
        "controlled_alias_9khz": alias,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "limitations": [
            "All M3 optimization evidence uses one seed and synthetic targets.",
            (
                "MAC counts exclude activations, additions, memory traffic, "
                "and Python overhead."
            ),
            (
                "S4's controlled alias result remains a synthetic diagnostic, "
                "not H3 validation."
            ),
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(("S0", "S1"), (final_esr["s0_slow"], final_esr["s1_slow"]))
    axes[0].set_title("slow_sag")
    axes[0].set_ylabel("validation ESR")
    axes[1].bar(("S0", "S2"), (final_esr["s0_short"], final_esr["s2_short"]))
    axes[1].set_title("short nonlinear memory")
    figure.tight_layout()
    figure.savefig(OUT_DIR / "branch_ablation.png", dpi=140)
    plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
