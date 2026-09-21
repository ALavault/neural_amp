#!/usr/bin/env python3
"""Run the blind native capacity gate for AMP-QUALITY-ARCH-v1."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from fssr_nam.campaign.amp_quality_arch_v1 import (
    CAMPAIGN_PATH,
    CANDIDATES,
    validate_repository_state,
)
from fssr_nam.campaign.quality_aa_provenance import write_new_json
from fssr_nam.models import DEPLOYMENT_PROFILES, build_arch_v1_candidate

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/amp_arch_skeleton"
EVIDENCE = ROOT / CAMPAIGN_PATH / "NATIVE_SKELETON.json"


def _run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _build() -> tuple[Path, Path]:
    _run(
        [
            "cmake",
            "-S",
            str(ROOT / "cpp"),
            "-B",
            str(BUILD),
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )
    _run(
        [
            "cmake",
            "--build",
            str(BUILD),
            "--target",
            "amp_arch_skeleton_runner",
            "amp_arch_skeleton_benchmark",
            "--parallel",
            "8",
        ]
    )
    return BUILD / "amp_arch_skeleton_runner", BUILD / "amp_arch_skeleton_benchmark"


def _validate_native_checks(runner: Path) -> dict[str, dict[str, dict[str, Any]]]:
    reports: dict[str, dict[str, dict[str, Any]]] = {}
    for family in CANDIDATES:
        reports[family] = {}
        for profile in DEPLOYMENT_PROFILES:
            report = json.loads(_run([str(runner), family, profile]).stdout)
            python_parameters = sum(
                parameter.numel()
                for parameter in build_arch_v1_candidate(
                    family, profile=profile
                ).parameters()
            )
            checks = {
                "finite": report.get("finite") is True,
                "block_parity": report.get("block_max_abs_error", 1.0) <= 2.0e-6,
                "reset": report.get("reset_max_abs_error", 1.0) <= 2.0e-6,
                "zero_audio_loop_allocations": report.get("audio_loop_allocations")
                == 0,
                "latency": report.get("latency_samples") == 32,
                "python_parameter_count": report.get("parameters") == python_parameters,
            }
            if not all(checks.values()):
                raise RuntimeError(
                    f"native skeleton check failed for {family}/{profile}: {checks}"
                )
            reports[family][profile] = {**report, "checks": checks}
    return reports


def _validate_benchmark(
    benchmark: dict[str, Any],
    native_checks: dict[str, dict[str, dict[str, Any]]],
    maximum_rtf: float,
) -> tuple[dict[str, str], dict[str, dict[str, dict[str, Any]]]]:
    required_metadata = {
        "format": "fssr-amp-arch-native-skeleton-benchmark-v1",
        "campaign_version": "AMP-QUALITY-ARCH-v1",
        "blind": True,
        "selection_uses_audio_or_esr": False,
        "trained_weights": False,
        "same_binary": True,
        "abi": "float32",
        "compiler_optimization": "-Ofast",
        "lto_ipo": True,
        "native_isa": True,
        "schedule": "interleaved_ab",
        "repetitions": 30,
        "scientific_eligible": True,
        "block_size": 64,
        "sample_rate_hz": 48_000,
    }
    for key, expected in required_metadata.items():
        if benchmark.get(key) != expected:
            raise RuntimeError(
                f"native skeleton benchmark metadata mismatch: {key}="
                f"{benchmark.get(key)!r}, expected {expected!r}"
            )
    expected_pairs = [
        (family, profile) for family in CANDIDATES for profile in DEPLOYMENT_PROFILES
    ]
    rows = benchmark.get("results")
    if (
        not isinstance(rows, list)
        or [(row.get("family"), row.get("profile")) for row in rows] != expected_pairs
    ):
        raise RuntimeError("native skeleton benchmark family/profile matrix drifted")
    evaluated: dict[str, dict[str, dict[str, Any]]] = {
        family: {} for family in CANDIDATES
    }
    for row in rows:
        family = str(row["family"])
        profile = str(row["profile"])
        check = native_checks[family][profile]
        p95_rtf = float(row["candidate"]["p95_rtf"])
        gates = {
            "native_checks": all(check["checks"].values()),
            "block64_p95_rtf": p95_rtf <= maximum_rtf,
            "latency": int(row["latency_samples"]) <= 48,
            "parameter_count": row["parameters"] == check["parameters"],
        }
        evaluated[family][profile] = {
            "passed": all(gates.values()),
            "gates": gates,
            "p95_rtf": p95_rtf,
            "median_rtf": float(row["candidate"]["median_rtf"]),
            "a2_p95_rtf": float(row["a2"]["p95_rtf"]),
            "parameters": int(row["parameters"]),
            "weight_bytes": int(row["weight_bytes"]),
            "persistent_state_bytes": int(row["persistent_state_bytes"]),
            "scratch_bytes": int(row["scratch_bytes"]),
            "latency_samples": int(row["latency_samples"]),
            "estimated_macs_per_sample": int(row["estimated_macs_per_sample"]),
        }
    selected: dict[str, str] = {}
    reverse_profiles = tuple(reversed(tuple(DEPLOYMENT_PROFILES)))
    for family in CANDIDATES:
        for profile in reverse_profiles:
            if evaluated[family][profile]["passed"]:
                selected[family] = profile
                break
    return selected, evaluated


def main() -> int:
    if EVIDENCE.exists():
        raise RuntimeError(f"native skeleton evidence already exists: {EVIDENCE}")
    protocol = validate_repository_state(
        ROOT, require_frozen=False, require_native_evidence=False
    )
    baseline = protocol["native_skeleton_baseline"]
    model_path = ROOT / baseline["model_path"]
    if not model_path.is_file():
        raise RuntimeError(f"missing pinned A2 native baseline: {model_path}")
    cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    for feature in baseline["required_isa"]:
        if feature not in cpuinfo.split("flags", maxsplit=1)[-1].split():
            raise RuntimeError(f"required native ISA feature is unavailable: {feature}")
    compiler = _run(["c++", "--version"]).stdout.splitlines()[0]
    if "15.2.0" not in compiler:
        raise RuntimeError(f"frozen GNU 15.2.0 compiler is unavailable: {compiler}")
    runner, benchmark_executable = _build()
    native_checks = _validate_native_checks(runner)
    print("native_skeleton_benchmark_started:24_profiles_x30_repetitions", flush=True)
    benchmark = json.loads(
        _run([str(benchmark_executable), str(model_path), "1", "30"]).stdout
    )
    maximum_rtf = float(
        protocol["mechanism_gate"]["native_skeleton_block64_p95_rtf_maximum"]
    )
    selected, evaluated = _validate_benchmark(benchmark, native_checks, maximum_rtf)
    passed = set(selected) == set(CANDIDATES)
    report = {
        "schema_version": 1,
        "campaign_version": "AMP-QUALITY-ARCH-v1",
        "stage": "native_skeleton_preflight",
        "status": "passed" if passed else "failed",
        "scientific_run": False,
        "physical_audio_samples_read": 0,
        "sealed_outputs_accessed": {"blackstar": False, "ua1176": False},
        "selection_uses_audio_or_esr": False,
        "selection_rule": "largest_profile_passing_blind_native_skeleton_gate",
        "maximum_block64_p95_rtf": maximum_rtf,
        "baseline": baseline,
        "compiler": compiler,
        "selected_profiles": selected,
        "profiles": evaluated,
        "native_checks": native_checks,
        "benchmark": benchmark,
    }
    write_new_json(EVIDENCE, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
