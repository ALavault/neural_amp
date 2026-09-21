from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.signal import freqz
from torch import nn

from fssr_nam.campaign.quality_aa_provenance import (
    capture_provenance,
    digest_text,
    replace_json,
    strict_json,
    write_new_json,
)
from fssr_nam.campaign.quality_aa_v2 import (
    CAMPAIGN_VERSION,
    make_run_id,
    validate_repository_configs,
    validate_stage_authorization,
)
from fssr_nam.campaign.quality_aa_v2_gates import evaluate_preflight_gate
from fssr_nam.campaign.quality_aa_v2_registry import (
    append_gate_event,
    gate_decisions,
)
from fssr_nam.data.r2_fixtures import R2_FIXTURES, apply_r2_fixture
from fssr_nam.metrics.quality_aa_mechanism import reference_pair
from fssr_nam.metrics.quality_aliasing import (
    K0_VALUES,
    calibrate_identity_floor,
    coherent_sine_probe,
    fundamental_delay_samples,
    harmonic_fidelity_guard,
)
from fssr_nam.models import FullRateIsland
from fssr_nam.models.oversampling import design_resampling_lowpass
from fssr_nam.reporting.ledger import append_run, read_runs

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / ".codex_campaign/quality_aa_v2"
GLOBAL_LEDGER = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
RUN_ID = make_run_id("preflight")
RUN_DIR = ROOT / "experiments/runs" / RUN_ID
SUMMARY_PATH = ROOT / "experiments/summaries/quality_aa_v2/preflight.json"


class _Identity(nn.Module):
    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return signal

    def stream(self, signal: torch.Tensor) -> torch.Tensor:
        return signal

    def reset_state(self) -> None:
        pass


def _filter_evidence(route: str, factor: int) -> dict:
    taps = factor * 32 + 1
    coefficients = design_resampling_lowpass(factor, taps).numpy()
    frequency, response = freqz(coefficients, worN=262_144, fs=48_000 * factor)

    def magnitude_db(at_hz: float) -> float:
        index = int(np.argmin(np.abs(frequency - at_hz)))
        return float(20.0 * np.log10(max(abs(response[index]), 1.0e-20)))

    torch.manual_seed(20_260_828 + factor)
    signal = torch.randn(2, 257)
    expected = torch.zeros_like(signal)
    expected[:, 32:] = signal[:, :-32]
    island = FullRateIsland(_Identity(), factor=factor, latency_samples=32)
    whole = island(signal)
    island.reset_state()
    streamed = torch.cat(
        [
            island.stream(signal[:, :1]),
            island.stream(signal[:, 1:8]),
            island.stream(signal[:, 8:61]),
            island.stream(signal[:, 61:]),
        ],
        dim=-1,
    )
    identity_error = float(torch.max(torch.abs(whole - expected)))
    block_error = float(torch.max(torch.abs(streamed - expected)))
    passband_db = magnitude_db(20_000.0)
    stopband_db = magnitude_db(30_000.0)
    checks = {
        "taps": len(coefficients) == taps,
        "passband": abs(passband_db) <= 0.01,
        "stopband": stopband_db <= -80.0,
        "identity_delay": identity_error == 0.0,
        "irregular_blocks": block_error == 0.0,
    }
    return {
        "route": route,
        "factor": factor,
        "filter_taps": taps,
        "latency_samples": 32,
        "passband_20khz_db": passband_db,
        "stopband_30khz_db": stopband_db,
        "identity_max_abs_error": identity_error,
        "irregular_block_max_abs_error": block_error,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _reference_convergence(reference_dir: Path) -> dict:
    rows = []
    reference_dir.mkdir(parents=True, exist_ok=False)
    for fixture in R2_FIXTURES:
        print(f"reference_started:{fixture}", flush=True)
        for k0 in K0_VALUES:
            for amplitude in (0.10, 0.25, 0.48):
                x8, x16 = reference_pair(fixture, k0, amplitude)
                guard = harmonic_fidelity_guard(x8, x16, x16, k0=k0)
                values = guard["values"]
                checks = {
                    "harmonic_complex_error": values["candidate_complex_harmonic_error"]
                    <= 1.0e-5,
                    "dc_complex_error": values["dc_complex_error"] <= 1.0e-5,
                    "correlation": values["correlation"] >= 0.999,
                }
                token = str(amplitude).replace(".", "p")
                path = reference_dir / f"{fixture}_k{k0}_a{token}_x16.npy"
                np.save(path, x16, allow_pickle=False)
                rows.append(
                    {
                        "fixture": fixture,
                        "k0": k0,
                        "amplitude": amplitude,
                        "harmonic_complex_error": values[
                            "candidate_complex_harmonic_error"
                        ],
                        "dc_complex_error": values["dc_complex_error"],
                        "correlation": values["correlation"],
                        "checks": checks,
                        "passed": all(checks.values()),
                        "selected_x16_path": str(path.relative_to(ROOT)),
                    }
                )
        print(f"reference_completed:{fixture}", flush=True)
    return {
        "kind": "direct_synthetic_x8_x16",
        "conditions": len(rows),
        "all_conditions_passed": all(row["passed"] for row in rows),
        "maximum_harmonic_complex_error": max(
            row["harmonic_complex_error"] for row in rows
        ),
        "maximum_dc_complex_error": max(row["dc_complex_error"] for row in rows),
        "minimum_correlation": min(row["correlation"] for row in rows),
        "selected_factor": 16,
        "rows": rows,
    }


def _adaa_alignment() -> dict:
    rows = []
    for fixture in R2_FIXTURES:
        for k0 in K0_VALUES:
            signal = coherent_sine_probe(k0, 0.25)
            plain = apply_r2_fixture(fixture, signal, 48_000).output
            adaa = apply_r2_fixture(fixture, signal, 48_000, adaa=True).output
            rows.append(
                {
                    "fixture": fixture,
                    "k0": k0,
                    "delay_samples": fundamental_delay_samples(adaa, plain, k0=k0),
                }
            )
    delays = np.asarray([row["delay_samples"] for row in rows])
    fitted = float(np.median(delays))
    maximum_residual = float(np.max(np.abs(delays - fitted)))
    alignment_passed = 0.0 <= fitted <= 48.0 and maximum_residual <= 0.01
    native_compensation_parity = False
    eligible = alignment_passed and native_compensation_parity
    return {
        "model": "single_causal_global_fractional_delay",
        "fitted_delay_samples": fitted,
        "maximum_global_alignment_residual_samples": maximum_residual,
        "alignment_passed": alignment_passed,
        "native_compensation_parity": native_compensation_parity,
        "eligible": eligible,
        "route_action": (
            "eligible" if eligible else "excluded_without_route_poisoning"
        ),
        "rows": rows,
    }


def main() -> int:
    decisions = gate_decisions(CAMPAIGN_DIR / "GATE_LEDGER.jsonl")
    validate_stage_authorization("preflight", decisions)
    if "preflight" in decisions:
        raise RuntimeError("QUALITY-AA-v2 preflight is already recorded")
    if (
        RUN_DIR.exists()
        or SUMMARY_PATH.exists()
        or any(run["run_id"] == RUN_ID for run in read_runs(GLOBAL_LEDGER))
    ):
        raise RuntimeError("QUALITY-AA-v2 preflight run ID is already reserved")
    protocol = validate_repository_configs(ROOT)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    provenance = capture_provenance(
        ROOT,
        RUN_DIR,
        ["uv", "run", "python", "scripts/campaigns/run_quality_aa_preflight.py"],
    )
    write_new_json(
        RUN_DIR / "status.json",
        {"status": "running", "started_at": started_at, "finished_at": None},
    )
    floor = calibrate_identity_floor()
    filters = {
        "full_island_x2": _filter_evidence("full_island_x2", 2),
        "teacher_x4": _filter_evidence("teacher_x4", 4),
    }
    reference = _reference_convergence(RUN_DIR / "references")
    adaa = _adaa_alignment()
    evidence = {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "candidate_route_outputs_observed": False,
        "physical_audio_samples_read": 0,
        "external_report_only_accessed": False,
        "floor_calibration": floor,
        "filters": filters,
        "reference_convergence": reference,
        "adaa_alignment": adaa,
        "provenance": provenance,
    }
    gate = evaluate_preflight_gate(evidence)
    evidence["gate"] = gate
    write_new_json(RUN_DIR / "preflight.json", evidence)
    write_new_json(SUMMARY_PATH, evidence)
    measurement_lock = {
        "schema_version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "status": "locked_before_candidate_mechanism_render",
        "locked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "locked_floor_db": gate["locked_floor_db"],
        "candidate_route_outputs_observed": False,
        "reference_convergence_passed": gate["checks"]["reference_x8_x16_convergence"],
        "filter_profiles": {
            route: {
                "factor": value["factor"],
                "filter_taps": value["filter_taps"],
                "latency_samples": value["latency_samples"],
            }
            for route, value in filters.items()
        },
        "adaa_status": gate["adaa_status"],
        "preflight_evidence_path": str((RUN_DIR / "preflight.json").relative_to(ROOT)),
    }
    write_new_json(CAMPAIGN_DIR / "MEASUREMENT_LOCK.json", measurement_lock)
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    append_run(
        GLOBAL_LEDGER,
        {
            "date": finished_at,
            "run_id": RUN_ID,
            "phase": "QUALITY-AA-PREFLIGHT",
            "model": "instrumentation_without_candidate_routes",
            "device": "synthetic",
            "seed": 0,
            "commit": provenance["git_head"],
            "config_sha256": digest_text(strict_json(protocol)),
            "data_sha256": digest_text(
                strict_json({"fixtures": list(R2_FIXTURES), "reference": [8, 16]})
            ),
            "status": "completed" if gate["passed"] else "failed",
            "failure_reason": "" if gate["passed"] else "preflight gate failed",
            "results_path": str(RUN_DIR.relative_to(ROOT)),
        },
    )
    replace_json(
        RUN_DIR / "status.json",
        {
            "status": "completed" if gate["passed"] else "failed",
            "started_at": started_at,
            "finished_at": finished_at,
            "failure_reason": "" if gate["passed"] else "preflight gate failed",
        },
    )
    append_gate_event(
        CAMPAIGN_DIR / "GATE_LEDGER.jsonl",
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "preflight",
            "status": "passed" if gate["passed"] else "invalid",
            "evidence_path": str(SUMMARY_PATH.relative_to(ROOT)),
        },
    )
    replace_json(
        CAMPAIGN_DIR / "MATURITY.json",
        {
            "campaign_version": CAMPAIGN_VERSION,
            "current_stage": (
                "measurement_locked" if gate["passed"] else "terminal_invalid"
            ),
            "status": "active" if gate["passed"] else "terminal_invalid",
            "scientific_runs_launched": 1,
            "invalid": not gate["passed"],
        },
    )
    print(json.dumps(gate, allow_nan=False, sort_keys=True), flush=True)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
