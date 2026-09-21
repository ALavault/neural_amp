from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

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
from fssr_nam.campaign.quality_aa_v2_gates import (
    evaluate_mechanism_gate,
    evaluate_native_gate,
    final_backend_decision,
)
from fssr_nam.campaign.quality_aa_v2_registry import (
    append_gate_event,
    gate_decisions,
)
from fssr_nam.inference import export_r2_native_model, verify_r2_cpp_parity
from fssr_nam.models import AAFSSR
from fssr_nam.reporting.ledger import append_run, read_runs

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / ".codex_campaign/quality_aa_v2"
GLOBAL_LEDGER = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
RUN_ID = make_run_id("native")
RUN_DIR = ROOT / "experiments/runs" / RUN_ID
SUMMARY_PATH = ROOT / "experiments/summaries/quality_aa_v2/native.json"
MECHANISM_PATH = ROOT / "experiments/summaries/quality_aa_v2/mechanism.json"
BUILD_DIR = ROOT / "build/quality_aa_v2_native"


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return value


def _make_nontrivial(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for parameter_index, parameter in enumerate(model.parameters()):
            positions = torch.arange(parameter.numel(), dtype=parameter.dtype)
            values = 0.02 * torch.sin(0.19 * positions + 0.13 * parameter_index)
            parameter.copy_(values.reshape_as(parameter))


def _build_tools() -> tuple[Path, Path]:
    subprocess.run(
        [
            "cmake",
            "-S",
            str(ROOT / "cpp"),
            "-B",
            str(BUILD_DIR),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(BUILD_DIR),
            "--target",
            "r2_block_runner",
            "r2_benchmark",
            "--parallel",
            "4",
        ],
        cwd=ROOT,
        check=True,
    )
    return BUILD_DIR / "r2_block_runner", BUILD_DIR / "r2_benchmark"


def _benchmark(
    executable: Path,
    model_path: Path,
    parity_error: float,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    benchmark = protocol["benchmark"]
    completed = subprocess.run(
        [
            str(executable),
            str(ROOT / benchmark["a2_model"]),
            str(model_path),
            str(benchmark["a2_parameters"]),
            str(benchmark["a2_weight_bytes"]),
            str(benchmark["a2_persistent_state_bytes"]),
            str(benchmark["a2_scratch_bytes"]),
            f"{parity_error:.17g}",
            "1",
            str(benchmark["repetitions"]),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    if not isinstance(report, dict):
        raise RuntimeError("native benchmark did not return an object")
    return report


def _record_invalid(
    *,
    reason: str,
    started_at: str,
    provenance: dict[str, Any],
    protocol: dict[str, Any],
) -> int:
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    failure = {
        "campaign_version": CAMPAIGN_VERSION,
        "run_id": RUN_ID,
        "status": "INVALID",
        "failure_reason": reason,
        "physical_audio_samples_read": 0,
    }
    write_new_json(RUN_DIR / "failure.json", failure)
    write_new_json(CAMPAIGN_DIR / "VERDICT.json", failure)
    append_run(
        GLOBAL_LEDGER,
        {
            "date": finished_at,
            "run_id": RUN_ID,
            "phase": "QUALITY-AA-NATIVE",
            "model": "latency32_x2_x4",
            "device": "host_cpu",
            "seed": 0,
            "commit": provenance["git_head"],
            "config_sha256": digest_text(strict_json(protocol)),
            "data_sha256": digest_text("deterministic_native_parity_signal_v1"),
            "status": "failed",
            "failure_reason": reason,
            "results_path": str(RUN_DIR.relative_to(ROOT)),
        },
    )
    replace_json(
        RUN_DIR / "status.json",
        {
            "status": "failed",
            "started_at": started_at,
            "finished_at": finished_at,
            "failure_reason": reason,
        },
    )
    append_gate_event(
        CAMPAIGN_DIR / "GATE_LEDGER.jsonl",
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "native",
            "status": "invalid",
            "evidence_path": str((RUN_DIR / "failure.json").relative_to(ROOT)),
        },
    )
    replace_json(
        CAMPAIGN_DIR / "MATURITY.json",
        {
            "campaign_version": CAMPAIGN_VERSION,
            "current_stage": "terminal_invalid",
            "status": "terminal_invalid",
            "scientific_runs_launched": 3,
            "invalid": True,
        },
    )
    return 1


def main() -> int:
    decisions = gate_decisions(CAMPAIGN_DIR / "GATE_LEDGER.jsonl")
    validate_stage_authorization("native", decisions)
    if "native" in decisions:
        raise RuntimeError("QUALITY-AA-v2 native stage is already recorded")
    if (
        RUN_DIR.exists()
        or SUMMARY_PATH.exists()
        or any(run["run_id"] == RUN_ID for run in read_runs(GLOBAL_LEDGER))
    ):
        raise RuntimeError("QUALITY-AA-v2 native run ID is already reserved")
    protocol = validate_repository_configs(ROOT)
    mechanism = _load_json(MECHANISM_PATH, "mechanism evidence")
    mechanism_gate = evaluate_mechanism_gate(mechanism)
    routes = list(mechanism_gate["passing_routes"])
    if not routes:
        raise RuntimeError("native stage requires at least one mechanism route")
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    provenance = capture_provenance(
        ROOT,
        RUN_DIR,
        ["uv", "run", "python", "scripts/campaigns/run_quality_aa_native.py"],
    )
    write_new_json(
        RUN_DIR / "status.json",
        {"status": "running", "started_at": started_at, "finished_at": None},
    )
    try:
        runner, benchmark_executable = _build_tools()
        route_evidence = {}
        for route in routes:
            print(f"native_started:{route}", flush=True)
            torch.manual_seed(20_260_828)
            model = AAFSSR(
                core_kind="cascade",
                aa_mode=route,
                aa_latency_samples=32,
            )
            _make_nontrivial(model)
            model_path = export_r2_native_model(
                model, RUN_DIR / "models" / f"{route}.json"
            )
            signal = torch.linspace(-0.6, 0.7, 257) + 0.04 * torch.sin(
                torch.arange(257) * 0.21
            )
            parity = verify_r2_cpp_parity(
                model_path,
                runner,
                signal.numpy(),
                irregular_blocks=(1, 7, 3, 31, 5, 64),
                report_path=RUN_DIR / f"parity-{route}.json",
            )
            error = parity.get("max_abs_error")
            if not isinstance(error, (int, float)) or not math.isfinite(error):
                raise RuntimeError(f"{route} parity error is not finite")
            print(f"benchmark_started:{route}", flush=True)
            benchmark = _benchmark(
                benchmark_executable, model_path, float(error), protocol
            )
            write_new_json(RUN_DIR / f"benchmark-{route}.json", benchmark)
            route_evidence[route] = {"parity": parity, "benchmark": benchmark}
            print(f"native_completed:{route}", flush=True)
        evidence = {
            "schema_version": 1,
            "campaign_version": CAMPAIGN_VERSION,
            "run_id": RUN_ID,
            "physical_audio_samples_read": 0,
            "mechanism_evidence_path": str(MECHANISM_PATH.relative_to(ROOT)),
            "routes": route_evidence,
            "provenance": provenance,
        }
        gate = evaluate_native_gate(evidence, mechanism_gate)
        decision = final_backend_decision(mechanism_gate, gate)
        evidence["gate"] = gate
        evidence["preliminary_decision"] = decision
        write_new_json(RUN_DIR / "native.json", evidence)
        write_new_json(SUMMARY_PATH, evidence)
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        return _record_invalid(
            reason=reason,
            started_at=started_at,
            provenance=provenance,
            protocol=protocol,
        )
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    append_run(
        GLOBAL_LEDGER,
        {
            "date": finished_at,
            "run_id": RUN_ID,
            "phase": "QUALITY-AA-NATIVE",
            "model": "latency32_x2_x4",
            "device": "host_cpu",
            "seed": 0,
            "commit": provenance["git_head"],
            "config_sha256": digest_text(strict_json(protocol)),
            "data_sha256": digest_text("deterministic_native_parity_signal_v1"),
            "status": "completed",
            "failure_reason": "",
            "results_path": str(RUN_DIR.relative_to(ROOT)),
        },
    )
    replace_json(
        RUN_DIR / "status.json",
        {
            "status": "completed",
            "started_at": started_at,
            "finished_at": finished_at,
            "failure_reason": "",
        },
    )
    append_gate_event(
        CAMPAIGN_DIR / "GATE_LEDGER.jsonl",
        {
            "campaign_version": CAMPAIGN_VERSION,
            "stage": "native",
            "status": "passed",
            "evidence_path": str(SUMMARY_PATH.relative_to(ROOT)),
        },
    )
    replace_json(
        CAMPAIGN_DIR / "MATURITY.json",
        {
            "campaign_version": CAMPAIGN_VERSION,
            "current_stage": "native_complete",
            "status": "active",
            "scientific_runs_launched": 3,
            "invalid": False,
        },
    )
    print(
        json.dumps(
            {"gate": gate, "preliminary_decision": decision},
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
