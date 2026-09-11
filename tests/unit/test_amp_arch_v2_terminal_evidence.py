from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from fssr_nam.campaign.amp_arch_v2_gates import evaluate_competence_gate
from fssr_nam.campaign.amp_competence_arch_v2 import (
    CHECKPOINTS,
    CONTROL_FAMILY,
    SEEDS,
    SYSTEMS,
    make_run_id,
)
from fssr_nam.reporting.ledger import read_runs

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_DIR = ROOT / ".codex_campaign/amp_competence_arch_v2"
SUMMARY_PATH = ROOT / "experiments/summaries/amp_competence_arch_v2/competence.json"
GLOBAL_LEDGER = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"


def _protocol() -> dict[str, object]:
    return yaml.safe_load(
        (ROOT / "configs/amp_competence_arch_v2/protocol.yaml").read_text(
            encoding="utf-8"
        )
    )


def _expected_run_ids() -> list[str]:
    return [
        make_run_id("competence", system, CONTROL_FAMILY, seed)
        for system in SYSTEMS
        for seed in SEEDS
    ]


def test_terminal_competence_evidence_recomputes_exact_no_go() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    recomputed = evaluate_competence_gate(summary["trajectories"], _protocol())
    assert recomputed == summary["gate"]
    assert summary["status"] == "failed"
    assert summary["run_ids"] == _expected_run_ids()
    assert summary["trajectory_count"] == 9
    assert summary["candidate_runs_launched"] == 0
    assert summary["physical_audio_samples_read"] == 0
    assert summary["sealed_outputs_accessed"] == {
        "blackstar": False,
        "ua1176": False,
    }
    assert recomputed["passed"] is False
    assert recomputed["selected_budget_updates"] is None
    verdict = json.loads((CAMPAIGN_DIR / "VERDICT.json").read_text(encoding="utf-8"))
    maturity = json.loads((CAMPAIGN_DIR / "MATURITY.json").read_text(encoding="utf-8"))
    assert verdict["verdict"] == "NO-GO-COMPETENCE-v2"
    assert maturity["verdict"] == "NO-GO-COMPETENCE-v2"
    assert maturity["competence_runs_launched"] == 9
    assert maturity["comparison_runs_launched"] == 0


def test_terminal_run_artifacts_are_complete_loadable_and_source_identical() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    by_key = {(row["system"], row["seed"]): row for row in summary["trajectories"]}
    assert set(by_key) == {(system, seed) for system in SYSTEMS for seed in SEEDS}
    reference_snapshot: dict[str, bytes] | None = None
    for system in SYSTEMS:
        for seed in SEEDS:
            run_id = make_run_id("competence", system, CONTROL_FAMILY, seed)
            run_dir = ROOT / "experiments/runs" / run_id
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            assert status["status"] == "completed"
            assert result == by_key[(system, seed)]
            assert result["family"] == CONTROL_FAMILY
            assert result["auxiliary_weight"] == 0.0
            assert result["updates"] == CHECKPOINTS[-1]
            assert [row["update"] for row in result["checkpoints"]] == list(CHECKPOINTS)
            checkpoint_dir = run_dir / "checkpoints"
            assert {path.name for path in checkpoint_dir.iterdir()} == {
                f"update_{update}.pt" for update in CHECKPOINTS
            }
            for update in CHECKPOINTS:
                state = torch.load(
                    checkpoint_dir / f"update_{update}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                assert state
                assert all(torch.isfinite(value).all() for value in state.values())
            snapshot_root = run_dir / "source_snapshot"
            snapshot = {
                str(path.relative_to(snapshot_root)): path.read_bytes()
                for path in snapshot_root.rglob("*")
                if path.is_file()
            }
            assert snapshot
            if reference_snapshot is None:
                reference_snapshot = snapshot
            else:
                assert snapshot == reference_snapshot


def test_terminal_ledgers_have_exactly_nine_controls_and_zero_candidates() -> None:
    expected = set(_expected_run_ids())
    entries = [
        entry
        for entry in read_runs(GLOBAL_LEDGER)
        if entry["run_id"].startswith("arch_v2_")
    ]
    assert {entry["run_id"] for entry in entries} == expected
    assert len(entries) == len(expected)
    assert all(entry["status"] == "completed" for entry in entries)
    assert all(
        entry["phase"] == "AMP-COMPETENCE-ARCH-v2-COMPETENCE" for entry in entries
    )
    assert all(entry["model"] == CONTROL_FAMILY for entry in entries)
    assert not list((ROOT / "experiments/runs").glob("arch_v2_comparison_*"))
    gate_events = [
        json.loads(line)
        for line in (CAMPAIGN_DIR / "GATE_LEDGER.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    expected_events = [
        ("preflight", "passed"),
        ("competence", "failed"),
    ]
    if (CAMPAIGN_DIR / "AUDIT.json").is_file():
        expected_events.append(("audit", "passed"))
    assert [(event["stage"], event["status"]) for event in gate_events] == (
        expected_events
    )


def test_v1_contract_and_sealed_boundaries_remain_preserved() -> None:
    v1_protocol = yaml.safe_load(
        (ROOT / "configs/amp_quality_arch_v1/protocol.yaml").read_text(encoding="utf-8")
    )
    v1_lock = yaml.safe_load(
        (ROOT / ".codex_campaign/amp_quality_arch_v1/PROTOCOL_LOCK.yaml").read_text(
            encoding="utf-8"
        )
    )
    v1_verdict = json.loads(
        (ROOT / ".codex_campaign/amp_quality_arch_v1/VERDICT.json").read_text(
            encoding="utf-8"
        )
    )
    freeze = json.loads(
        (CAMPAIGN_DIR / "EXTERNAL_FREEZE.json").read_text(encoding="utf-8")
    )
    assert v1_protocol == v1_lock
    assert v1_verdict["verdict"] == "NO-GO-ARCH"
    assert freeze["physical_audio_allowed"] is False
    assert freeze["sealed_test_output_accessed"] is False
    assert freeze["external_report_only_locked"] is True
