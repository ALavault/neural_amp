#!/usr/bin/env python3
"""Compute R1 diagnostic gates from complete validation-only run artifacts."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from fssr_nam.campaign.r1 import (
    active_gate_registry_path,
    make_run_id,
    validate_active_gate_registry,
)
from fssr_nam.campaign.r1_gates import (
    R1GateEvidenceError,
    evaluate_cascade_physical_gate,
    evaluate_cascade_synthetic_gate,
    evaluate_factorial_gate,
    evaluate_horizon_gate,
)
from fssr_nam.reporting.r1_preflight import validate_lock_digest
from fssr_nam.training.r1_diagnostic import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATES = active_gate_registry_path(ROOT)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("factorial", "horizon", "cascade-synthetic", "cascade-physical"),
    )
    parser.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise R1GateEvidenceError(f"JSON artifact must be a mapping: {path}")
    return value


def _passed(value: object) -> bool:
    if value is True or (isinstance(value, str) and value in {"passed", "promoted"}):
        return True
    return isinstance(value, dict) and (
        value.get("passed") is True or value.get("decision") in {"passed", "promoted"}
    )


def _existing_decisions(path: Path) -> dict[str, object]:
    active = validate_active_gate_registry(ROOT)
    if path.resolve() != active.resolve():
        raise R1GateEvidenceError("gate evaluation requires the active registry")
    document = _load_json(active)
    gates = document.get("gates", document)
    if not isinstance(gates, dict):
        raise R1GateEvidenceError("gate artifact 'gates' must be a mapping")
    return dict(gates)


def _require_gate(decisions: dict[str, object], name: str) -> dict[str, Any] | object:
    value = decisions.get(name)
    if not _passed(value):
        raise R1GateEvidenceError(f"prerequisite gate is not passed: {name}")
    return value


def _metrics(run_id: str) -> dict[str, Any]:
    run_dir = ROOT / "experiments/runs" / run_id
    if not run_dir.is_dir():
        raise R1GateEvidenceError(f"missing run directory: {run_id}")
    status = _load_json(run_dir / "status.json")
    if status.get("status") != "completed":
        raise R1GateEvidenceError(f"run is not completed: {run_id}")
    metrics = _load_json(run_dir / "metrics.json")
    if metrics.get("run_id") != run_id:
        raise R1GateEvidenceError(f"metrics run_id mismatch: {run_id}")
    return metrics


def _factorial_rows(losses: tuple[str, ...] = ("m4", "wright")) -> list[dict[str, Any]]:
    return [
        _metrics(make_run_id("factorial", device, model, loss, 0))
        for device in ("fulltone", "bigmuff")
        for model in ("a2", "s3")
        for loss in losses
    ]


def _horizon_rows(
    loss: str, models: tuple[str, ...] = ("rf31", "rf2047")
) -> list[dict[str, Any]]:
    return [
        _metrics(make_run_id("horizon", device, model, loss, 0))
        for device in ("fulltone", "bigmuff")
        for model in models
    ]


def _cascade_rows(loss: str, *, physical: bool) -> list[dict[str, Any]]:
    if physical:
        conditions = (("bigmuff", "cascade"), ("fulltone", "cascade"))
    else:
        conditions = (("synthetic", "mono"), ("synthetic", "cascade"))
    return [
        _metrics(make_run_id("cascade", device, model, loss, 0))
        for device, model in conditions
    ]


def _selected_loss(decisions: dict[str, object]) -> str:
    factorial = _require_gate(decisions, "factorial")
    if not isinstance(factorial, dict) or factorial.get("selected_loss") not in {
        "m4",
        "wright",
    }:
        raise R1GateEvidenceError("factorial gate lacks selected_loss")
    return str(factorial["selected_loss"])


def _promoted_residuals(loss: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for device in ("fulltone", "bigmuff"):
        run_id = make_run_id("horizon", device, "rf2047", loss, 0)
        run_dir = ROOT / "experiments/runs" / run_id
        index = _load_json(run_dir / "checkpoints/index.json")
        reference = index.get("promoted_residual")
        if not isinstance(reference, dict):
            raise R1GateEvidenceError(f"RF2047 checkpoint is not promotable: {run_id}")
        relative = reference.get("path")
        recorded_digest = reference.get("sha256")
        if relative != "checkpoints/promoted-residual-state.pt":
            raise R1GateEvidenceError(f"unexpected promoted residual path: {run_id}")
        checkpoint = run_dir / relative
        if (
            not isinstance(recorded_digest, str)
            or sha256_file(checkpoint) != recorded_digest
        ):
            raise R1GateEvidenceError(f"promoted residual digest mismatch: {run_id}")
        result[device] = {
            "run_id": run_id,
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": recorded_digest,
        }
    return result


def _evaluate(stage: str, decisions: dict[str, object]) -> tuple[str, dict[str, Any]]:
    configs = {
        name: yaml.safe_load(
            (ROOT / f"configs/training/r1_{name}.yaml").read_text(encoding="utf-8")
        )
        for name in ("factorial", "horizon", "cascade")
    }
    if stage == "factorial":
        _require_gate(decisions, "competence")
        return "factorial", evaluate_factorial_gate(
            _factorial_rows(), configs["factorial"]
        )
    loss = _selected_loss(decisions)
    if stage == "horizon":
        return "horizon", evaluate_horizon_gate(
            _horizon_rows(loss),
            _factorial_rows((loss,)),
            selected_loss=loss,
            config=configs["horizon"],
            promoted_residuals=_promoted_residuals(loss),
        )
    _require_gate(decisions, "horizon")
    if stage == "cascade-synthetic":
        return "cascade_synthetic", evaluate_cascade_synthetic_gate(
            _cascade_rows(loss, physical=False),
            selected_loss=loss,
            config=configs["cascade"],
        )
    _require_gate(decisions, "cascade_synthetic")
    return "cascade", evaluate_cascade_physical_gate(
        _cascade_rows(loss, physical=True),
        _horizon_rows(loss, ("rf2047",)),
        selected_loss=loss,
        config=configs["cascade"],
    )


def _persist(path: Path, gate_name: str, result: dict[str, Any]) -> None:
    if path.exists():
        document = _load_json(path)
    else:
        document = {
            "schema_version": 1,
            "campaign_version": "FSSR-R1-v1",
            "gates": {},
        }
    gates = document.get("gates")
    if not isinstance(gates, dict):
        raise R1GateEvidenceError("gate artifact 'gates' must be a mapping")
    if gate_name in gates:
        raise R1GateEvidenceError(
            f"gate decision is immutable and already exists: {gate_name}"
        )
    result = dict(result)
    result["evaluated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    result["protocol_sha256"] = validate_lock_digest(ROOT)
    gates[gate_name] = result
    document["updated_at"] = result["evaluated_at"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = _parser().parse_args()
    validate_lock_digest(ROOT)
    decisions = _existing_decisions(args.gates)
    gate_name, result = _evaluate(args.stage, decisions)
    output = {"gate": gate_name, **result, "writes_performed": not args.dry_run}
    if not args.dry_run:
        _persist(args.gates, gate_name, result)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
