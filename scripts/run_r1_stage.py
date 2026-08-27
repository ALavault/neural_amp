#!/usr/bin/env python3
"""Plan, reserve, or finalize one declaratively authorized FSSR-R1 run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from fssr_nam.campaign.r1 import (
    DIAGNOSTIC_STAGE_CAPS,
    R1Executor,
    RunSpec,
    load_gate_decisions,
    validate_repository_configs,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / ".codex_campaign/r1/CONFIRMATORY_LOCK.yaml"


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_lock(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("confirmatory lock must be a YAML mapping")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["competence", "factorial", "horizon", "cascade", "confirm"],
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="print the full stage matrix")
    mode.add_argument(
        "--finalize-status",
        choices=["completed", "failed", "stopped_by_gate"],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--model")
    parser.add_argument("--loss")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gates", type=Path)
    parser.add_argument("--confirmatory-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--data-sha256")
    parser.add_argument("--commit")
    parser.add_argument("--command", default="")
    parser.add_argument("--failure-reason", default="")
    return parser


def _selected_spec(args: argparse.Namespace) -> RunSpec:
    missing = [
        name
        for name in ("device", "model", "loss", "seed")
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(f"selected run missing arguments: {', '.join(missing)}")
    return RunSpec(args.stage, args.device, args.model, args.loss, args.seed)


def main() -> None:
    args = _parser().parse_args()
    configs = validate_repository_configs(ROOT)
    decisions = load_gate_decisions(args.gates) if args.gates else {}
    executor = R1Executor(ROOT, stage_configs=configs, gate_decisions=decisions)

    if args.plan:
        specs = executor.plan(args.stage, resolved_loss=args.loss)
        cap = DIAGNOSTIC_STAGE_CAPS.get(args.stage, 76)
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "count": len(specs),
                    "cap": cap,
                    "writes_performed": False,
                    "runs": [spec.as_dict() for spec in specs],
                },
                indent=2,
            )
        )
        return

    spec = _selected_spec(args)
    if args.finalize_status:
        if args.dry_run:
            raise SystemExit("--dry-run cannot be combined with --finalize-status")
        result = executor.finalize_run(
            spec,
            status=args.finalize_status,
            failure_reason=args.failure_reason,
        )
    else:
        if args.data_sha256 is None:
            raise SystemExit("reservation requires --data-sha256")
        lock = _load_lock(args.confirmatory_lock) if args.stage == "confirm" else None
        result = executor.prepare_run(
            spec,
            data_sha256=args.data_sha256,
            commit=args.commit or _git_commit(),
            command=args.command,
            confirmatory_lock=lock,
            dry_run=args.dry_run,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
