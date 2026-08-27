#!/usr/bin/env python3
"""Evaluate the frozen FSSR-R1 confirmation from source-level JSON rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fssr_nam.statistics.r1 import (
    CostObservation,
    EsrObservation,
    decide_confirmation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--esr", type=Path, required=True)
    parser.add_argument("--cost", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_827)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list: {path}")
    return payload


def main() -> None:
    args = parse_args()
    esr = [EsrObservation(**row) for row in load_rows(args.esr)]
    cost = [CostObservation(**row) for row in load_rows(args.cost)]
    result = decide_confirmation(
        esr,
        cost,
        replicates=args.replicates,
        seed=args.seed,
    ).to_dict()
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        if args.output.exists():
            raise RuntimeError(f"refusing to overwrite result: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
