#!/usr/bin/env python3
"""Run the metadata-only AMP-QUALITY-ARCH-v1 preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fssr_nam.reporting.arch_preflight import build_preflight_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--require-frozen", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    report = build_preflight_report(root, require_frozen=args.require_frozen)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
