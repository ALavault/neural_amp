#!/usr/bin/env python3
"""Run the strict AMP-SOTA-PROTOTYPE-v1.1 Python interface."""

from pathlib import Path

from fssr_nam.inference.prototype_cli import main

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(main(ROOT))
