#!/usr/bin/env python3
"""Exit 0 if a VS100R probe left the plateau: last validation loss < 2.0 and test ESR
< 0.9 (rule of scripts/product_vs100r_probe_queue.sh, fixed before the probes)."""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
run = sys.argv[1]
record = json.loads((ROOT / f"demo/nablafx_bench/{run}.json").read_text())
with (ROOT / f"demo/runs/nablafx_{run}/logs/metrics.csv").open() as stream:
    val = [
        float(r["loss/val/tot"]) for r in csv.DictReader(stream) if r["loss/val/tot"]
    ]
esr = record["test_last"]["metric/test/esr"]
print(f"=== probes: {run} last val loss {val[-1]:.3f}, test ESR {esr:.3f}")
sys.exit(0 if val[-1] < 2.0 and esr < 0.9 else 1)
