#!/usr/bin/env python3
"""Reproducibility gate of 2026-09-24. (Read-only on the runs, no GPU.)

The bench and ssm_wavenet.py changed since the reference runs were trained (FiLM
controls, parametric devices, val_indices). Under --deterministic a same-seed rerun is
bitwise identical (diagnosis/butterfly/determinism.json), so a short rerun at HEAD must
reproduce every value its reference logged over the same steps: the logged train loss,
per-parameter gradient norms, learning rate and every validation metric. Any difference
is a confound to resolve before comparing new runs with old ones.

Values are compared as the strings CSVLogger wrote, which are exact float reprs, so
string equality is bitwise equality. The gate runs are recorded in demo/RUNS.jsonl
by this script, once, with the verdict.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "demo/runs"
PAIRS = {
    "repro_gate_ssm_bigmuff": "pilotC_decide_seed42",
    "repro_gate_s4tf_bigmuff": "paired_s4_tf_l_16_s925",
    "repro_gate_ssm_marshall": "param_ssm_wavenet_marshall",
}


def rows(run: str) -> list[dict[str, str]]:
    with (RUNS / f"nablafx_{run}/logs/metrics.csv").open() as stream:
        return [
            {k: v for k, v in row.items() if v != ""} for row in csv.DictReader(stream)
        ]


def environment() -> dict:
    import lightning
    import torch

    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "lightning": lightning.__version__,
        "driver_gpu": driver,
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
    }


def compare(gate: str, reference: str) -> dict:
    new, old = rows(gate), rows(reference)
    last = max(int(r["step"]) for r in new if "step" in r)
    # Rows the reference logged at steps the gate also reached, in logging order.
    old = [r for r in old if "step" in r and int(r["step"]) <= last]
    new = [r for r in new if "step" in r and int(r["step"]) <= last]
    # Stopping at --max-steps closes the running epoch early, and Lightning then logs
    # that epoch's train loss averaged over its first few steps only. The reference
    # logs the same row at the end of the full epoch, past `last`. Such trailing rows,
    # after every reference row, are set aside and reported, not compared.
    truncated = (
        new[len(old) :] if all(int(r["step"]) == last for r in new[len(old) :]) else []
    )
    new = new[: len(new) - len(truncated)]
    values = sum(len(r) for r in old)
    mismatches = []
    for i, (a, b) in enumerate(zip(old, new, strict=False)):
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                mismatches.append(
                    {
                        "row": i,
                        "step": a.get("step"),
                        "column": key,
                        "reference": a.get(key),
                        "gate": b.get(key),
                    }
                )
    return {
        "reference": reference,
        "last_step": last,
        "rows_reference": len(old),
        "rows_gate": len(new),
        "truncated_epoch_rows": truncated,
        "values_compared": values,
        "mismatches": len(mismatches),
        "first_mismatches": mismatches[:10],
        "identical": len(old) == len(new) and not mismatches,
    }


def main() -> int:
    missing = [
        g for g in PAIRS if not (RUNS / f"nablafx_{g}/logs/metrics.csv").exists()
    ]
    if missing:
        print(f"gate runs missing: {', '.join(missing)}")
        return 1
    results = {gate: compare(gate, ref) for gate, ref in PAIRS.items()}
    for gate, r in results.items():
        verdict = "IDENTIQUE" if r["identical"] else "DIFFERE"
        print(
            f"{gate:26s} vs {r['reference']:28s} pas <= {r['last_step']:3d} : "
            f"{r['rows_reference']} lignes / {r['rows_gate']}, "
            f"{r['values_compared']} valeurs, {r['mismatches']} ecarts -> {verdict}"
        )
        for row in r["truncated_epoch_rows"]:
            print(f"    ecartee, epoque tronquee {row['epoch']}, pas {row['step']}")
        for m in r["first_mismatches"]:
            print(
                f"    ligne {m['row']} pas {m['step']} {m['column']}: "
                f"{m['reference']} != {m['gate']}"
            )
    passed = all(r["identical"] for r in results.values())
    print(f"\nporte franchie : {passed}")
    out = {"environment": environment(), "results": results, "passed": passed}
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    log = ROOT / "demo/RUNS.jsonl"
    if not any('"repro_gate_' in line for line in log.read_text().splitlines()):
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        with log.open("a", encoding="utf-8") as stream:
            for gate, r in results.items():
                stream.write(
                    json.dumps(
                        {
                            "run_id": gate,
                            "status": "gate",
                            "timestamp": stamp,
                            "reference": r["reference"],
                            "last_step": r["last_step"],
                            "identical": r["identical"],
                            "mismatches": r["mismatches"],
                            "record": "diagnosis/seeds/repro_gate.json",
                        }
                    )
                    + "\n"
                )
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
