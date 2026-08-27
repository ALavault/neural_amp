#!/usr/bin/env python3
"""Execute and persist the conditional three-seed R1 competence gate."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

from fssr_nam.reporting.ledger import read_runs

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/training/r1_competence.yaml"
LEDGER_PATH = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
R1_ROOT = ROOT / ".codex_campaign/r1"


def run_id(seed: int) -> str:
    return f"r1_competence_bigmuff_wright_lstm64_wright_seed{seed}_v1"


def run_seed(seed: int) -> None:
    identifier = run_id(seed)
    entries = {entry["run_id"]: entry for entry in read_runs(LEDGER_PATH)}
    if identifier in entries:
        if entries[identifier]["status"] != "completed":
            raise RuntimeError(f"existing competence run failed: {identifier}")
        return
    directory = ROOT / "experiments/runs" / identifier
    command = [
        sys.executable,
        str(ROOT / "scripts/run_r1_competence.py"),
        "--seed",
        str(seed),
        "--run-id",
        identifier,
    ]
    if directory.is_dir():
        status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        if status.get("status") != "running":
            raise RuntimeError(f"unregistered non-running directory: {identifier}")
        command.append("--resume")
    subprocess.run(command, cwd=ROOT, check=True)


def test_esr(seed: int) -> float:
    path = ROOT / "experiments/runs" / run_id(seed) / "metrics.json"
    return float(json.loads(path.read_text(encoding="utf-8"))["test_esr"])


def persist(status: str, used: int, values: list[float], reason: str = "") -> None:
    maturity_path = R1_ROOT / "MATURITY.json"
    maturity = json.loads(maturity_path.read_text(encoding="utf-8"))
    maturity["status"] = "diagnostic" if status == "passed" else "failed_gate"
    maturity["current_stage"] = "factorial" if status == "passed" else "audit"
    maturity["gates"]["competence"] = status
    if status == "passed":
        maturity["gates"]["factorial"] = "not_started"
    maturity["diagnostic_trajectories_used"] = used
    maturity["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    maturity_path.write_text(json.dumps(maturity, indent=2) + "\n", encoding="utf-8")

    verdict = "passed" if status == "passed" else f"failed: {reason}"
    current_gate = (
        "factorial not started"
        if status == "passed"
        else "competence failed; audit required"
    )
    state = f"""# FSSR-R1 campaign state

- Campaign: `FSSR-R1-v1`
- Lineage: prospective; the M0-M6 `NO-GO` remains immutable
- State: competence gate {verdict}
- Diagnostic trajectories used: {used} / 19
- Confirmatory runs used: 0 / 76
- Current gate: {current_gate}
- Competence test ESR values: {values}
- Candidate: none
- Confirmatory hypothesis: none
- Verdict: not evaluated
- `EXTERNAL_REPORT_ONLY`: locked and unaccessed
"""
    (R1_ROOT / "STATE.md").write_text(state, encoding="utf-8")
    action = (
        "Proceed to the locked loss x budget factorial."
        if status == "passed"
        else "Do not execute architectural diagnostics; audit the competence failure."
    )
    handoff = f"""# FSSR-R1 handoff

The competence gate is **{status}** after {used} counted trajectory or
trajectories. Test ESR values: `{values}`. {reason}

{action}
`EXTERNAL_REPORT_ONLY` and INTERNAL_VALIDATION model outputs remain locked.
"""
    (R1_ROOT / "HANDOFF.md").write_text(handoff, encoding="utf-8")
    claims_path = R1_ROOT / "CLAIMS.md"
    claims = claims_path.read_text(encoding="utf-8")
    claims = claims.replace(
        "| Wright LSTM-64 competence is reproduced. | untested | no run |",
        "| Wright LSTM-64 competence is reproduced. "
        f"| {status} | registered R1 competence runs |",
    )
    claims_path.write_text(claims, encoding="utf-8")
    if status != "passed":
        with (R1_ROOT / "FAILURES.md").open("a", encoding="utf-8") as stream:
            stream.write(
                f"\n## {datetime.now().date()} - Competence gate\n\n{reason}; "
                f"test ESR values: `{values}`.\n"
            )


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    seed_zero_limit = float(
        config["conditional_seed_rule"]["launch_if_seed0_esr_test_at_most"]
    )
    run_seed(0)
    values = [test_esr(0)]
    if values[0] > seed_zero_limit:
        reason = f"seed 0 ESR {values[0]:.9g} exceeds {seed_zero_limit}"
        persist("failed", 1, values, reason)
        print(reason)
        return
    for seed in config["conditional_seed_rule"]["additional_seeds"]:
        run_seed(int(seed))
        values.append(test_esr(int(seed)))
    median = statistics.median(values)
    maximum = max(values)
    gate = config["promotion_gate"]
    median_limit = float(gate["median_esr_test_at_most"])
    maximum_limit = float(gate["maximum_seed_esr_test_at_most"])
    if median <= median_limit and maximum <= maximum_limit:
        persist("passed", len(values), values)
        print(
            f"competence passed: median={median:.9g} maximum={maximum:.9g}",
            flush=True,
        )
    else:
        reason = (
            f"median={median:.9g} (limit {median_limit}) and "
            f"maximum={maximum:.9g} (limit {maximum_limit})"
        )
        persist("failed", len(values), values, reason)
        print(f"competence failed: {reason}", flush=True)


if __name__ == "__main__":
    main()
