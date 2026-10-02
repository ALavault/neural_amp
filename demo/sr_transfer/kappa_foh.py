#!/usr/bin/env python3
"""T1-P1: κ per octave band of the FOH-trained SSM-WaveNets (THEORY.md, T1).

Read-only, CPU. Rules and gates written in THEORY.md (commit a717d5b) before any
measurement. Same method as kappa.py:bigmuff (Δ rescaled to 96 and 192 kHz, back to
48 kHz, delay per octave band against the 48 kHz output), on
paired_ssm_wavenet_foh_s{925,736,688}, checkpoint last. Gates: every SSM is FOH; the
48 kHz ESR is within 2 % of the _last_plain.json record; the round-trip ESR is
<= 1e-3. Writes demo/sr_transfer/kappa_foh.json.
"""

from __future__ import annotations

import importlib
import json
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
kappa = importlib.import_module("kappa")
bench, transfer = kappa.bench, kappa.transfer

RUNS = [
    "paired_ssm_wavenet_foh_s925",
    "paired_ssm_wavenet_foh_s736",
    "paired_ssm_wavenet_foh_s688",
]
PREDICTED = ("500", "1000", "2000")


@torch.no_grad()
def measure(run_id: str, x48: torch.Tensor, y48: torch.Tensor) -> dict:
    model, _ = transfer.load_model(run_id, "last")
    modules = transfer.ssm_modules(model)
    assert modules and all(m.discretization == "foh" for m in modules), run_id
    base = {id(m): (m.log_dt.detach().clone(), m.C.detach().clone()) for m in modules}
    kappa.set_block(model, kappa.RATE, base, None)
    y = transfer.run(model, x48)
    esr48 = transfer.score(y, y48)["esr"]
    plain = HERE / f"sr_transfer_{run_id}_last_plain.json"
    recorded = json.loads(plain.read_text())["arms"]["48k"]["esr"]
    out48 = transfer.interior(y, kappa.RATE)
    out = {"esr_48k": esr48, "esr_48k_recorded": recorded}
    gates = [abs(esr48 / recorded - 1) <= 0.02]
    for rate in kappa.RATES:
        kappa.set_block(model, rate, base, None)
        z = transfer.run(model, transfer.whole(x48, kappa.RATE, rate))
        back = transfer.interior(transfer.whole(z, rate, kappa.RATE), kappa.RATE)
        roundtrip = transfer.score(back, out48)["esr"]
        gates.append(roundtrip <= 1e-3)
        delays = kappa.band_delays(out48.reshape(-1).numpy(), back.reshape(-1).numpy())
        kappas = {c: d / kappa.gauge(rate) for c, d in delays.items() if d is not None}
        low = [delays[c] for c in PREDICTED if delays[c] is not None]
        out[str(rate)] = {
            "roundtrip_esr": roundtrip,
            "delay_by_band": delays,
            "kappa_by_band": kappas,
            "missing_predicted_bands": [c for c in PREDICTED if delays[c] is None],
            "kappa_low": statistics.median(low) / kappa.gauge(rate) if low else None,
        }
    out["gates_passed"] = all(gates)
    print(f"{run_id} fait, portes {out['gates_passed']}", flush=True)
    return out


def main() -> None:
    torch.set_num_threads(8)
    data = bench.data_module("test")
    data.setup("test")
    x48 = torch.stack([s[0] for s in data.test_dataset])
    y48 = torch.stack([s[1] for s in data.test_dataset])
    result = {"checkpoints": {r: measure(r, x48, y48) for r in RUNS}}
    cks = result["checkpoints"].values()
    rates = [str(r) for r in kappa.RATES]
    gates = all(ck["gates_passed"] for ck in cks)
    readings = {"gates_passed": gates}
    if gates:
        primary = statistics.median(ck[r]["kappa_low"] for ck in cks for r in rates)
        pooled = [
            ck[r]["kappa_by_band"][c]
            for ck in cks
            for r in rates
            for c in PREDICTED
            if c in ck[r]["kappa_by_band"]
        ]
        readings |= {
            "kappa_primary": primary,
            "verdict": "soutenue"
            if abs(primary) <= 0.15
            else "réfutée"
            if abs(primary) >= 0.5
            else "intermédiaire",
            "kappa_pooled_secondary": statistics.median(pooled),
            "pooled_count": len(pooled),
        }
    result["readings"] = readings
    HERE.joinpath("kappa_foh.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(readings, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
