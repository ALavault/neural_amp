#!/usr/bin/env python3
"""κ: the network's delay when moved to another sample rate (predictions_kappa.md).

Read-only, CPU. Pre-registered 2026-10-02 (commit 2b447f3) before any measurement.
Per octave band (500 Hz to 8 kHz), the delay of the model with its Δ rescaled to
96 or 192 kHz (no correction), brought back to 48 kHz, against the same model at
48 kHz, in samples at 48 kHz (positive: the rescaled model lags); κ = delay / ((1 -
48000/fs) / 2). Checkpoints: vs100r_gc_film_joint_s925_r2 (8 test takes) and
paired_ssm_wavenet_s925/s736/s688 (bench test segments); for the VS100R also each
block rescaled alone. Writes demo/sr_transfer/kappa.json.
"""

from __future__ import annotations

import importlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import csd, resample_poly, welch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
bench = importlib.import_module("product_nablafx_bench")
transfer = importlib.import_module("product_sr_transfer")
ev = importlib.import_module("product_vs100r_eval")

RATE = 48_000
RATES = (96_000, 192_000)
NPERSEG = 4096
CENTRES = (500, 1_000, 2_000, 4_000, 8_000)
VS100R = "vs100r_gc_film_joint_s925_r2"
BIGMUFF = [
    "paired_ssm_wavenet_s925",
    "paired_ssm_wavenet_s736",
    "paired_ssm_wavenet_s688",
]


def band_delays(a: np.ndarray, b: np.ndarray) -> dict[str, float | None]:
    """Delay of b relative to a, in samples at 48 kHz, per octave band
    (group_delay.py's method)."""
    freqs, s_ab = csd(a, b, RATE, nperseg=NPERSEG)
    _, s_aa = welch(a, RATE, nperseg=NPERSEG)
    _, s_bb = welch(b, RATE, nperseg=NPERSEG)
    coherence = np.abs(s_ab) ** 2 / (s_aa * s_bb + 1e-30)
    out = {}
    for centre in CENTRES:
        band = (freqs >= centre / 2**0.5) & (freqs < centre * 2**0.5)
        keep = band & (coherence >= 0.5)
        if keep.sum() < 4:
            out[str(centre)] = None
            continue
        omega = 2 * np.pi * freqs[keep] / RATE
        phase = np.unwrap(np.angle(s_ab[keep]))
        out[str(centre)] = float(-np.polyfit(omega, phase, 1, w=np.abs(s_ab[keep]))[0])
    return out


def gauge(rate: int) -> float:
    return (1 - RATE / rate) / 2


def set_block(model, rate: int, base: dict, block: int | None) -> None:
    """Rescale Δ of every SSM (block None) or of one block only; others at 48 kHz."""
    for i, module in enumerate(transfer.ssm_modules(model)):
        log_dt, _ = base[id(module)]
        target = rate if block is None or i == block else RATE
        module.log_dt.data = log_dt + math.log(RATE / target)


def summarise(delays: list[dict], rate: int) -> dict:
    """Per band: median over items; then the checkpoint's median and band spread."""
    per_band = {}
    for centre in map(str, CENTRES):
        values = [d[centre] for d in delays if d[centre] is not None]
        per_band[centre] = statistics.median(values) if values else None
    pooled = [d[c] for d in delays for c in map(str, CENTRES) if d[c] is not None]
    bands = [v for v in per_band.values() if v is not None]
    median = statistics.median(pooled)
    return {
        "delay_by_band": per_band,
        "delay_median": median,
        "kappa": median / gauge(rate),
        "band_sd": statistics.stdev(bands) if len(bands) > 1 else None,
        "pure_delay": (statistics.stdev(bands) <= 0.1) if len(bands) > 1 else None,
    }


@torch.no_grad()
def vs100r() -> dict:
    model = ev.load(VS100R)
    base = {
        id(m): (m.log_dt.detach().clone(), m.C.detach().clone())
        for m in transfer.ssm_modules(model)
    }
    chosen = [t for t in ev.takes() if t["split"] == "test" and not t["low_signal"]]
    variants = [None, *range(len(transfer.ssm_modules(model)))]
    raw = {f"{v}|{r}": [] for v in variants for r in RATES}
    for take in chosen:
        loop, _ = ev.audio(take)
        controls = torch.tensor([v / 10 for v in take["gain_contour"]])
        set_block(model, RATE, base, None)
        m48 = ev.scored(ev.run(model, resample_poly(loop, 1, 4), controls))
        for rate in RATES:
            x = resample_poly(loop, 1, ev.CAPTURE // rate)
            for v in variants:
                set_block(model, rate, base, v)
                back = resample_poly(ev.run(model, x, controls), 1, rate // RATE)
                raw[f"{v}|{rate}"].append(band_delays(m48, ev.scored(back)))
        print(f"VS100R prise {take['order']} faite", flush=True)
    out = {"takes": [t["order"] for t in chosen]}
    for rate in RATES:
        whole = summarise(raw[f"None|{rate}"], rate)
        blocks = [
            summarise(raw[f"{b}|{rate}"], rate)["delay_median"] for b in variants[1:]
        ]
        out[str(rate)] = {
            **whole,
            "block_delays": blocks,
            "block_sum": sum(blocks),
            "cascade": abs(sum(blocks) - whole["delay_median"])
            <= 0.2 * abs(whole["delay_median"]),
            "intra_layer": max(blocks) > gauge(rate),
        }
    out["raw"] = raw
    return out


@torch.no_grad()
def bigmuff(run_id: str) -> dict:
    model, _ = transfer.load_model(run_id, "last")
    base = {
        id(m): (m.log_dt.detach().clone(), m.C.detach().clone())
        for m in transfer.ssm_modules(model)
    }
    data = bench.data_module("test")
    data.setup("test")
    x48 = torch.stack([s[0] for s in data.test_dataset])
    set_block(model, RATE, base, None)
    out48 = transfer.interior(transfer.run(model, x48), RATE).reshape(-1).numpy()
    out = {}
    for rate in RATES:
        set_block(model, rate, base, None)
        y = transfer.run(model, transfer.whole(x48, RATE, rate))
        back = transfer.interior(transfer.whole(y, rate, RATE), RATE).reshape(-1)
        delays = band_delays(out48, back.numpy())
        out[str(rate)] = summarise([delays], rate)
    print(f"{run_id} fait", flush=True)
    return out


def main() -> None:
    torch.set_num_threads(8)
    result = {"vs100r": {VS100R: vs100r()}, "bigmuff": {r: bigmuff(r) for r in BIGMUFF}}
    v = result["vs100r"][VS100R]
    kappa_v = statistics.median(v[str(r)]["kappa"] for r in RATES)
    kappa_b = statistics.median(
        result["bigmuff"][r][str(rate)]["kappa"] for r in BIGMUFF for rate in RATES
    )
    readings = {
        "R1_form": {
            name: 0.8 <= ck["192000"]["kappa"] / ck["96000"]["kappa"] <= 1.25
            for name, ck in {VS100R: v, **result["bigmuff"]}.items()
        },
        "kappa_vs100r": kappa_v,
        "R2_agrees_with_1.86": abs(kappa_v - 1.86) <= 0.3,
        "kappa_bigmuff_median": kappa_b,
        "R3_architecture": abs(kappa_b - kappa_v) <= 0.2 * abs(kappa_v),
    }
    result["readings"] = readings
    Path(__file__).with_name("kappa.json").write_text(
        json.dumps(result, indent=1) + "\n"
    )
    print(json.dumps(readings, indent=1))


if __name__ == "__main__":
    main()
