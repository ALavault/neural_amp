#!/usr/bin/env python3
"""What do the fast modes of arm D carry? (predictions_fast_modes.md) Read-only, CPU.

Pre-registered 2026-10-02 (commit 2b447f3) before any measurement. Per D checkpoint:
fast modes (s > 10 for at least one training setting) and switched ones (min s < 1);
C1, unseen ESR with each fast mode's C set to zero; C2, D + K[0] per channel over the
28 training settings and its range along each knob's axis; C3, each knob's share of
the pre-tanh pole shift of each switched mode (the additive head sums one-knob terms).

Usage: fast_modes.py --run <run_id>   (writes a part under /fastdata/lavaulta/tmp)
       fast_modes.py --merge          (writes demo/param/fast_modes.json)
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "demo/param"))
pv = importlib.import_module("poles_verdict")
bench = importlib.import_module("product_nablafx_bench")

RUNS = [
    "g2_poles_additive_s925",
    "g2_poles_additive_s736",
    "g2_poles_additive_s688_r2",
    "g2_poles_additive_s744",
    "g2_poles_additive_s369",
]
KNOBS = ("B", "M", "T", "G")
PARTS = Path("/fastdata/lavaulta/tmp/fast_modes_parts")
CENTRE = 0.5


def training_settings() -> torch.Tensor:
    data = bench.data_module("trainval", 16, "marshall")
    data.setup("fit")
    settings = sorted(
        {tuple(s["params"].tolist()) for s in data.train_dataset.dataset.samples}
    )
    return torch.tensor(settings)


def axes(controls: torch.Tensor) -> dict[str, list[int]]:
    """Settings on each knob's training axis: the other knobs at the centre."""
    out = {}
    for k, name in enumerate(KNOBS):
        others = [j for j in range(len(KNOBS)) if j != k]
        out[name] = [
            i
            for i, u in enumerate(controls)
            if all(abs(float(u[j]) - CENTRE) < 1e-6 for j in others)
        ]
    return out


def unseen_esr(model, dataset) -> dict[str, float]:
    scores = pv.unseen(model, dataset)
    return {k: v["esr"] for k, v in scores.items() if "@" not in k}


@torch.no_grad()
def analyse(run: str) -> dict:
    torch.set_num_threads(6)
    model = pv.load(run)
    processor = model.processor
    controls = training_settings()
    on_axis = axes(controls)
    shifts = processor.pole_shifts(controls)
    head = processor.pole_net
    widths = [layer.out_features for layer in head.inputs]
    starts = [sum(widths[:k]) for k in range(len(widths))]
    pre = {}  # knob -> (settings, outputs) pre-tanh contribution of that knob
    for k, layer in enumerate(head.inputs):
        hidden = torch.relu(layer(controls[:, k : k + 1]))
        weight = head.output.weight[:, starts[k] : starts[k] + widths[k]]
        pre[KNOBS[k]] = hidden @ weight.T
    blocks = processor.blocks
    channels, modes = blocks[0].ssm.channels, blocks[0].ssm.state_dim
    fast, k0_rows = [], []
    for b, (block, (dlog_dt, dlog_a)) in enumerate(zip(blocks, shifts, strict=True)):
        ssm = block.ssm
        _, now, _ = ssm._shifted_inputs((dlog_dt, dlog_a))
        c = ssm._complex(ssm.C)
        k0 = ssm.D.unsqueeze(0) + 2 * (c.unsqueeze(0) * now).real.sum(-1)
        speed = torch.exp(ssm.log_A_real + dlog_a) * torch.exp(
            ssm.log_dt + dlog_dt
        ).unsqueeze(-1)
        top, low = speed.max(0).values, speed.min(0).values
        for ch, m in (top > 10).nonzero().tolist():
            switched = bool(low[ch, m] < 1)
            entry = {
                "block": b,
                "channel": ch,
                "mode": m,
                "s_max": float(top[ch, m]),
                "s_min": float(low[ch, m]),
                "switched": switched,
            }
            if switched:
                index = b * channels * (1 + modes) + channels + ch * modes + m
                ranges = {
                    name: float(
                        pre[name][on_axis[name], index].max()
                        - pre[name][on_axis[name], index].min()
                    )
                    for name in KNOBS
                }
                total = sum(ranges.values())
                entry["pre_tanh_range"] = ranges
                entry["share"] = {n: v / total for n, v in ranges.items()}
            fast.append(entry)
        for ch in range(channels):
            values = k0[:, ch]
            k0_rows.append(
                {
                    "block": b,
                    "channel": ch,
                    "range_by_knob": {
                        n: float(values[on_axis[n]].max() - values[on_axis[n]].min())
                        for n in KNOBS
                    },
                }
            )
    data = bench.data_module("test", device="marshall")
    data.setup("test")
    base = unseen_esr(model, data.test_dataset)
    print(f"{run}: base {statistics.median(base.values()):.4f}", flush=True)
    for entry in fast:
        ssm = blocks[entry["block"]].ssm
        saved = ssm.C.data[entry["channel"], entry["mode"]].clone()
        ssm.C.data[entry["channel"], entry["mode"]] = 0
        ablated = unseen_esr(model, data.test_dataset)
        ssm.C.data[entry["channel"], entry["mode"]] = saved
        log_r = statistics.median(math.log(ablated[k] / base[k]) for k in base)
        entry["median_log_ratio_unseen"] = log_r
        entry["reading"] = (
            "mort"
            if abs(log_r) < 0.01
            else "porteur"
            if log_r >= math.log(1.1)
            else "mineur"
        )
        print(f"  {entry['block']}/{entry['channel']}/{entry['mode']}: {log_r:+.4f}")
    hosting = {(e["block"], e["channel"]) for e in fast if e["switched"]}
    return {
        "run": run,
        "unseen_base": base,
        "fast_modes": fast,
        "k0_switched_channels": [
            r for r in k0_rows if (r["block"], r["channel"]) in hosting
        ],
    }


def merge() -> None:
    parts = {r: json.loads((PARTS / f"{r}.json").read_text()) for r in RUNS}
    switched = [e for p in parts.values() for e in p["fast_modes"] if e["switched"]]
    k0 = [row for p in parts.values() for row in p["k0_switched_channels"]]
    g_shares = [e["share"]["G"] >= 0.5 for e in switched]
    g_dominates = [
        r["range_by_knob"]["G"]
        >= 2 * max(r["range_by_knob"][n] for n in ("B", "M", "T"))
        for r in k0
    ]
    readings = [e["reading"] for p in parts.values() for e in p["fast_modes"]]
    summary = {
        "fast_modes": len(readings),
        "readings": {x: readings.count(x) for x in ("mort", "mineur", "porteur")},
        "switched_modes": len(switched),
        "c3_g_share_ge_half": sum(g_shares),
        "c3_commanded_by_g": sum(g_shares) >= 2 / 3 * len(switched),
        "switched_channels": len(k0),
        "c2_g_dominates": sum(g_dominates),
        "c2_label": sum(g_dominates) >= 2 / 3 * len(k0),
    }
    out = {"summary": summary, "runs": parts}
    Path(__file__).with_name("fast_modes.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )
    print(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=RUNS)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merge()
        return
    PARTS.mkdir(parents=True, exist_ok=True)
    result = analyse(args.run)
    (PARTS / f"{args.run}.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
