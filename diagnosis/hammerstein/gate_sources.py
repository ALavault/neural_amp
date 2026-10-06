#!/usr/bin/env python3
"""Where do the gate's even harmonics come from: the product of two audio-rate signals,
or the constant offsets (biases, FiLM shift)? (NOTE.md, predictions written first.)

For each A48 model, take and sweep level, the gate inputs a and b of every block are
recorded on the sweep (gate_harmonics.record), then the gate output is recomputed in
four variants, locally, without propagating:
  V0 tanh(a) * sigmoid(b)                      as is
  V1 tanh(a) * sigmoid(mean_t b)                no product
  V2 tanh(a - mean_t a) * sigmoid(b - mean_t b) no offset
  V3 tanh(a - mean_t a) * sigmoid(mean_t b)     neither
and D2/D1 of each (median over channels) is measured. Descriptive: the same variants
applied to the whole network (no retraining), output D2/D1 and di_test ESR.
Read-only, CPU. Writes gate_sources.json.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_harmonics as gh  # noqa: E402
import vs100r_hammerstein as vh  # noqa: E402

RUNS = vh.RUNS["A48"]
TAKES = (17, 36, 16)  # 17 and 36 primary (seen pairs, gains 5 and 2.5); 16 descriptive
VARIANTS = ("V0", "V1", "V2", "V3")


def gate(a: torch.Tensor, b: torch.Tensor, variant: str) -> torch.Tensor:
    """a, b: (1, channels, time); means over time, per channel."""
    ma, mb = a.mean(-1, keepdim=True), b.mean(-1, keepdim=True)
    if variant == "V0":
        return torch.tanh(a) * torch.sigmoid(b)
    if variant == "V1":
        return torch.tanh(a) * torch.sigmoid(mb)
    if variant == "V2":
        return torch.tanh(a - ma) * torch.sigmoid(b - mb)
    return torch.tanh(a - ma) * torch.sigmoid(mb)


@torch.no_grad()
def forward(model, x: torch.Tensor, controls: torch.Tensor, variant: str):
    """SSMWaveNet forward with the gate of every block replaced by `variant`."""
    net = model.processor
    net.reset_states()
    films = list(net.film_net(controls).chunk(len(net.blocks), dim=1))
    h = net.input_conv(x)
    skip_sum = torch.zeros_like(h)
    for block, film in zip(net.blocks, films, strict=True):
        g = block.pre_gate(block.ssm(h))
        scale, shift = film.unsqueeze(-1).chunk(2, dim=1)
        a, b = ((1 + scale) * g + shift).chunk(2, dim=1)
        out = gate(a, b, variant)
        skip_sum = skip_sum + block.skip_conv(out)
        h = block.res_conv(out) + h
    z = skip_sum
    for layer in net.output_net:
        z = layer(z)
    return z


def median_db(t: torch.Tensor, ref: np.ndarray) -> float:
    return float(np.median(gh.channels_db(t, ref)))


def local(model, take: dict, level: int) -> dict:
    loop, _ = vh.section(take, f"sweep_{level}dBFS")
    ref = vh.decimate(loop)
    x = torch.tensor(ref, dtype=torch.float32)[None, None]
    controls = torch.tensor([[v / 10 for v in take["gain_contour"]]])
    nodes = gh.record(model, x, controls)
    blocks = len(model.processor.blocks)
    return {
        str(i): {
            v: median_db(gate(nodes[f"b{i}_a"], nodes[f"b{i}_b"], v), ref)
            for v in VARIANTS
        }
        for i in range(blocks)
    }


def whole(model, take: dict) -> dict:
    """Variants over the whole network: output D2/D1 at -10 dBFS and di_test ESR."""
    controls = torch.tensor([[v / 10 for v in take["gain_contour"]]])
    loop, _ = vh.section(take, "sweep_-10dBFS")
    ref = vh.decimate(loop)
    x = torch.tensor(ref, dtype=torch.float32)[None, None]
    model.reset_states()
    direct = model(x, controls)
    assert torch.allclose(direct, forward(model, x, controls, "V0"), atol=1e-6)
    di_loop, di_fx = vh.ev.audio(take)
    di_x = torch.tensor(resample_poly(di_loop, 1, 4), dtype=torch.float32)[None, None]
    target = resample_poly(di_fx, 1, 4)
    out = {}
    for v in VARIANTS:
        y = forward(model, x, controls, v)[0, 0].double().numpy()
        di_y = forward(model, di_x, controls, v)[0, 0].double().numpy()
        out[v] = {
            "output_d2_db": gh.ratio_db(y - y.mean(), ref),
            "di_test_esr": vh.ev.esr(di_y, target),
        }
    return out


def main() -> None:
    torch.set_num_threads(16)
    takes = {t["order"]: t for t in vh.ev.takes() if t["order"] in TAKES}
    result = {"local": {}, "whole": {}}
    for run_id in RUNS:
        model = vh.ev.load(run_id)
        for order in TAKES:
            take = takes[order]
            for level in vh.LEVELS:
                key = f"{run_id}|{order}|{level}"
                result["local"][key] = local(model, take, level)
                b0 = result["local"][key]["0"]
                print(key, " ".join(f"{v} {b0[v]:.1f}" for v in VARIANTS), flush=True)
            result["whole"][f"{run_id}|{order}"] = whole(model, take)
    primary = [
        v for k, v in result["local"].items() if int(k.split("|")[1]) in (17, 36)
    ]
    reading = {v: statistics.median(c["0"][v] for c in primary) for v in VARIANTS}
    delta = {
        v: statistics.median(c["0"][v] - c["0"]["V0"] for c in primary)
        for v in VARIANTS
    }
    control = reading["V3"] <= -60
    d1, d2 = delta["V1"], delta["V2"]
    if not control:
        verdict = "aucune lecture (contrôle V3)"
    elif d1 <= -10 and d2 <= -10:
        verdict = "les deux"
    elif d1 <= -10 and abs(d2) < 3:
        verdict = "produit"
    elif d2 <= -10 and abs(d1) < 3:
        verdict = "décalages"
    else:
        verdict = "interaction ou compensation (rapporté tel quel)"
    result["reading"] = {
        "block0_median_db": reading,
        "block0_delta_db": delta,
        "control_ok": control,
        "verdict": verdict,
        "cases": len(primary),
    }
    print(json.dumps(result["reading"], indent=1, ensure_ascii=False))
    (HERE / "gate_sources.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
