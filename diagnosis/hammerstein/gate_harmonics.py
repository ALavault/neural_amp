#!/usr/bin/env python3
"""Where do the models' extra even harmonics come from? (NOTE.md, exploration)

One trained model (default vs100r_gc_film_joint_s925_r2), one take's sweeps at the three
levels. Every internal signal of the network is recorded on the sweep: per block the
SSM output, the two gate inputs a and b (1x1 conv + FiLM), the gate output
tanh(a) * sigmoid(b), the skip and the residual; then the head (skip sum, after each
LeakyReLU, final output). For each signal and channel: energy of the 2nd harmonic
response over the 1st (D2/D1, dB, output frequencies 200 Hz - 8 kHz), by the sweep
deconvolution of src/fssr_nam/product/hammerstein.py. The gate's own contribution is
the rise of D2/D1 from its inputs to its output. Read-only, CPU. Writes
gate_harmonics_<run>_take<n>.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vs100r_hammerstein as vh  # noqa: E402

from fssr_nam.product.hammerstein import deconvolve, harmonic_spectra  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "vs100r_gc_film_joint_s925_r2"
TAKE = int(sys.argv[2]) if len(sys.argv) > 2 else 17
NFFT = 8192


def ratio_db(signal: np.ndarray, ref: np.ndarray) -> float:
    """D2/D1 energy ratio in dB over 200 Hz - 8 kHz (output frequency)."""
    d = harmonic_spectra(
        deconvolve(signal, ref, vh.RATE, vh.SWEEP), vh.RATE, vh.SWEEP, 2, NFFT
    )
    freqs = np.fft.rfftfreq(NFFT, 1 / vh.RATE)
    band = (freqs >= 200) & (freqs <= 8000)
    e1 = (np.abs(d[0][band]) ** 2).sum()
    e2 = (np.abs(d[1][band]) ** 2).sum()
    return float(10 * np.log10(e2 / (e1 + 1e-300) + 1e-300))


def channels_db(t: torch.Tensor, ref: np.ndarray) -> list[float]:
    """t: (1, channels, length) -> D2/D1 per channel."""
    x = t[0].double().numpy()
    return [ratio_db(ch - ch.mean(), ref) for ch in x]


@torch.no_grad()
def record(model, x: torch.Tensor, controls: torch.Tensor) -> dict[str, torch.Tensor]:
    """Forward pass of SSMWaveNet with every intermediate kept."""
    net = model.processor
    out = {}
    net.reset_states()
    films = list(net.film_net(controls).chunk(len(net.blocks), dim=1))
    h = net.input_conv(x)
    out["input_conv"] = h
    skip_sum = torch.zeros_like(h)
    for i, (block, film) in enumerate(zip(net.blocks, films, strict=True)):
        s = block.ssm(h)
        g = block.pre_gate(s)
        scale, shift = film.unsqueeze(-1).chunk(2, dim=1)
        g = (1 + scale) * g + shift
        a, b = g.chunk(2, dim=1)
        gate = torch.tanh(a) * torch.sigmoid(b)
        skip = block.skip_conv(gate)
        h = block.res_conv(gate) + h
        skip_sum = skip_sum + skip
        out.update(
            {
                f"b{i}_ssm": s,
                f"b{i}_a": a,
                f"b{i}_b": b,
                f"b{i}_gate": gate,
                f"b{i}_skip": skip,
                f"b{i}_res": h,
            }
        )
    out["skip_sum"] = skip_sum
    z = skip_sum
    for j, layer in enumerate(net.output_net):
        z = layer(z)
        out[f"head{j}_{type(layer).__name__}"] = z
    return out


def main() -> None:
    torch.set_num_threads(16)
    model = vh.ev.load(RUN)
    take = next(t for t in vh.ev.takes() if t["order"] == TAKE)
    controls = torch.tensor([[v / 10 for v in take["gain_contour"]]])
    result = {"run": RUN, "take": TAKE, "levels": {}}
    for level in vh.LEVELS:
        loop, fx = vh.section(take, f"sweep_{level}dBFS")
        ref = vh.decimate(loop)
        x = torch.tensor(ref, dtype=torch.float32)[None, None]
        nodes = record(model, x, controls)
        # Check the instrumented pass against the model's own forward.
        model.reset_states()
        direct = model(x, controls)
        assert torch.allclose(direct, nodes[list(nodes)[-1]], atol=1e-6)
        row = {"device_fx": ratio_db(vh.decimate(fx), ref)}
        for name, t in nodes.items():
            values = channels_db(t, ref)
            row[name] = {
                "median": float(np.median(values)),
                "max": float(np.max(values)),
            }
        result["levels"][str(level)] = row
        gates = [
            (
                row[f"b{i}_gate"]["median"],
                max(row[f"b{i}_a"]["median"], row[f"b{i}_b"]["median"]),
            )
            for i in range(len(model.processor.blocks))
        ]
        print(
            level,
            "appareil %.1f dB ; sortie modèle %.1f dB"
            % (row["device_fx"], row[list(nodes)[-1]]["median"]),
            flush=True,
        )
        for i, (g, inp) in enumerate(gates):
            print(
                f"  bloc {i}: entrées a/b max {inp:.1f} dB -> porte {g:.1f} dB",
                flush=True,
            )
        for name in [n for n in nodes if n.startswith(("skip_sum", "head"))]:
            print(f"  {name}: {row[name]['median']:.1f} dB", flush=True)
    (HERE / f"gate_harmonics_{RUN}_take{TAKE}.json").write_text(
        json.dumps(result, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
