#!/usr/bin/env python3
"""Is the gap between two children just an EQ difference? (Read-only, CPU.)

The listener reports hearing no difference except, sometimes, slightly more low end, half
a decibel at most. That is a testable claim with a sharp consequence: if what separates the
two models is a static frequency response, the divergence of ESR is perceptually cheap and
one filter removes it. If it is not, the models behave differently.

Measured here, on the whole test set and on the three listening excerpts:

- |H(f)|, the least-squares linear filter from the child's output to the control's, in dB
  per third octave: the EQ difference the ear may be picking up;
- the magnitude-squared coherence, which says how much of the gap a static filter of any
  shape can remove. The residual 1 - coherence is what no EQ can reach.

No filtering is actually applied: the residual energy of the best linear filter is
S_yy (1 - coherence), so the ratio follows from the spectra.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import signal as dsp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from product_fork_pilot_analysis import outputs  # noqa: E402

import product_nablafx_bench as bench  # noqa: E402

SAMPLE_RATE = 48_000
NPERSEG = 8192
CONTROL = "butterfly_ssm_seed42_f100_decide_k0"
CHILDREN = {
    "decide_k1": "butterfly_ssm_seed42_f100_decide_k1",
    "rejoue_k1": "butterfly_ssm_seed42_f100_replay_k1",
}
THIRDS = 1000 * 2.0 ** (np.arange(-10, 7) / 3)  # 100 Hz to 6.3 kHz


def analyse(child: np.ndarray, control: np.ndarray) -> dict:
    kwargs = dict(fs=SAMPLE_RATE, nperseg=NPERSEG, noverlap=NPERSEG // 2)
    f, sxx = dsp.welch(child, **kwargs)
    _, syy = dsp.welch(control, **kwargs)
    _, sxy = dsp.csd(child, control, **kwargs)
    coherence = np.abs(sxy) ** 2 / (sxx * syy)
    response = np.abs(sxy) / sxx  # |H(f)| of the best linear filter, child -> control
    # Weight by the control's own spectrum: what the ear meets, not what is empty.
    weight = syy / syy.sum()
    bands = {}
    for low, high in zip(THIRDS, THIRDS[1:]):
        inside = (f >= low) & (f < high)
        if inside.any():
            bands[f"{low:.0f}"] = float(
                20 * np.log10((response[inside] * weight[inside]).sum() / weight[inside].sum())
            )
    # Gain first, so the residual is not inflated by a plain level difference.
    gain = float(child @ control / (child @ child))
    raw = float(((control - gain * child) ** 2).sum() / (control**2).sum())
    return {
        "esr_raw": raw,
        "esr_after_best_filter": float((syy * (1 - coherence)).sum() / syy.sum()),
        "bands_db": bands,
        "tilt_db": float(
            np.mean([v for k, v in bands.items() if float(k) < 300])
            - np.mean([v for k, v in bands.items() if float(k) > 2000])
        ),
    }


def main() -> None:
    torch.set_num_threads(6)
    data = bench.data_module("test")
    data.setup("test")
    x = torch.stack([a for a, _ in data.test_dataset])
    control = outputs(CONTROL, x).numpy()[:, 0].reshape(-1)
    out = {}
    for name, run in CHILDREN.items():
        child = outputs(run, x).numpy()[:, 0].reshape(-1)
        out[name] = analyse(child, control)
        r = out[name]
        print(
            f"{name}: ESR brut {r['esr_raw']:.4f}, apres le meilleur filtre lineaire"
            f" {r['esr_after_best_filter']:.4f}"
            f" ({r['esr_after_best_filter'] / r['esr_raw']:.0%} du brut),"
            f" pente grave-aigu {r['tilt_db']:+.2f} dB"
        )
        print(
            "   "
            + "  ".join(
                f"{k}Hz:{v:+.2f}"
                for k, v in list(r["bands_db"].items())[::3]
            )
        )
    (Path(__file__).parent / "is_it_eq.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
