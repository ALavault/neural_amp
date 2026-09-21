#!/usr/bin/env python3
"""What kind of material carries the divergence between two children? (Read-only, CPU.)

The first listening page took three test segments by position and nothing was audible
there; it turned out they were among the least divergent of the twelve. This measures, per
segment, how far the two models stand from each other against how far the control stands
from the device, and relates that to measurable features of the material.

Two things are separated here, because one of them was already refuted for a different
effect. The seed multiplies the error of every segment by one global factor (F-0007, which
replaced an earlier "quiet segments carry the spread" claim, withdrawn in
paper/RESULTS_SSM_SEEDS.md). The question here is not the seed effect but the divergence
between two children of one seed, so the global-factor part is measured and removed before
anything is said about localisation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import product_nablafx_bench as bench  # noqa: E402
from product_fork_pilot_analysis import outputs  # noqa: E402

SAMPLE_RATE = 48_000
CONTROL = "butterfly_ssm_seed42_f100_decide_k0"
CHILD = "butterfly_ssm_seed42_f100_decide_k1"


def features(signal: np.ndarray) -> dict[str, float]:
    peak = float(np.max(np.abs(signal)))
    rms = float(np.sqrt((signal**2).mean()))
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(len(signal), 1 / SAMPLE_RATE)
    # Envelope over 20 ms, to say how much of the segment is decay or silence.
    frame = int(0.020 * SAMPLE_RATE)
    envelope = np.sqrt(
        (signal[: len(signal) // frame * frame] ** 2).reshape(-1, frame).mean(1)
    )
    return {
        "rms_dbfs": 20 * np.log10(rms),
        "crest_db": 20 * np.log10(peak / rms),
        "centroid_hz": float((freqs * spectrum).sum() / spectrum.sum()),
        "quiet_share": float(
            (envelope < peak / 100).mean()
        ),  # below -40 dB of the peak
    }


def main() -> None:
    torch.set_num_threads(6)
    data = bench.data_module("test")
    data.setup("test")
    x = torch.stack([a for a, _ in data.test_dataset])
    target = torch.stack([b for _, b in data.test_dataset]).numpy()[:, 0]
    control = outputs(CONTROL, x).numpy()[:, 0]
    child = outputs(CHILD, x).numpy()[:, 0]

    rows = []
    for i in range(len(target)):
        gain = float(child[i] @ control[i] / (child[i] @ child[i]))
        rows.append(
            {
                "segment": i,
                # Model against model, level matched, so only the shape differs.
                "esr_model_model": float(
                    ((control[i] - gain * child[i]) ** 2).sum()
                    / (control[i] ** 2).sum()
                ),
                "esr_control_device": float(
                    ((control[i] - target[i]) ** 2).sum() / (target[i] ** 2).sum()
                ),
                "esr_child_device": float(
                    ((child[i] - target[i]) ** 2).sum() / (target[i] ** 2).sum()
                ),
                **features(target[i]),
            }
        )
    for row in rows:
        row["share"] = row["esr_model_model"] / row["esr_control_device"]
        row["child_over_control"] = row["esr_child_device"] / row["esr_control_device"]

    print(
        "segment  ESR modele-modele  part de l'erreur au reel  enfant/temoin"
        "  RMS dBFS  crete dB  centroide Hz  part calme"
    )
    for r in rows:
        print(
            f"   {r['segment']:2d}        {r['esr_model_model']:.4f}"
            f"            {r['share']:.2f}              {r['child_over_control']:.2f}"
            f"       {r['rms_dbfs']:6.1f}   {r['crest_db']:5.1f}     {r['centroid_hz']:7.0f}"
            f"      {r['quiet_share']:.2f}"
        )

    # Is the child's excess a single global factor, as the seed effect is, or localised?
    ratios = np.array([r["child_over_control"] for r in rows])
    print(
        f"\nenfant/temoin par segment: mediane {np.median(ratios):.2f},"
        f" min {ratios.min():.2f}, max {ratios.max():.2f},"
        f" ecart-type du log {np.std(np.log(ratios), ddof=1):.3f}"
    )

    from scipy import stats

    print("\ncorrelation de rang avec la part de l'erreur au reel (n = 12):")
    for key in ("rms_dbfs", "crest_db", "centroid_hz", "quiet_share"):
        r = stats.spearmanr([row[key] for row in rows], [row["share"] for row in rows])
        print(f"  {key:12s} rho = {r.statistic:+.2f}  p = {r.pvalue:.3f}")

    (Path(__file__).parent / "where_audible.json").write_text(
        json.dumps(rows, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
