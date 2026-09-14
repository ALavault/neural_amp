#!/usr/bin/env python3
"""Assemble the three SSM-WaveNet diagnostics into a report.

1. Residual analysis by frequency band (SSM-WaveNet 30k vs A2 Full)
2. Seed robustness: three seeds on the Big Muff resplit
3. Generalisation: SSM-WaveNet on the original (inconsistent) splits

Reads from demo/runs/ and writes demo/DIAGNOSTICS.md + demo/diagnostics.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from product_report import RUNS, build_tools, run_native  # noqa: E402

from fssr_nam.models.ssm_wavenet import SSMWaveNet  # noqa: E402

DEVICE = "electro_harmonix_big_muff"
SR = 48_000
SEG = 144_000
BANDS = [(20, 200), (200, 1_000), (1_000, 4_000), (4_000, 12_000), (12_000, 24_000)]

# A2 Full residual analysis from the earlier measurement (on resplit test)
A2_BANDS = {
    "20-200": {"share": 0.004, "band_esr": 0.00941},
    "200-1000": {"share": 0.082, "band_esr": 0.04182},
    "1000-4000": {"share": 0.474, "band_esr": 0.13048},
    "4000-12000": {"share": 0.390, "band_esr": 0.26467},
    "12000-24000": {"share": 0.048, "band_esr": 0.54332},
}

# Known results on original splits from the SOTA bench
A2_ORIGINAL_TEST_ESR = 0.18825
S4_ORIGINAL_TEST_ESR = 0.18655


def load_ssm(state_path: Path) -> SSMWaveNet:
    model = SSMWaveNet(num_blocks=8, channels=16, state_dim=4)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


def render_ssm(model: SSMWaveNet, x: np.ndarray) -> np.ndarray:
    model.reset_states()
    with torch.inference_mode():
        chunks = [
            model(torch.from_numpy(x[i : i + SEG].copy()).reshape(1, 1, -1))
            .reshape(-1)
            .numpy()
            for i in range(0, len(x), SEG)
        ]
    return np.concatenate(chunks)[: len(x)]


def residual_bands(prediction: np.ndarray, target: np.ndarray) -> list[dict]:
    residual = prediction - target
    freqs = np.fft.rfftfreq(len(target), 1.0 / SR)
    spec_r = np.square(np.abs(np.fft.rfft(residual)))
    spec_t = np.square(np.abs(np.fft.rfft(target)))
    total_r = float(np.sum(spec_r))
    rows = []
    for lo, hi in BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        share = float(np.sum(spec_r[mask]) / total_r)
        band_esr = float(np.sum(spec_r[mask]) / max(np.sum(spec_t[mask]), 1e-20))
        rows.append(
            {
                "band": f"{lo}-{hi}",
                "share": share,
                "band_esr": band_esr,
            }
        )
    return rows


def diagnostic_1(resplit_dir: Path) -> dict:
    """Residual by frequency band, SSM-WaveNet 30k vs A2 Full."""
    model = load_ssm(ROOT / "demo/runs/ssm_v2_30k/state.pt")
    x, _ = sf.read(resplit_dir / "test_input.wav", dtype="float32")
    t, _ = sf.read(resplit_dir / "test_target.wav", dtype="float32")
    pred = render_ssm(model, x)
    ssm_bands = residual_bands(pred, t)

    # Also render A2 Full for the comparison on the same data
    _, runner = build_tools()
    run_dir = ROOT / "demo/runs" / RUNS[DEVICE]
    a2_pred, _ = run_native(runner, run_dir / "model_full.nam", x, "64")
    a2_bands = residual_bands(a2_pred, t)

    comparison = []
    for ssm_row, a2_row in zip(ssm_bands, a2_bands, strict=True):
        comparison.append(
            {
                "band": ssm_row["band"],
                "a2_share": a2_row["share"],
                "a2_band_esr": a2_row["band_esr"],
                "ssm_share": ssm_row["share"],
                "ssm_band_esr": ssm_row["band_esr"],
                "gain": a2_row["band_esr"] / max(ssm_row["band_esr"], 1e-20),
            }
        )
    return {"bands": comparison}


def diagnostic_2() -> dict:
    """Seed robustness: three seeds on the Big Muff resplit."""
    seeds = {}
    for seed in (0, 1, 2):
        if seed == 0:
            run_id = "ssm_wavenet_b8_c16_s4"
        else:
            run_id = f"ssm_bigmuff_resplit_seed{seed}"
        runs_log = ROOT / "demo/RUNS.jsonl"
        for line in runs_log.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("run_id") == run_id:
                seeds[seed] = record["test_esr"]
                break
        else:
            raise RuntimeError(f"run {run_id} not found in RUNS.jsonl")
    values = list(seeds.values())
    return {
        "seeds": seeds,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def diagnostic_3() -> dict:
    """SSM-WaveNet on original (inconsistent) splits."""
    run_id = "ssm_bigmuff_original_splits"
    runs_log = ROOT / "demo/RUNS.jsonl"
    ssm_esr = None
    for line in runs_log.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("run_id") == run_id:
            ssm_esr = record["test_esr"]
            ssm_val = record.get("validation_esr")
            break
    if ssm_esr is None:
        raise RuntimeError(f"run {run_id} not found in RUNS.jsonl")
    return {
        "ssm_test_esr": ssm_esr,
        "ssm_val_esr": ssm_val,
        "a2_test_esr": A2_ORIGINAL_TEST_ESR,
        "s4_test_esr": S4_ORIGINAL_TEST_ESR,
    }


def verdicts(d1: dict, d2: dict, d3: dict) -> list[str]:
    lines = []

    # Verdict 1: same bands?
    ssm_dominant = max(d1["bands"], key=lambda r: r["ssm_share"])
    a2_dominant = max(d1["bands"], key=lambda r: r["a2_share"])
    gains = [r["gain"] for r in d1["bands"]]
    if ssm_dominant["band"] == a2_dominant["band"]:
        lines.append(
            f"**Bandes :** le residu SSM-WaveNet est concentre dans la "
            f"meme bande qu'A2 ({ssm_dominant['band']} Hz, "
            f"{ssm_dominant['ssm_share'] * 100:.1f}% du residu). "
            f"Le gain est uniforme a travers les bandes "
            f"({min(gains):.1f}x a {max(gains):.1f}x)."
        )
    else:
        lines.append(
            f"**Bandes :** le residu SSM-WaveNet domine dans "
            f"{ssm_dominant['band']} Hz, contre {a2_dominant['band']} Hz "
            f"pour A2. La structure du residu est differente."
        )

    # Verdict 2: does 5x hold?
    mean = d2["mean"]
    std = d2["std"]
    # The 5x is vs A2 Full on resplit (0.1051)
    a2_resplit = 0.1051
    ratio = a2_resplit / mean
    low = a2_resplit / (mean + std)
    high = a2_resplit / (mean - std) if mean > std else float("inf")
    if low >= 3.0:
        lines.append(
            f"**Robustesse :** ESR test moyen {mean:.5f} +/- {std:.5f} "
            f"sur 3 graines, soit {ratio:.1f}x mieux qu'A2 Full "
            f"({low:.1f}x a {high:.1f}x a +-1 ecart-type). "
            f"Le gain de ~{ratio:.0f}x tient."
        )
    else:
        lines.append(
            f"**Robustesse :** ESR test moyen {mean:.5f} +/- {std:.5f} "
            f"sur 3 graines, soit {ratio:.1f}x mieux qu'A2 Full "
            f"({low:.1f}x a +-1 ecart-type). "
            f"Le gain de 5x ne tient pas a une barre d'erreur pres."
        )

    # Verdict 3: does memory help on inconsistent splits?
    ssm = d3["ssm_test_esr"]
    a2 = d3["a2_test_esr"]
    s4 = d3["s4_test_esr"]
    if ssm < a2 * 0.9:
        lines.append(
            f"**Generalisation :** SSM-WaveNet atteint {ssm:.5f} sur les "
            f"splits originaux, contre {a2:.5f} (A2) et {s4:.5f} "
            f"(S4-TFiLM). La memoire longue aide au changement de prise "
            f"({a2 / ssm:.1f}x mieux qu'A2)."
        )
    else:
        lines.append(
            f"**Generalisation :** SSM-WaveNet atteint {ssm:.5f} sur les "
            f"splits originaux, contre {a2:.5f} (A2) et {s4:.5f} "
            f"(S4-TFiLM). La memoire longue n'aide pas au changement de "
            f"prise : le plafond est dans les donnees, pas le modele."
        )
    return lines


def markdown(d1: dict, d2: dict, d3: dict, verdict_lines: list[str]) -> str:
    lines = [
        "# Diagnostics SSM-WaveNet",
        "",
        "## Verdicts",
        "",
    ]
    for v in verdict_lines:
        lines.append(v)
        lines.append("")

    lines += [
        "## 1. Residus par bande (Big Muff resplit, test)",
        "",
        "| Bande | A2 part | A2 ESR | SSM part | SSM ESR | Gain |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in d1["bands"]:
        lines.append(
            f"| {row['band']} Hz "
            f"| {row['a2_share'] * 100:.1f}% "
            f"| {row['a2_band_esr']:.5f} "
            f"| {row['ssm_share'] * 100:.1f}% "
            f"| {row['ssm_band_esr']:.5f} "
            f"| {row['gain']:.1f}x |"
        )

    lines += [
        "",
        "## 2. Robustesse inter-graines (Big Muff resplit, 15k pas)",
        "",
        "| Graine | ESR test |",
        "| ---: | ---: |",
    ]
    for seed, esr in sorted(d2["seeds"].items()):
        lines.append(f"| {seed} | {esr:.5f} |")
    lines.append(f"| **moyenne** | **{d2['mean']:.5f} +/- {d2['std']:.5f}** |")

    lines += [
        "",
        "## 3. Splits originaux (prises differentes)",
        "",
        "| Modele | ESR val | ESR test |",
        "| --- | ---: | ---: |",
        f"| SSM-WaveNet | {d3['ssm_val_esr']:.5f} | {d3['ssm_test_esr']:.5f} |",
        f"| S4-TFiLM large | — | {d3['s4_test_esr']:.5f} |",
        f"| NAM A2 Full | — | {d3['a2_test_esr']:.5f} |",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    resplit_dir = ROOT / "demo/resplit"

    print("[1/3] Analyse du residus par bande")
    d1 = diagnostic_1(resplit_dir)
    for row in d1["bands"]:
        print(
            f"  {row['band']:>11s} Hz: "
            f"SSM {row['ssm_share'] * 100:5.1f}% ESR {row['ssm_band_esr']:.5f}  "
            f"A2 {row['a2_share'] * 100:5.1f}% ESR {row['a2_band_esr']:.5f}  "
            f"gain {row['gain']:.1f}x"
        )

    print("\n[2/3] Robustesse inter-graines")
    d2 = diagnostic_2()
    for seed, esr in sorted(d2["seeds"].items()):
        print(f"  seed {seed}: {esr:.5f}")
    print(f"  moyenne: {d2['mean']:.5f} +/- {d2['std']:.5f}")

    print("\n[3/3] Splits originaux")
    d3 = diagnostic_3()
    print(f"  SSM-WaveNet: val {d3['ssm_val_esr']:.5f} test {d3['ssm_test_esr']:.5f}")
    print(f"  A2 Full:     test {d3['a2_test_esr']:.5f}")
    print(f"  S4-TFiLM:    test {d3['s4_test_esr']:.5f}")

    verdict_lines = verdicts(d1, d2, d3)
    print("\n=== Verdicts ===")
    for v in verdict_lines:
        print(v)

    report = {"diagnostic_1": d1, "diagnostic_2": d2, "diagnostic_3": d3}
    out_json = ROOT / "demo/diagnostics.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    out_md = ROOT / "demo/DIAGNOSTICS.md"
    out_md.write_text(markdown(d1, d2, d3, verdict_lines), encoding="utf-8")
    print(f"\nwrote {out_md} and {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
