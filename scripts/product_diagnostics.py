#!/usr/bin/env python3
"""Assemble the three SSM-WaveNet diagnostics into a report.

1. Residual analysis by frequency band (SSM-WaveNet 30k vs A2 Full)
2. Seed robustness: three seeds at the same configuration (lr 0.01) on the
   Big Muff resplit, diverged runs included
3. SSM-WaveNet on the M4 splits of the published files (truncated to
   120/30/30 s), every attempt listed

Reads from demo/runs/ and demo/RUNS.jsonl, writes demo/DIAGNOSTICS.md and
demo/diagnostics.json.
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

# Test ESR on the same truncated M4 splits, from demo/sota_comparison.json
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


def runs(run_id: str) -> list[dict]:
    lines = (ROOT / "demo/RUNS.jsonl").read_text(encoding="utf-8").splitlines()
    return [r for r in map(json.loads, lines) if r.get("run_id") == run_id]


def diagnostic_2() -> dict:
    """Seed robustness: seeds 0, 1 and 2 at lr 0.01 and 15k steps."""
    run_ids = {
        0: "ssm_wavenet_b8_c16_s4",
        1: "ssm_bigmuff_resplit_seed1",
        2: "ssm_bigmuff_resplit_seed2",
    }
    seeds = {}
    for seed, run_id in run_ids.items():
        (record,) = [r for r in runs(run_id) if r.get("lr", 0.01) == 0.01]
        if record["steps"] != 15_000:
            raise RuntimeError(f"{run_id} has {record['steps']} steps")
        seeds[seed] = {
            "run_id": run_id,
            "test_esr": record["test_esr"],
            "status": record.get("status", "converged"),
        }
    values = [s["test_esr"] for s in seeds.values()]
    converged = [s["test_esr"] for s in seeds.values() if s["status"] == "converged"]
    (rerun,) = [r for r in runs("ssm_bigmuff_resplit_seed2") if r.get("lr") == 0.005]
    return {
        "lr": 0.01,
        "seeds": seeds,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "converged": len(converged),
        "converged_mean": float(np.mean(converged)),
        "seed2_lr0005_test_esr": rerun["test_esr"],
    }


def diagnostic_3() -> dict:
    """SSM-WaveNet on the truncated M4 splits: every attempt."""
    attempts = [
        {
            "lr": r["lr"],
            "status": r["status"],
            "steps": r.get("steps"),
            "test_esr": r.get("test_esr"),
            "best_validation_esr": r.get("best_validation_esr"),
        }
        for r in runs("ssm_bigmuff_original_splits")
    ]
    return {
        "data": "published Big Muff files truncated to 120/30/30 s",
        "ssm_attempts": attempts,
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

    # Verdict 2: does 5x hold? Resplit A2 Full test ESR from RESPLIT_COMPARISON.md.
    mean, std = d2["mean"], d2["std"]
    a2_resplit = 0.10513
    if d2["converged"] == 3 and a2_resplit / (mean + std) >= 3.0:
        lines.append(
            f"**Robustesse :** ESR test moyen {mean:.5f} +/- {std:.5f} sur 3 graines "
            f"a lr 0,01, soit {a2_resplit / mean:.1f}x mieux qu'A2 Full. Le gain tient."
        )
    else:
        lines.append(
            f"**Robustesse :** a lr 0,01, {d2['converged']} graines sur 3 convergent "
            f"(ESR test moyen des convergees {d2['converged_mean']:.5f}) ; la graine 2 "
            f"diverge (ESR test {d2['seeds'][2]['test_esr']:.4f}). Sur les 3 graines, "
            f"ESR moyen {mean:.4f} +/- {std:.4f}. Le 5x ne tient pas : le modele "
            f"n'est pas stable a ce lr. Relancee a lr 0,005, la graine 2 atteint "
            f"{d2['seed2_lr0005_test_esr']:.5f}, mais ce n'est plus la meme "
            "configuration."
        )

    # Verdict 3: all attempts on the truncated M4 splits.
    a2, s4 = d3["a2_test_esr"], d3["s4_test_esr"]
    tested = [a["test_esr"] for a in d3["ssm_attempts"] if a["test_esr"] is not None]
    if tested and min(tested) < a2 * 0.9:
        lines.append(
            f"**Splits M4 :** SSM-WaveNet atteint {min(tested):.5f} contre "
            f"{a2:.5f} (A2) "
            f"et {s4:.5f} (S4-TFiLM)."
        )
    else:
        lines.append(
            f"**Splits M4 :** aucune des {len(d3['ssm_attempts'])} tentatives de "
            f"SSM-WaveNet ne converge (voir tableau), contre {a2:.5f} (A2) et "
            f"{s4:.5f} (S4-TFiLM) sur les memes fichiers. Le 5x ne tient pas hors de "
            f"la prise d'entrainement dans cette boucle. Ces fichiers sont tronques a "
            f"120/30/30 s : la comparaison au protocole publie est dans "
            f"`demo/nablafx_bench/`."
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
        "## 2. Robustesse inter-graines (Big Muff resplit, lr 0,01, 15k pas)",
        "",
        "| Graine | Run | Statut | ESR test |",
        "| ---: | --- | --- | ---: |",
    ]
    for seed, row in sorted(d2["seeds"].items()):
        lines.append(
            f"| {seed} | `{row['run_id']}` | {row['status']} | {row['test_esr']:.5f} |"
        )
    lines.append(f"| **moyenne** | | | **{d2['mean']:.5f} +/- {d2['std']:.5f}** |")
    lines += [
        "",
        f"Hors configuration : graine 2 relancee a lr 0,005, ESR test "
        f"{d2['seed2_lr0005_test_esr']:.5f}.",
        "",
        "## 3. Splits M4 (fichiers publies tronques a 120/30/30 s)",
        "",
        "| Modele | lr | Statut | Pas | ESR test | Meilleure ESR val |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for a in d3["ssm_attempts"]:
        steps = "?" if a["steps"] is None else str(a["steps"])
        test = "non teste" if a["test_esr"] is None else f"{a['test_esr']:.5f}"
        val = (
            "?"
            if a["best_validation_esr"] is None
            else f"{a['best_validation_esr']:.5f}"
        )
        lines.append(
            f"| SSM-WaveNet | {a['lr']} | {a['status']} | {steps} | {test} | {val} |"
        )
    lines += [
        f"| S4-TFiLM large | 0.01 | converged | 15000 | {d3['s4_test_esr']:.5f} | |",
        f"| NAM A2 Full | | converged | | {d3['a2_test_esr']:.5f} | |",
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
    for seed, row in sorted(d2["seeds"].items()):
        print(f"  seed {seed}: {row['test_esr']:.5f} ({row['status']})")
    print(f"  moyenne: {d2['mean']:.5f} +/- {d2['std']:.5f}")

    print("\n[3/3] Splits M4 tronques")
    d3 = diagnostic_3()
    for a in d3["ssm_attempts"]:
        print(f"  SSM-WaveNet lr {a['lr']}: {a['status']}, test {a['test_esr']}")
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
