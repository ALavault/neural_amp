#!/usr/bin/env python3
"""Evaluate the first VS100R model against the raw 192 kHz takes. (Product lane, CPU.)

Measures of demo/capture/valvestate/predictions_model.md, written before the run.
Per OD2 take (Power Dimension off), on the 30 s test section of that take: the input is
its own loopback, the reference its FX loop send (spikes repaired as in the dataset).
- At 48 kHz, both decimated by 4: ESR of the model (P1 to P4).
- At 96 and 192 kHz, on the native loopback: dt rescaled (`delta`), then read as first
  order hold and rescaled (`foh`). The output is decimated to 48 kHz and scored against
  the device (P5), against the 48 kHz model output (P6). Its share of power between
  24 kHz and its Nyquist frequency is set beside the device's in the same band, on the
  same scored window, the device's noise (its silence section) subtracted (P7).
Each section runs in one piece from a zero state; the first 0.5 s and the last 0.1 s are
not scored (29.4 s per take).

Usage: product_vs100r_eval.py [--run-id RUN] [--dry-run]
--dry-run: untrained model, two takes, output to the scratch directory; checks the
pipeline, measures nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly, welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import product_nablafx_bench as bench  # noqa: E402
import product_sr_transfer as transfer  # noqa: E402
from nablafx.core import BlackBoxModel  # noqa: E402

from fssr_nam.models.ssm_wavenet import zoh_to_foh  # noqa: E402

SHEET = ROOT / "demo/capture/valvestate"
MEDIA = ROOT / "datasets/amp_capture/Media"
LAYOUT = json.loads((SHEET / "layout.json").read_text())
SECTION = next(s for s in LAYOUT["sections"] if s["section"] == "di_test")
SILENCE = next(s for s in LAYOUT["sections"] if s["section"] == "silence")
CAPTURE = LAYOUT["sample_rate"]
RATES = (96_000, 192_000)
DRY_SSM = {
    "num_blocks": 8,
    "channels": 16,
    "state_dim": 4,
    "output_act": "tanh",
    "discretization": "zoh",
    "cond": "film",
    "cond_head": "joint",
    "cond_hidden": [32],
}


def load(run_id: str | None) -> BlackBoxModel:
    ssm, model = DRY_SSM, "ssm-wavenet"
    if run_id:
        record = json.loads((bench.RESULTS_DIR / f"{run_id}.json").read_text())
        ssm, model = record["ssm"] or {}, record["model"]
    args = argparse.Namespace(model=model, device="vs100r", **ssm)
    processor = bench.build_processor(args)
    if run_id:
        path = bench.RUNS_DIR / f"nablafx_{run_id}/checkpoints/last.ckpt"
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        prefix = "model.processor."
        processor.load_state_dict(
            {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        )
    return BlackBoxModel(processor).eval()


def takes() -> list[dict]:
    passes = {p["order"]: p for p in json.loads((SHEET / "passes.json").read_text())}
    qc = {r["order"]: r for r in json.loads((SHEET / "qc.json").read_text())}
    with (SHEET / "settings.csv").open() as stream:
        rows = [
            r
            for r in csv.DictReader(stream)
            if r["channel"] == "OD2" and r["power_dimension"] == "off"
        ]
    pair = {int(r["order"]): (float(r["G"]), float(r["C"])) for r in rows}
    trained = [pair[int(r["order"])] for r in rows if r["split"] == "train"]
    out = []
    for r in rows:
        order = int(r["order"])
        out.append(
            {
                "order": order,
                "setting": r["setting"],
                "split": "test" if r["split"] == "test" else "test_seen",
                "gain_contour": pair[order],
                "takes_same_pair_in_training": trained.count(pair[order]),
                "low_signal": qc[order]["low_signal"],
                "pass": passes[order],
                "spikes": qc[order]["spikes_wav_sample"],
            }
        )
    return out


def audio(take: dict) -> tuple[np.ndarray, np.ndarray]:
    p = take["pass"]
    a = p["start"] + SECTION["start"]
    b = a + SECTION["samples"]
    stamp = p["file"].removeprefix("02-Looopbck-")
    loop = sf.read(MEDIA / p["file"], dtype="float64", start=a, stop=b)[0]
    fx = sf.read(MEDIA / f"03-FX Out-{stamp}", dtype="float64", start=a, stop=b)[0]
    for i in take["spikes"]:
        if a < i < b - 1:
            fx[i - a] = 0.5 * (fx[i - a - 1] + fx[i - a + 1])
    return loop, fx


@torch.no_grad()
def run(model, x: np.ndarray, controls: torch.Tensor) -> np.ndarray:
    model.reset_states()
    signal = torch.from_numpy(x.astype(np.float32))[None, None]
    return model(signal, controls[None]).numpy()[0, 0].astype(np.float64)


def scored(x: np.ndarray, rate: int = 48_000) -> np.ndarray:
    return x[rate // 2 : -rate // 10]


def esr(pred: np.ndarray, ref: np.ndarray) -> float:
    p, r = scored(pred), scored(ref)
    return float(((r - p) ** 2).sum() / (r**2).sum())


def band_share_db(
    x: np.ndarray, rate: int, high: float, noise: np.ndarray | None = None
) -> float:
    """Share of power between 24 kHz and `high`, noise PSD subtracted if given."""
    freqs, power = welch(scored(x, rate), rate, nperseg=8192)
    if noise is not None:
        power = np.maximum(power - noise, 0.0)
    band = (freqs >= 24_000) & (freqs < high)
    return float(10 * np.log10(power[band].sum() / power.sum() + 1e-30))


def device_noise(take: dict) -> np.ndarray:
    p = take["pass"]
    a = p["start"] + SILENCE["start"] + CAPTURE // 10
    b = p["start"] + SILENCE["start"] + SILENCE["samples"] - CAPTURE // 10
    stamp = p["file"].removeprefix("02-Looopbck-")
    x = sf.read(MEDIA / f"03-FX Out-{stamp}", dtype="float64", start=a, stop=b)[0]
    return welch(x, CAPTURE, nperseg=8192)[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="vs100r_gc_film_joint_s925")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(16)
    model = load(None if args.dry_run else args.run_id)
    chosen = takes()
    if args.dry_run:
        chosen = [t for t in chosen if t["order"] in (1, 16)]
    base = {
        id(m): (m.log_dt.detach().clone(), m.C.detach().clone())
        for m in transfer.ssm_modules(model)
    }
    rows = []
    for take in chosen:
        loop, fx = audio(take)
        controls = torch.tensor([v / 10 for v in take["gain_contour"]])
        x48, y48 = resample_poly(loop, 1, 4), resample_poly(fx, 1, 4)
        transfer.set_rate(model, 48_000, base)
        m48 = run(model, x48, controls)
        row = {k: v for k, v in take.items() if k not in ("pass", "spikes")}
        row["esr_48k"] = esr(m48, y48)
        noise = device_noise(take)
        for high in (48_000, 96_000):
            row[f"device_24k_{high // 1000}k_db"] = band_share_db(
                fx, CAPTURE, high, noise
            )
        rows.append(row)
    for variant in ("delta", "foh"):
        if variant == "foh":
            transfer.set_rate(model, 48_000, base)
            zoh_to_foh(model)
        current = {
            id(m): (m.log_dt.detach().clone(), m.C.detach().clone())
            for m in transfer.ssm_modules(model)
        }
        for take, row in zip(chosen, rows, strict=True):
            loop, fx = audio(take)
            controls = torch.tensor([v / 10 for v in take["gain_contour"]])
            y48 = resample_poly(fx, 1, 4)
            transfer.set_rate(model, 48_000, current)
            m48 = run(model, resample_poly(loop, 1, 4), controls)
            for rate in RATES:
                transfer.set_rate(model, rate, current)
                x = resample_poly(loop, 1, CAPTURE // rate)
                m = run(model, x, controls)
                back = resample_poly(m, 1, rate // 48_000)
                key = f"{variant}_{rate // 1000}k"
                row[f"{key}_esr_vs_device"] = esr(back, y48)
                row[f"{key}_ratio_R"] = row[f"{key}_esr_vs_device"] / row["esr_48k"]
                row[f"{key}_vs_48k_model"] = esr(back, m48)
                row[f"{key}_above_24k_db"] = band_share_db(m, rate, rate / 2)
            print(
                f"{variant} prise {row['order']:>2} ESR48 {row['esr_48k']:.4f}"
                f"  R192 {row[f'{variant}_192k_ratio_R']:.3f}"
                f"  ecart au 48k {row[f'{variant}_192k_vs_48k_model']:.4f}",
                flush=True,
            )
    summary = summarise(rows)
    print(json.dumps(summary, indent=1))
    out = (
        Path("/fastdata/lavaulta/tmp/vs100r_eval_dry.json")
        if args.dry_run
        else SHEET / f"eval_{args.run_id}.json"
    )
    out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1) + "\n")


def summarise(rows: list[dict]) -> dict:
    def median(values: list[float]) -> float | None:
        return statistics.median(values) if values else None

    kept = [r for r in rows if not r["low_signal"]]
    seen = [r for r in kept if r["split"] == "test_seen"]
    summary = {
        "P1_median_esr_test": median(
            [r["esr_48k"] for r in kept if r["split"] == "test"]
        ),
        "P2_median_seen_pair_55": median(
            [r["esr_48k"] for r in seen if r["gain_contour"] == (5.0, 5.0)]
        ),
        "P2_median_seen_alone": median(
            [r["esr_48k"] for r in seen if r["takes_same_pair_in_training"] == 1]
        ),
        "P3_esr_gain0": {
            r["order"]: r["esr_48k"] for r in rows if r["gain_contour"][0] == 0
        },
        "P4_take16": next((r["esr_48k"] for r in rows if r["order"] == 16), None),
    }
    for variant in ("delta", "foh"):
        for rate in RATES:
            key = f"{variant}_{rate // 1000}k"
            if f"{key}_ratio_R" not in rows[0]:
                continue
            ratio = median([r[f"{key}_ratio_R"] for r in kept])
            summary[f"P5_{key}_median_R"] = ratio
            summary[f"P6_{key}_median_vs_48k"] = median(
                [r[f"{key}_vs_48k_model"] for r in kept]
            )
            summary[f"P7_{key}_median_above_24k_db"] = median(
                [r[f"{key}_above_24k_db"] for r in kept]
            )
    for high in (48_000, 96_000):
        key = f"device_24k_{high // 1000}k_db"
        summary[f"P7_{key}_median"] = median([r[key] for r in kept])
    ratio = summary.get("P5_foh_192k_median_R")
    if ratio is not None:
        summary["P5_outcome"] = (
            "192k plus proche de l'appareil"
            if ratio <= 0.95
            else "indiscernable"
            if ratio < 1.05
            else "192k ajoute de l'erreur"
        )
    return summary


if __name__ == "__main__":
    main()
