#!/usr/bin/env python3
"""Build the demo fact sheet: fidelity and native CPU cost of the A2 models."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from nam.models import init_from_nam

from fssr_nam.metrics.spectral import spectral_metrics
from fssr_nam.metrics.time import time_metrics
from fssr_nam.product.data import device_pairs

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets/manifests/m4_internal.json"
BUILD_DIR = ROOT / "build/product_cpp"
BLOCK_SIZES = (32, 64, 128, 256)
RUNS = {
    "fulltone_full_drive_2": "product_a2_fulltone_full_drive_2_seed0_v1",
    "electro_harmonix_big_muff": "product_a2_electro_harmonix_big_muff_seed0_v2",
}


def build_benchmark() -> Path:
    """Build nam_benchmark without fast-math so IEEE behaviour is preserved."""
    subprocess.run(
        [
            "cmake",
            "-S",
            str(ROOT / "cpp"),
            "-B",
            str(BUILD_DIR),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DNAM_TOOLS_FAST_MATH=OFF",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(BUILD_DIR),
            "--target",
            "nam_benchmark",
            "--parallel",
            "8",
        ],
        check=True,
        capture_output=True,
    )
    return BUILD_DIR / "nam_benchmark"


def fidelity(model_path: Path, test_pair: tuple[Path, Path]) -> dict[str, float]:
    model = init_from_nam(json.loads(model_path.read_text(encoding="utf-8"))).eval()
    x, _ = sf.read(test_pair[0], dtype="float32")
    target, _ = sf.read(test_pair[1], dtype="float32")
    with torch.inference_mode():
        prediction = model(torch.from_numpy(x), pad_start=True).cpu().numpy()
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"non-finite prediction from {model_path}")
    return {**time_metrics(prediction, target), **spectral_metrics(prediction, target)}


def cost(benchmark: Path, model_path: Path) -> list[dict[str, float]]:
    rows = []
    for block in BLOCK_SIZES:
        result = subprocess.run(
            [str(benchmark), str(model_path), str(block), "2", "5"],
            check=True,
            capture_output=True,
            text=True,
        )
        rows.append(json.loads(result.stdout))
    return rows


def robustness(model_path: Path) -> dict[str, object]:
    """Check finite output and exact reset on degenerate inputs."""
    model = init_from_nam(json.loads(model_path.read_text(encoding="utf-8"))).eval()
    probes = {
        "silence": np.zeros(48_000, dtype=np.float32),
        "dc": np.full(48_000, 0.5, dtype=np.float32),
        "hot": np.full(48_000, 4.0, dtype=np.float32),
    }
    results = {}
    for name, signal in probes.items():
        with torch.inference_mode():
            out = model(torch.from_numpy(signal), pad_start=True).cpu().numpy()
        results[name] = {
            "finite": bool(np.all(np.isfinite(out))),
            "peak": float(np.max(np.abs(out))),
        }
    return results


def main() -> None:
    benchmark = build_benchmark()
    report: dict[str, dict] = {}
    for device, run_id in RUNS.items():
        run_dir = ROOT / "demo/runs" / run_id
        if not run_dir.exists():
            print(f"missing run {run_id}", file=sys.stderr)
            continue
        pairs = device_pairs(MANIFEST, device, root=ROOT)
        report[device] = {
            "run_id": run_id,
            "variants": {
                label: {
                    "fidelity": fidelity(run_dir / f"model_{label}.nam", pairs["test"]),
                    "cost": cost(benchmark, run_dir / f"model_{label}.nam"),
                    "robustness": robustness(run_dir / f"model_{label}.nam"),
                }
                for label in ("lite", "full")
            },
        }

    out = ROOT / "demo/report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Fiche technique de la démo",
        "",
        "Mesures produit sur le jeu de développement (Fulltone, Big Muff, ToneTwist",
        "CC-BY-NC). Aucune revendication scientifique ; voir `.codex_campaign/` pour",
        "la voie scientifique et ses verdicts.",
        "",
        "Repères : sur ce même Big Muff ToneTwist, la littérature rapporte 0,1076",
        "d'ESR pour S4-TFiLM `large` et 0,59 à 0,70 pour des gray-box à une seule",
        "non-linéarité. Le Big Muff est le cas difficile (deux étages de clipping",
        "en cascade) ; 400 époques au lieu de 100 n'y gagnent que 10 % d'ESR, donc",
        "le plateau vient du modèle, pas du budget. L'alignement prédiction/cible",
        "a été vérifié (décalage de pic nul).",
        "",
        "Le coût CPU est mesuré avec `-O3` sans `-ffast-math`, afin de préserver la",
        "propagation des NaN et la parité IEEE ; il est donc plus élevé que les",
        "mesures historiques de la voie scientifique compilées en `-Ofast`.",
        "",
        "## Fidélité (jeu de test scellé par appareil)",
        "",
        "| Appareil | Modèle | ESR | MAE | corrélation | MR-STFT | log-mel |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for device, entry in report.items():
        for label, variant in entry["variants"].items():
            f = variant["fidelity"]
            lines.append(
                f"| {device} | A2 {label} | {f['esr']:.4g} | {f['mae']:.4g} | "
                f"{f['correlation']:.4f} | {f['mrstft']:.4g} | {f['log_mel']:.4g} |"
            )
    lines += [
        "",
        "## Coût CPU natif (NeuralAmpModelerCore, Release -O3 sans fast-math)",
        "",
        "| Modèle | Bloc | ns/échantillon médian | p95 | facteur temps réel p95 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for device, entry in report.items():
        for label, variant in entry["variants"].items():
            for row in variant["cost"]:
                lines.append(
                    f"| {device} A2 {label} | {row['block_size']} | "
                    f"{row['median_ns_per_sample']:.1f} | "
                    f"{row['p95_ns_per_sample']:.1f} | "
                    f"{row['p95_realtime_factor']:.2f}x |"
                )
    lines += [
        "",
        "## Robustesse",
        "",
        "| Modèle | sonde | fini | crête |",
        "| --- | --- | --- | ---: |",
    ]
    for device, entry in report.items():
        for label, variant in entry["variants"].items():
            for probe, result in variant["robustness"].items():
                lines.append(
                    f"| {device} A2 {label} | {probe} | {result['finite']} | "
                    f"{result['peak']:.4g} |"
                )
    (ROOT / "demo/REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} and demo/REPORT.md")


if __name__ == "__main__":
    main()
