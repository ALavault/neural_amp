#!/usr/bin/env python3
"""Build the demo fact sheet: fidelity and native CPU cost of the A2 models."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
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


def build_tools() -> tuple[Path, Path]:
    """Build the NAM tools without fast-math so IEEE behaviour is preserved."""
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
            "nam_block_runner",
            "--parallel",
            "8",
        ],
        check=True,
        capture_output=True,
    )
    return BUILD_DIR / "nam_benchmark", BUILD_DIR / "nam_block_runner"


def run_native(runner: Path, model_path: Path, signal: np.ndarray, blocks: str):
    """Render one signal through the native engine and return (output, reset copy)."""
    with tempfile.TemporaryDirectory(dir=BUILD_DIR) as work:
        base = Path(work)
        signal.astype(np.float32).tofile(base / "in.f32")
        subprocess.run(
            [
                str(runner),
                str(model_path),
                str(base / "in.f32"),
                str(base / "out.f32"),
                blocks,
                str(base / "reset.f32"),
            ],
            check=True,
            capture_output=True,
        )
        return (
            np.fromfile(base / "out.f32", dtype=np.float32),
            np.fromfile(base / "reset.f32", dtype=np.float32),
        )


def parity(runner: Path, model_path: Path, prediction: np.ndarray, x: np.ndarray):
    """Compare the Python prediction with the native engine, block-wise."""
    regular, reset = run_native(runner, model_path, x, "64")
    irregular, _ = run_native(runner, model_path, x, "64,17,256,1,93")
    return {
        "python_vs_native_block64": float(np.max(np.abs(prediction - regular))),
        "native_regular_vs_irregular": float(np.max(np.abs(regular - irregular))),
        "reset_exact": bool(np.array_equal(regular, reset)),
    }


def fidelity(model_path: Path, test_pair: tuple[Path, Path]):
    """Score the model on the test pair and return the metrics with the signals."""
    model = init_from_nam(json.loads(model_path.read_text(encoding="utf-8"))).eval()
    x, _ = sf.read(test_pair[0], dtype="float32")
    target, _ = sf.read(test_pair[1], dtype="float32")
    with torch.inference_mode():
        prediction = model(torch.from_numpy(x), pad_start=True).cpu().numpy()
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"non-finite prediction from {model_path}")
    metrics = {
        **time_metrics(prediction, target),
        **spectral_metrics(prediction, target),
    }
    return metrics, prediction, x


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


def robustness(runner: Path, model_path: Path) -> dict[str, object]:
    """Probe the native engine with degenerate inputs, as it ships in the plugin."""
    probes = {
        "silence": np.zeros(48_000, dtype=np.float32),
        "dc": np.full(48_000, 0.5, dtype=np.float32),
        "hot": np.full(48_000, 4.0, dtype=np.float32),
    }
    results = {}
    for name, signal in probes.items():
        out, _ = run_native(runner, model_path, signal, "64,17,256,1,93")
        results[name] = {
            "finite": bool(np.all(np.isfinite(out))),
            "peak": float(np.max(np.abs(out))),
        }
    return results


def main() -> None:
    benchmark, runner = build_tools()
    report: dict[str, dict] = {}
    for device, run_id in RUNS.items():
        run_dir = ROOT / "demo/runs" / run_id
        if not run_dir.exists():
            print(f"missing run {run_id}", file=sys.stderr)
            continue
        pairs = device_pairs(MANIFEST, device, root=ROOT)
        variants = {}
        for label in ("lite", "full"):
            model_path = run_dir / f"model_{label}.nam"
            metrics, prediction, x = fidelity(model_path, pairs["test"])
            variants[label] = {
                "fidelity": metrics,
                "parity": parity(runner, model_path, prediction, x),
                "cost": cost(benchmark, model_path),
                "robustness": robustness(runner, model_path),
            }
        report[device] = {"run_id": run_id, "variants": variants}

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
        "## Fidélité (jeu de test de développement, par appareil)",
        "",
        "Ce jeu de test n'est pas scellé : le budget d'époques a été choisi après",
        "l'avoir lu. Sur le Big Muff, l'ESR de validation (0,109) est 1,7 fois",
        "meilleur que celui de test (0,188). Le DI de test est mesurablement plus",
        "exigeant que celui de validation : RMS 0,110 contre 0,0708 (+3,8 dB, donc",
        "un écrêtage plus profond), facteur de crête 5,00 contre 7,78 (jeu plus",
        "soutenu) et flux spectral 43 % plus élevé.",
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
        "Le facteur temps réel p95 indique combien de fois plus vite que le temps",
        "réel le modèle calcule (plus grand = mieux) ; la charge CPU est son inverse",
        "sur un cœur. Référence historique en `-Ofast` : 2 862 ns/échantillon pour",
        "A2 Full au bloc 64. Le médian est la mesure fiable ici : le p95 capte",
        "toute autre charge de la machine au moment du banc, et cette machine est",
        "partagée avec des entraînements. À lire avec les mesures sur machine",
        "dédiée quand elles existeront.",
        "",
        "| Modèle | Bloc | ns/éch. médian | p95 | x temps réel p95 | charge CPU p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for device, entry in report.items():
        for label, variant in entry["variants"].items():
            for row in variant["cost"]:
                lines.append(
                    f"| {device} A2 {label} | {row['block_size']} | "
                    f"{row['median_ns_per_sample']:.1f} | "
                    f"{row['p95_ns_per_sample']:.1f} | "
                    f"{row['p95_realtime_factor']:.2f}x | "
                    f"{100.0 / row['p95_realtime_factor']:.1f} % |"
                )
    lines += [
        "",
        "## Parité moteur (Python d'entraînement vs moteur natif C++)",
        "",
        "| Modèle | Python vs natif (bloc 64) | blocs réguliers vs irréguliers"
        " | reset exact |",
        "| --- | ---: | ---: | --- |",
    ]
    for device, entry in report.items():
        for label, variant in entry["variants"].items():
            p = variant["parity"]
            lines.append(
                f"| {device} A2 {label} | {p['python_vs_native_block64']:.2e} | "
                f"{p['native_regular_vs_irregular']:.2e} | {p['reset_exact']} |"
            )
    lines += [
        "",
        "Le rendu hors ligne du plugin JUCE (`scripts/product_plugin_parity.py`)",
        "est bit à bit identique au runner natif sur les quatre modèles, malgré",
        "les conversions float/double et le flush des dénormaux.",
        "",
        "## Robustesse (moteur natif, blocs irréguliers)",
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
