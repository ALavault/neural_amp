#!/usr/bin/env python3
"""Is the 0.187 ceiling on the Big Muff test file in the data or in the model?

Three measurements answer it, cheapest first:
  1. do two models make the SAME error, or merely errors of the same size?
  2. where does that error live, in time and in frequency?
  3. can a model fitted ON the test pair reproduce it at all?

Only the last one needs a GPU, and only for twenty minutes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

from product_report import MANIFEST, ROOT, RUNS, build_tools, run_native
from product_train import run_training

from fssr_nam.product.data import device_pairs

DEVICE = "electro_harmonix_big_muff"
OUT_MARKDOWN = ROOT / "demo/DATA_DIAGNOSIS.md"
OUT_JSON = ROOT / "demo/data_diagnosis.json"
SAMPLE_RATE = 48_000
WINDOW = SAMPLE_RATE  # one second
BANDS = ((20, 200), (200, 1_000), (1_000, 4_000), (4_000, 12_000), (12_000, 24_000))
# The reference run: 400 epochs over the 120 s train file. An epoch over a 30 s
# file is four times fewer updates, so the budget is scaled to match.
REFERENCE_EPOCHS = 400
OVERFIT_LIMIT = 0.02
# A window whose target is this far below the median is silence: its ESR divides
# by almost nothing and says nothing about the model.
SILENCE_FRACTION = 0.1
# Bins thinner than this are noise, not a transfer curve.
MIN_BIN = 10_000


def read(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32")
    if rate != SAMPLE_RATE:
        raise RuntimeError(f"{path.name} is at {rate} Hz")
    return audio


def predictions(pairs: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Render the test input through both A2 submodels with the native engine."""
    _, runner = build_tools()
    run_dir = ROOT / "demo/runs" / RUNS[DEVICE]
    x = read(pairs["test"][0])
    rendered = {}
    for label in ("full", "lite"):
        out, _ = run_native(runner, run_dir / f"model_{label}.nam", x, "64")
        rendered[label] = out
    return x, rendered


def residual_agreement(rendered: dict[str, np.ndarray], target: np.ndarray) -> dict:
    """Same error, or merely errors of the same size?

    A high correlation means both models miss the same component of the target,
    which no architecture can produce from this input. A low one means each is
    simply limited in its own way, and the ceiling is not in the data.
    """
    full = rendered["full"] - target
    lite = rendered["lite"] - target
    correlation = float(
        np.corrcoef(full.astype(np.float64), lite.astype(np.float64))[0, 1]
    )
    return {
        "residual_correlation_full_lite": correlation,
        "residual_rms_full": float(np.sqrt(np.mean(np.square(full)))),
        "residual_rms_lite": float(np.sqrt(np.mean(np.square(lite)))),
    }


def residual_shape(prediction: np.ndarray, target: np.ndarray) -> dict:
    """Where the error lives: concentrated in a passage, in a band, or spread."""
    residual = prediction - target
    windows = []
    for start in range(0, len(target) - WINDOW + 1, WINDOW):
        span = slice(start, start + WINDOW)
        windows.append(
            {
                "second": start // WINDOW,
                "esr": float(
                    np.sum(np.square(residual[span]))
                    / max(np.sum(np.square(target[span])), 1e-20)
                ),
                "target_rms": float(np.sqrt(np.mean(np.square(target[span])))),
            }
        )
    frequencies = np.fft.rfftfreq(len(target), 1.0 / SAMPLE_RATE)
    residual_spectrum = np.square(np.abs(np.fft.rfft(residual)))
    target_spectrum = np.square(np.abs(np.fft.rfft(target)))
    bands = []
    for low, high in BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        bands.append(
            {
                "band_hz": f"{low}-{high}",
                "residual_share": float(
                    np.sum(residual_spectrum[mask]) / np.sum(residual_spectrum)
                ),
                "band_esr": float(
                    np.sum(residual_spectrum[mask])
                    / max(np.sum(target_spectrum[mask]), 1e-20)
                ),
            }
        )
    levels = np.array([w["target_rms"] for w in windows])
    floor = float(np.median(levels)) * SILENCE_FRACTION
    for window in windows:
        window["silent"] = window["target_rms"] < floor
    voiced = np.array([w["esr"] for w in windows if not w["silent"]])
    return {
        "windows": windows,
        "bands": bands,
        "silence_floor_rms": floor,
        "silent_windows": int(sum(w["silent"] for w in windows)),
        "window_esr_median": float(np.median(voiced)),
        "window_esr_max": float(np.max(voiced)),
        "window_esr_ratio_max_median": float(np.max(voiced) / np.median(voiced)),
    }


def coverage(pairs: dict) -> dict:
    """Does the test input explore amplitudes the training input never showed?"""
    train = np.abs(read(pairs["train"][0]))
    ceiling = float(np.percentile(train, 99.9))
    rows = {}
    for split in ("train", "validation", "test"):
        x = np.abs(read(pairs[split][0]))
        rows[split] = {
            "p50": float(np.percentile(x, 50)),
            "p99": float(np.percentile(x, 99)),
            "p999": float(np.percentile(x, 99.9)),
            "peak": float(np.max(x)),
            "share_above_train_p999": float(np.mean(x > ceiling)),
        }
    return {"train_p999": ceiling, "splits": rows}


def transfer_curves(pairs: dict, bins: int = 24) -> dict:
    """Binned E[y | x] per split: a different pedal setting shows up here."""
    edges = np.linspace(-0.6, 0.6, bins + 1)
    curves, populations = {}, {}
    for split in ("train", "validation", "test"):
        x = read(pairs[split][0])
        y = read(pairs[split][1])
        index = np.digitize(x, edges) - 1
        means, counts = [], []
        for slot in range(bins):
            mask = index == slot
            count = int(np.count_nonzero(mask))
            counts.append(count)
            means.append(float(np.mean(y[mask])) if count >= MIN_BIN else None)
        curves[split] = means
        populations[split] = counts
    reference = curves["train"]
    deviations = {}
    for split in ("validation", "test"):
        pairs_kept = [
            (a, b)
            for a, b in zip(reference, curves[split], strict=True)
            if a is not None and b is not None
        ]
        spread = max(abs(a) for a, _ in pairs_kept)
        deviations[split] = float(
            max(abs(a - b) for a, b in pairs_kept) / max(spread, 1e-9)
        )
    return {
        "bin_centres": ((edges[:-1] + edges[1:]) / 2).tolist(),
        "curves": curves,
        "populations": populations,
        "minimum_bin_population": MIN_BIN,
        "max_relative_deviation_from_train": deviations,
    }


def oracle(pairs: dict) -> dict:
    """Fit a model ON one pair and score it on that same pair.

    This is deliberate overfitting: it asks whether the pair is internally
    consistent at all, not whether a model generalises to it.
    """
    results = {}
    train_samples = len(read(pairs["train"][0]))
    for split in ("test", "validation"):
        pair = pairs[split]
        scale = train_samples / len(read(pair[0]))
        epochs = round(REFERENCE_EPOCHS * scale)
        record = run_training(
            {"train": pair, "validation": pair, "test": pair},
            run_id=f"diag_overfit_{split}",
            device="DIAG_OVERFIT_not_a_real_capture",
            seed=0,
            max_epochs=epochs,
            progress_bar=False,
        )
        results[split] = {
            **record["test_esr"],
            "epochs": epochs,
            "minutes": record["minutes"],
        }
        print(
            f"  overfit sur {split}: {epochs} époques, "
            f"{record['minutes']:.1f} min, {record['test_esr']}",
            flush=True,
        )
    return results


def verdict(agreement: dict, shape: dict, fitted: dict, covered: dict) -> list[str]:
    correlation = agreement["residual_correlation_full_lite"]
    overfit_test = fitted["test"]["full"]
    overfit_validation = fitted["validation"]["full"]
    lines = []
    if correlation >= 0.9:
        lines.append(
            f"Les deux modèles font la MÊME erreur (corrélation des résidus "
            f"{correlation:.3f}) : il existe une composante de la cible de test "
            "qu'aucune architecture ne produit à partir de cette entrée."
        )
    elif correlation <= 0.5:
        lines.append(
            f"Les deux modèles font des erreurs DIFFÉRENTES (corrélation des "
            f"résidus {correlation:.3f}) : chacun est limité à sa façon, et le "
            "plafond n'est pas imputable aux données."
        )
    else:
        lines.append(
            f"Corrélation des résidus {correlation:.3f} : partiellement partagée, "
            "ni franchement la donnée ni franchement le modèle."
        )
    if overfit_test >= OVERFIT_LIMIT:
        lines.append(
            f"Un A2 Full entraîné SUR la paire de test n'y descend qu'à "
            f"{overfit_test:.5f} d'ESR, contre {overfit_validation:.5f} "
            "sur la paire de validation traitée de la même façon. La paire de "
            "test n'est pas cohérente avec elle-même : la cible n'est pas une "
            "fonction stable de cette entrée."
        )
        lines.append(
            "**Décision : recapturer avant toute ambition SOTA.** Optimiser une "
            "architecture contre une paire que l'on ne peut pas reproduire même "
            "en la surapprenant revient à courir après du bruit de mesure."
        )
    else:
        lines.append(
            f"Un A2 Full entraîné SUR la paire de test y descend à "
            f"{overfit_test:.5f} : la paire est apprenable en elle-même. Le "
            "plafond de 0,187 est donc un défaut de généralisation, pas une "
            "incohérence de la capture."
        )
        share = covered["splits"]["test"]["share_above_train_p999"]
        lines.append(
            f"**Décision : la chasse à l'architecture est légitime.** Cible "
            f"chiffrée : ESR de test < {overfit_test:.5f} est le plancher "
            f"atteignable en surapprentissage ; viser d'abord 0,15, soit 20 % "
            f"sous le plafond commun. À surveiller : {share * 100:.2f} % des "
            "échantillons de test dépassent le 99,9e centile de l'entraînement."
        )
    dominant = max(shape["bands"], key=lambda band: band["residual_share"])
    lines.append(
        f"Localisation de l'erreur : ESR médian par seconde "
        f"{shape['window_esr_median']:.4f}, maximum {shape['window_esr_max']:.4f} "
        f"(rapport {shape['window_esr_ratio_max_median']:.1f}). Bande dominante du "
        f"résidu : {dominant['band_hz']} Hz."
    )
    return lines


def markdown(report: dict) -> str:
    lines = [
        "# Diagnostic de données : le plafond est-il dans la donnée ou le modèle ?",
        "",
        "> **Erratum (15 septembre 2026).** Les fichiers analysés sont les fichiers"
        " publiés tronqués (120/30/30 s au lieu de 340/51/60 s). Le test ToneTwist "
        "a volontairement un contenu différent du train/val. Qu'il soit plus dur à "
        "surapprendre est compatible avec un matériau plus difficile et ne prouve "
        "pas que la cible est incohérente. La décision « recapturer avant toute "
        "ambition SOTA » reposait sur cette lecture et ne tient plus.",
        "",
        f"Appareil `{DEVICE}`, **fichier de test uniquement**. Les conclusions",
        "portent sur ce découpage, pas sur « le jeu de données ».",
        "",
        "## Verdict",
        "",
    ]
    lines += [f"{line}\n" for line in report["verdict"]]
    lines += [
        "## 1. Les deux modèles font-ils la même erreur ?",
        "",
        "| Mesure | Valeur |",
        "| --- | ---: |",
        f"| Corrélation des résidus A2 Full / A2 Lite | "
        f"{report['agreement']['residual_correlation_full_lite']:.4f} |",
        f"| RMS du résidu, A2 Full | {report['agreement']['residual_rms_full']:.5f} |",
        f"| RMS du résidu, A2 Lite | {report['agreement']['residual_rms_lite']:.5f} |",
        "",
        "## 2. Où vit l'erreur",
        "",
        "| Bande | Part du résidu | ESR dans la bande |",
        "| --- | ---: | ---: |",
    ]
    for band in report["shape"]["bands"]:
        lines.append(
            f"| {band['band_hz']} Hz | {band['residual_share'] * 100:.1f} % | "
            f"{band['band_esr']:.4f} |"
        )
    lines += [
        "",
        "| Seconde | ESR | RMS cible |",
        "| ---: | ---: | ---: |",
    ]
    for window in report["shape"]["windows"]:
        mark = " (silence)" if window["silent"] else ""
        lines.append(
            f"| {window['second']}{mark} | {window['esr']:.4f} | "
            f"{window['target_rms']:.4f} |"
        )
    lines += [
        "",
        "## 3. Cohérence interne des paires (surapprentissage délibéré)",
        "",
        "Budget mis à l'échelle pour égaler le nombre de mises à jour du run de",
        "référence (400 époques sur 120 s), sinon l'oracle n'a pas les moyens de",
        "surapprendre et son échec ne prouve rien.",
        "",
        "| Paire ajustée | Époques | Minutes | ESR A2 Full ici | ESR A2 Lite |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split, scores in report["oracle"].items():
        lines.append(f"| {split} | {scores['full']:.5f} | {scores['lite']:.5f} |")
    lines += [
        "",
        "## 4. Couverture d'amplitude",
        "",
        f"99,9e centile de |x| à l'entraînement : "
        f"{report['coverage']['train_p999']:.4f}",
        "",
        "| Split | p50 | p99 | p99,9 | crête | part au-dessus du p99,9 train |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, row in report["coverage"]["splits"].items():
        lines.append(
            f"| {split} | {row['p50']:.4f} | {row['p99']:.4f} | {row['p999']:.4f} | "
            f"{row['peak']:.4f} | {row['share_above_train_p999'] * 100:.3f} % |"
        )
    lines += [
        "",
        "## 5. Courbe de transfert E[y | x] par split",
        "",
        "Un réglage de pédale différent entre les prises se voit ici.",
        "",
        "| Split | Écart relatif maximal à la courbe d'entraînement |",
        "| --- | ---: |",
    ]
    for split, value in report["transfer"]["max_relative_deviation_from_train"].items():
        lines.append(f"| {split} | {value * 100:.2f} % |")
    return "\n".join(lines) + "\n"


def main() -> int:
    pairs = device_pairs(MANIFEST, DEVICE, root=ROOT)
    target = read(pairs["test"][1])
    print(f"=== Diagnostic de données, {DEVICE}, fichier de test ===\n")

    print("[1/5] Rendu des deux modèles A2 par le moteur natif")
    _, rendered = predictions(pairs)
    agreement = residual_agreement(rendered, target)
    print(
        "  corrélation des résidus Full/Lite : "
        f"{agreement['residual_correlation_full_lite']:.4f}"
    )

    print("\n[2/5] Localisation de l'erreur")
    shape = residual_shape(rendered["full"], target)
    print(
        f"  ESR par seconde : médian {shape['window_esr_median']:.4f}, "
        f"max {shape['window_esr_max']:.4f}"
    )
    for band in shape["bands"]:
        print(
            f"  {band['band_hz']:>11s} Hz : {band['residual_share'] * 100:5.1f} % "
            f"du résidu, ESR {band['band_esr']:.4f}"
        )

    print("\n[3/5] Couverture d'amplitude")
    covered = coverage(pairs)
    for split, row in covered["splits"].items():
        print(
            f"  {split:11s} p99.9 {row['p999']:.4f} crête {row['peak']:.4f} "
            f"au-dessus du train {row['share_above_train_p999'] * 100:.3f} %"
        )

    print("\n[4/5] Courbes de transfert")
    transfer = transfer_curves(pairs)
    for split, value in transfer["max_relative_deviation_from_train"].items():
        print(f"  {split}: écart relatif max à l'entraînement {value * 100:.2f} %")

    print("\n[5/5] Surapprentissage délibéré (2 x 100 époques)")
    fitted = oracle(pairs)

    report = {
        "device": DEVICE,
        "agreement": agreement,
        "shape": shape,
        "coverage": covered,
        "transfer": transfer,
        "oracle": fitted,
    }
    report["verdict"] = verdict(agreement, shape, fitted, covered)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    OUT_MARKDOWN.write_text(markdown(report), encoding="utf-8")
    print("\n=== Verdict ===")
    for line in report["verdict"]:
        print(line)
    print(f"\nwrote {OUT_MARKDOWN} and {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
