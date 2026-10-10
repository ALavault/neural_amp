#!/usr/bin/env python3
"""Reamp source for the next capture session (software device only), product lane.

Writes, with fixed seeds:

- datasets/raw/internal/capture_v2/reamp_v2_192k.wav, 192 kHz 24-bit mono (not
  committed: it carries ToneTwist DI material, CC BY-NC). Every section of
  reamp_valvestate_192k.wav up to di_test is copied sample for sample at its own offset,
  so the 150 s training part stays bit-identical (research-memory D-0012) and the 30 s
  extracts of demo/train_length stay contained in it. Then the measurement signals
  asked by T-0013, as fssr_nam.product.defect_bench generates them: synchronized
  sweeps, periodic odd multisines (R realizations x P periods), two tones and level
  steps; the end markers and padding close the file.
- demo/capture/v2/layout.json: where each section starts, the parameters of every new
  excitation, the file's sha256 and the training part's.
- demo/capture/v2/settings.csv and SESSION.md: the session sheet for an amp-sim plugin
  (Overloud TH-U 2 or Neural DSP Archetype: Nolly), rendered offline in Reaper; the
  VS100R design (centre three times, one knob at a time, combinations) on five generic
  tone controls mapped at the session start, plus a cab-on control block.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from fssr_nam.product import defect_bench as db

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "datasets/raw/internal/valvestate_vs100r/reamp_valvestate_192k.wav"
OLD_LAYOUT = ROOT / "demo/capture/valvestate/layout.json"
AUDIO = ROOT / "datasets/raw/internal/capture_v2/reamp_v2_192k.wav"
LAYOUT = ROOT / "demo/capture/v2/layout.json"
RATE = 192_000
GAP = RATE // 2
PAD_END_S = 6
TRAINING = ("di_train", "di_validation")
# Synchronized sweeps at the peaks of the old sweeps. 6.91 s gives L = 1.0 s: the
# 0.1 s harmonic windows of product.hammerstein stay apart up to order 10 (T >= 5.9 s
# is needed for order 9).
SWEEP_PEAKS_DB = (-20.0, -10.0, -3.0)
SWEEP_S = 6.91
# RMS levels of the multisines and of the loud step segments. The defect bench used
# -12/-9/-6 dBFS for the steps, which clip in a PCM file (crest factor ~15.5 dB, plus
# 2 dB of jitter); -18 keeps every peak below -1 dBFS, near the loudest DI passages.
NOISE_RMS_DB = (-30.0, -24.0, -18.0)
MULTISINE_REALIZATIONS, MULTISINE_PERIODS = 4, 3
TWO_TONE_S, TWO_TONE_SEED = 2.0, 7
STEPS_SEED = 300
# Lines of defect_bench.multisine_signal (1 Hz grid, 1 s period).
MULTISINE_LINES = {
    "excited_hz": "odd 21-19999, minus the detection lines",
    "detection_hz": "27, 35, ..., 19995 (every fourth odd line, step 8)",
    "even_hz": "not excited: even-order detection lines",
}
FADE_S = 0.01
SHEET_DIR = ROOT / "demo/capture/v2"
SEED = 20261010
# Five tone controls, mapped to the chosen amp's controls at the session start.
KNOBS = {"G": "gain", "B": "grave", "M": "médium", "T": "aigu", "P": "présence"}
VALUES = (0.0, 2.5, 5.0, 7.5, 10.0)
# The eight test combinations of the VS100R session, in the order of KNOBS.
TEST = (
    (5, 0, 10, 0, 10),
    (5, 10, 0, 10, 0),
    (10, 5, 10, 10, 0),
    (2.5, 5, 0, 0, 10),
    (7.5, 2.5, 7.5, 2.5, 5),
    (2.5, 7.5, 2.5, 7.5, 7.5),
    (5, 7.5, 7.5, 2.5, 2.5),
    (7.5, 2.5, 2.5, 7.5, 2.5),
)
SOFTWARE = {"thu2": "Overloud TH-U 2", "nolly": "Neural DSP Archetype: Nolly"}


def to_int(x: np.ndarray) -> np.ndarray:
    """Float samples to the int32 form soundfile uses for 24-bit files (int24 << 8)."""
    q = np.clip(np.round(x * 2**23), -(2**23), 2**23 - 1).astype(np.int32)
    return q << 8


def faded(x: np.ndarray) -> np.ndarray:
    n = int(FADE_S * RATE)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)
    x = x.copy()
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_hash(path: Path, sections: list[dict]) -> str:
    """sha256 of the int32 samples (soundfile dtype int32) of the training part."""
    parts = [s for s in sections if s["section"] in TRAINING]
    start = parts[0]["start"]
    stop = parts[-1]["start"] + parts[-1]["samples"]
    x = sf.read(path, dtype="int32", start=start, stop=stop)[0]
    return hashlib.sha256(x.astype("<i4").tobytes()).hexdigest()


def new_sections() -> list[tuple[str, np.ndarray, dict]]:
    out = []
    for peak in SWEEP_PEAKS_DB:
        x, sweep = db.synchronized_sweep(SWEEP_S, RATE)
        params = {
            "peak_dbfs": peak,
            "f0": sweep.f0,
            "f1": sweep.f1,
            "seconds": sweep.seconds,
            "lag": sweep.lag,
        }
        out.append((f"sweep_sync_{peak:g}dBFS", db.at_level(x, peak, "peak"), params))
    for level in NOISE_RMS_DB:
        for r in range(1, MULTISINE_REALIZATIONS + 1):
            x = db.multisine_signal(MULTISINE_PERIODS, RATE, seed=r)
            params = {
                "rms_dbfs": level,
                "seed": r,
                "period_samples": RATE,
                "periods": MULTISINE_PERIODS,
                **MULTISINE_LINES,
            }
            name = f"multisine_{level:g}dBFS_rms_r{r}"
            out.append((name, db.at_level(x, level, "rms"), params))
    for peak in SWEEP_PEAKS_DB:
        x = db.two_tone_signal(TWO_TONE_S, RATE, TWO_TONE_SEED)
        f1, f2 = db.two_tone_frequencies(TWO_TONE_SEED)
        params = {"peak_dbfs": peak, "seed": TWO_TONE_SEED, "f1": f1, "f2": f2}
        x = faded(db.at_level(x, peak, "peak"))
        out.append((f"two_tone_{peak:g}dBFS", x, params))
    for level in NOISE_RMS_DB:
        x, onsets = db.steps_signal(level, STEPS_SEED, RATE)
        params = {
            "loud_rms_dbfs": level,
            "drop_db": db.STEP_DROP_DB,
            "jitter_db": "uniform +-2 dB per segment, around the nominal levels",
            "segment_rms_dbfs": [
                round(20 * np.log10(np.sqrt(np.mean(v**2))), 3)
                for v in np.split(x, onsets)
            ],
            "segment_s": db.STEP_S,
            "segments": db.STEP_SEGMENTS,
            "seed": STEPS_SEED,
            "onsets": onsets.tolist(),
        }
        out.append((f"steps_{level:g}dBFS_rms", x, params))
    return out


def name(values: dict[str, float]) -> str:
    return "_".join(f"{k}{round(v * 10):03d}" for k, v in values.items())


def settings() -> list[dict]:
    """The VS100R design on KNOBS: one knob at a time around the centre, the eight
    test combinations, eight drawn training combinations, the centre at the start, the
    middle and the end; then three cab-on controls."""
    center = dict.fromkeys(KNOBS, 5.0)
    oat = [{**center, k: v} for k in KNOBS for v in VALUES if v != 5.0]
    test = [dict(zip(KNOBS, v, strict=True)) for v in TEST]
    rng = random.Random(SEED)
    taken = {name(s) for s in [center, *oat, *test]}
    grid = [
        dict(zip(KNOBS, v, strict=True)) for v in itertools.product(VALUES, repeat=5)
    ]
    train = []
    while len(train) < 8:
        s = rng.choice(grid)
        if sum(v != 5 for v in s.values()) >= 2 and name(s) not in taken:
            train.append(s)
            taken.add(name(s))
    rows = (
        [("one_knob", s, "train") for s in oat]
        + [("combination_test", s, "test") for s in test]
        + [("combination_train", s, "train") for s in train]
    )
    rng.shuffle(rows)
    middle = len(rows) // 2
    anchor = ("center", center, "train")
    rows = [anchor, *rows[:middle], anchor, *rows[middle:], anchor]
    rows = [(b, s, split, "off") for b, s, split in rows]
    cab = [center, test[0], test[4]]
    rng.shuffle(cab)
    rows += [("cab_on", s, "control", "on") for s in cab]
    order = []
    for i, (block, s, split, cab_state) in enumerate(rows, 1):
        # X100: cab on, one letter and a number so that nablafx's name parser reads it.
        setting = name(s) + ("_X100" if cab_state == "on" else "")
        file = f"{setting}.capv2_{i:02d}.wav"
        order.append(
            {"order": i, "block": block, **s, "cab": cab_state, "split": split}
            | {"setting": setting, "file": file}
        )
    return order


def knob(value: float) -> str:
    return f"{value:g}".replace(".", ",")


def session_sheet(rows: list[dict], layout: dict) -> str:
    mb = round(layout["seconds"] * RATE * 4 / 1e6)
    total = f"{mb * len(rows) / 1e3:.1f}".replace(".", ",")
    knobs = " | ".join(KNOBS.values())
    lines = [
        "# Séance de capture v2 : ampli logiciel",
        "",
        "Voie produit. Généré par `scripts/product_capture_v2.py` (graine "
        f"{SEED}) ; ne pas éditer à la main, relancer le script. Appareil **logiciel** "
        "seulement (research-memory D-0013) : "
        + " ou ".join(SOFTWARE.values())
        + ". Un dossier par logiciel ("
        + ", ".join(f"`{k}/`" for k in SOFTWARE)
        + ").",
        "",
        "## Avant la première prise",
        "",
        f"- **Source** : `{layout['file']}`, {layout['seconds']} s, "
        f"{RATE // 1000} kHz, 24 bits, sha256 `{layout['sha256'][:16]}…`. "
        "Sections : `layout.json`.",
        f"- **Reaper** : projet à **{RATE // 1000} kHz** (sinon Reaper rééchantillonne "
        "la source et l'entrée n'est plus identique au bit, D-0012). Item source : "
        "volume d'item 0 dB, sans fondu, sans changement de vitesse ni de hauteur. "
        "Piste source : volume 0 dB, pan centré, aucun effet hors le plugin ; master "
        "à 0 dB, sans effet. Auto-bypass de Reaper sur silence **désactivé** (pour "
        "le plugin et pendant les rendus) : il couperait le silence et les queues.",
        "- **Rendu** : hors ligne pleine vitesse, WAV **32 bits flottants** (un ampli "
        "logiciel peut dépasser 0 dBFS : 24 bits écrêterait), **mono**, "
        f"{RATE // 1000} kHz, ni dither ni normalisation, bornes = tout le projet "
        "(le fichier finit par 6 s de silence pour la queue). Méthode simple : une "
        "piste par prise, chacune avec son instance du plugin, puis « Render stems "
        "(selected tracks) » en une passe.",
        f"- **Place** : ~{mb} Mo par prise, ~{total} Go pour les {len(rows)} prises.",
        "",
        "### Contrôle 0 : bouclage logiciel (condition d'arrêt)",
        "",
        "Rendre la piste source **sans le plugin** (`loopback_software.wav`), avec "
        "exactement les réglages de rendu ci-dessus. Test du nul : sur un nouveau "
        "projet, importer le WAV source d'origine (nouvel item, réglages par défaut) "
        "et ce rendu, polarité inversée sur l'un des deux : le master doit rester à "
        "-inf. **Sinon, ne rien rendre d'autre** (rééchantillonnage, gain d'item, loi "
        "de panoramique, dither, normalisation ou fader à corriger). M'envoyer aussi "
        "ce rendu : je le compare au fichier d'origine échantillon par échantillon.",
        "",
        "### État fixe du plugin, toute la séance",
        "",
        "- **Coupés** : noise gate (il couperait les paliers faibles, le silence et le "
        "bruit rose), toutes les pédales, EQ graphique, doubler, delay, reverb, "
        "transposition, accordeur.",
        "- **Baffle (cab)** : coupé, sauf le bloc de contrôle *cab on* en fin de "
        "séance (même baffle, mêmes micros, même placement pour ses trois prises).",
        "- **Fixes et notés** : gain d'entrée, gain de sortie, volume ou master de "
        "l'ampli, mode de qualité ou de suréchantillonnage.",
        "- **Modèle d'ampli** : un seul pour toute la séance ; pas une émulation "
        "Blackstar ni 1176 (volet scientifique scellé).",
        "- **Boutons** : valeur exacte, vérifiée sur l'affichage numérique du plugin ; "
        "la méthode dépend du plugin (chez Nolly X, le double clic rétablit la valeur "
        "par défaut : réglage fin par glissement avec la touche indiquée dans le "
        "manuel). Si l'affichage est en %, 10 = 100 %. Pour chaque prise, enregistrer "
        "le preset sous le nom du fichier et une capture d'écran.",
        "",
        "### Pré-contrôles sur l'ampli choisi (centre, tous boutons à 5)",
        "",
        f"- Le plugin se charge et rend à **{RATE // 1000} kHz**. S'il refuse : "
        "s'arrêter et demander une variante (48 kHz par décimation de la source).",
        "- **Déterminisme** : rendre le centre deux fois, dans deux instances "
        "séparées ; test du nul entre les deux : -inf attendu. Sinon, vérifier les "
        "réglages, puis s'arrêter et m'envoyer les deux rendus : un résidu peut venir "
        "d'un réglage, d'un état initial ou d'un décalage, pas seulement du plugin.",
        "- **Stéréo** : rendre le centre une fois en stéréo (`center_stereo.wav`) ; "
        "les deux canaux doivent être identiques, sinon un étage stéréo est resté "
        "actif. L'envoyer avec `loopback_software.wav` : je vérifie les deux.",
        "- **Latence** : noter la PDC affichée par Reaper pour le plugin.",
        "",
        "Valeurs fixes, à noter une fois par logiciel :",
        "",
        "| | valeur |",
        "|---|---|",
        "| Logiciel, version, format (VST3/AU) | |",
        "| Système, version de Reaper | |",
        "| Modèle d'ampli, canal | |",
        *[f"| Commande de l'ampli pour « {v} » ({k}) | |" for k, v in KNOBS.items()],
        "| Autres commandes de l'ampli (valeur fixe) | |",
        "| Gain d'entrée, gain de sortie | |",
        "| Mode de qualité ou de suréchantillonnage | |",
        "| PDC affichée (échantillons) | |",
        "| Déterminisme : résidu du nul (dBFS) | |",
        "| Bloc cab on : baffle, micros, placement | |",
        "",
        "## Prises, dans cet ordre",
        "",
        "Boutons de 0 à 10. Nommer chaque rendu comme dans la dernière colonne, dans "
        "le dossier du logiciel.",
        "",
        f"| # | fait | {knobs} | baffle | rôle | fichier |",
        "|---|---|" + "---|" * len(KNOBS) + "---|---|---|",
    ]
    for r in rows:
        values = " | ".join(knob(r[k]) for k in KNOBS)
        lines.append(
            f"| {r['order']} | ☐ | {values} | {r['cab']} | "
            f"{r['block'].replace('_', ' ')} ({r['split']}) | `{r['file']}` |"
        )
    lines += [
        "",
        "Rôles : *one knob* et *combination train* servent à l'entraînement ; "
        "*combination test* n'y entre jamais ; *center* trois fois (déterminisme) ; "
        "*cab on* (suffixe `X100`) est un contrôle : la même chaîne avec le baffle, "
        "en principe un "
        "filtre linéaire de plus (les sondes le vérifieront).",
        "",
    ]
    return "\n".join(lines)


def write_sheet(layout: dict) -> None:
    rows = settings()
    with (SHEET_DIR / "settings.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (SHEET_DIR / "SESSION.md").write_text(session_sheet(rows, layout))


def main() -> None:
    old = json.loads(OLD_LAYOUT.read_text())
    assert sha256(OLD) == old["sha256"], "reamp_valvestate_192k.wav changed"
    by_name = {s["section"]: s for s in old["sections"]}
    test = by_name["di_test"]
    stop = test["start"] + test["samples"]
    head = sf.read(OLD, dtype="int32", stop=stop)[0]
    layout = [s for s in old["sections"] if s["start"] < stop]
    parts, position = [head, np.zeros(GAP, np.int32)], stop + GAP
    for name, x, params in new_sections():
        assert np.max(np.abs(x)) < 10 ** (-1 / 20), f"{name} peaks above -1 dBFS"
        layout.append({"section": name, "start": position, "samples": len(x)} | params)
        parts += [to_int(x), np.zeros(GAP, np.int32)]
        position += len(x) + GAP
    end = by_name["markers_end"]
    markers = sf.read(
        OLD, dtype="int32", start=end["start"], stop=end["start"] + end["samples"]
    )[0]
    tail = np.zeros(PAD_END_S * RATE, np.int32)
    layout.append(
        {"section": "markers_end", "start": position, "samples": len(markers)}
    )
    layout.append(
        {"section": "pad_end", "start": position + len(markers), "samples": len(tail)}
    )
    parts += [markers, tail]
    AUDIO.parent.mkdir(parents=True, exist_ok=True)
    sf.write(AUDIO, np.concatenate(parts), RATE, subtype="PCM_24")
    written = sf.read(AUDIO, dtype="int32")[0]
    copied = sf.read(OLD, dtype="int32", stop=stop)[0]
    assert np.array_equal(written[:stop], copied), "copied sections differ"
    training = training_hash(AUDIO, layout)
    assert training == training_hash(OLD, old["sections"])
    result = {
        "sample_rate": RATE,
        "sections": layout,
        "file": str(AUDIO.relative_to(ROOT)),
        "sha256": sha256(AUDIO),
        "seconds": round(len(written) / RATE, 3),
        "copied_from": {
            "file": old["file"],
            "sha256": old["sha256"],
            "through": "di_test",
            "same_offsets": True,
        },
        "training_part_sha256": training,
        "training_part_hash": "sha256 of the int32 samples (soundfile dtype int32, "
        "little-endian) of di_train + di_validation",
        "di_source": old["di_source"],
        "notes": [
            "Software device only (no new hardware captures). Render at 192 kHz: "
            "another rate breaks the bit-identity of the input (D-0012).",
            "steps: loud levels lowered from the defect bench's -12/-9/-6 dBFS RMS to "
            "fit a PCM file; descriptive reading of E2, not the bench protocol.",
            "multisine: the first period of each realization is the transient; with a "
            "deterministic device the repeated periods mostly check determinism, the "
            "detection lines give the even and odd nonlinear levels.",
            "sweep_sync: lag = seconds / ln(f1 / f0) = 1.0 s; the old sweep_*dBFS "
            "sections stay 5 s and unsynchronized.",
        ],
    }
    LAYOUT.parent.mkdir(parents=True, exist_ok=True)
    LAYOUT.write_text(json.dumps(result, indent=1) + "\n")
    write_sheet(result)
    print(f"{result['file']}: {result['seconds']} s, sha256 {result['sha256'][:16]}")
    print(f"training part sha256 {training[:16]} (identical to the VS100R file)")


if __name__ == "__main__":
    main()
