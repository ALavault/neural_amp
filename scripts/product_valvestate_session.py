#!/usr/bin/env python3
"""Reamp source and settings sheet for a Marshall Valvestate VS100R capture session.

Product lane. Writes, with fixed seeds:

- the reamp source, 192 kHz 24-bit mono, to datasets/raw/internal/valvestate_vs100r/
  (not committed: it carries ToneTwist DI material, CC BY-NC):
  calibration blips (as fssr_nam.product.capture, at 192 kHz), silence, low-level pink
  noise, three sine sweeps, DI guitar and bass (train 120 s, validation 30 s, test
  30 s, the test from recordings the train never uses), calibration blips again;
- demo/capture/valvestate/layout.json: where each section starts, and the file's sha256;
- demo/capture/valvestate/settings.csv and SESSION.md: the 47 settings in capture
  order, named like the ToneTwist parametric sets (G050_C050_B100_M050_T050) so that
  nablafx's parametric loader reads the knob values from the file name.
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
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
DRY = ROOT / "datasets/raw/external/tone_twist_param/extracted/DIGITAL-PARAMETRIC"
DRY = DRY / "DRY-with-markers"
AUDIO_DIR = ROOT / "datasets/raw/internal/valvestate_vs100r"
SHEET_DIR = ROOT / "demo/capture/valvestate"
# 192 kHz: one capture then gives native references at 192, 96 and 48 kHz by
# integer decimation (demo/sr_transfer: rate transfer and the -37 dB floor).
RATE = 192_000
SOURCE_RATE = 48_000
SEED = 20260926

# Calibration blips as fssr_nam.product.capture defines them, at RATE.
BLIP_MS, BLIP_HZ, BLIP_AMPLITUDE, GAP_S = 2.0, 1_000.0, 0.5, 0.5
FADE_S = 0.01
# DI excerpts (file, start in seconds): away from the calibration markers that open
# and close every ToneTwist dry file.
TRAIN = [
    ("idmt-gtr2", 60),
    ("idmt-gtr4-sg", 60),
    ("nam", 40),
    ("prvt-gtr", 60),
    ("yt-bass", 60),
]
VALIDATION = [
    ("idmt-gtr2", 200),
    ("idmt-gtr4-sg", 200),
    ("nam", 120),
    ("prvt-gtr", 200),
    ("yt-bass", 300),
]
TEST = [("idmt-bass", 60), ("idmt-gtr4-ib", 60)]
SECONDS = {"train": 120, "validation": 30, "test": 30}


def blip() -> np.ndarray:
    n = int(BLIP_MS * RATE / 1000)
    i = np.arange(n)
    return (
        BLIP_AMPLITUDE
        * np.cos(2 * np.pi * BLIP_HZ * i / RATE)
        * np.cos(0.5 * np.pi * i / n)
    )


def markers() -> np.ndarray:
    gap = np.zeros(int(GAP_S * RATE))
    return np.concatenate([gap, blip(), gap, blip(), gap])


def fade(x: np.ndarray) -> np.ndarray:
    n = int(FADE_S * RATE)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)
    x = x.copy()
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


def pink_noise(seconds: float, rms_dbfs: float, rng: np.random.Generator) -> np.ndarray:
    n = int(seconds * RATE)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / RATE)
    band = (freqs >= 20) & (freqs <= 20_000)
    spectrum[~band] = 0
    spectrum[band] /= np.sqrt(freqs[band])
    x = np.fft.irfft(spectrum, n)
    return fade(x * 10 ** (rms_dbfs / 20) / np.sqrt(np.mean(x**2)))


def sweep(seconds: float, peak_dbfs: float) -> np.ndarray:
    """Exponential sine sweep, 20 Hz to 20 kHz."""
    t = np.arange(int(seconds * RATE)) / RATE
    f0, f1 = 20.0, 20_000.0
    k = np.log(f1 / f0)
    phase = 2 * np.pi * f0 * seconds / k * (np.exp(t * k / seconds) - 1)
    return fade(10 ** (peak_dbfs / 20) * np.sin(phase))


def di(excerpts: list[tuple[str, int]], seconds: int) -> np.ndarray:
    each = seconds / len(excerpts)
    parts = []
    for name, start in excerpts:
        split = "test" if (DRY / "test" / f"{name}.input.wav").exists() else "trainval"
        x, rate = sf.read(DRY / split / f"{name}.input.wav", dtype="float64")
        assert rate == SOURCE_RATE
        segment = x[int(start * rate) : int((start + each) * rate)]
        rms = 20 * np.log10(np.sqrt(np.mean(segment**2)))
        assert rms > -45, f"{name} at {start} s is nearly silent ({rms:.1f} dBFS)"
        parts.append(fade(resample_poly(segment, RATE // SOURCE_RATE, 1)))
    return np.concatenate(parts)


def reamp() -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(SEED)
    gap = np.zeros(int(GAP_S * RATE))
    sections = [
        ("markers_start", markers()),
        ("silence", np.zeros(5 * RATE)),
        ("gap", gap),
        ("pink_noise_-30dBFS_rms", pink_noise(5, -30, rng)),
        ("gap", gap),
    ]
    for level in (-20, -10, -3):
        sections += [(f"sweep_{level}dBFS", sweep(5, level)), ("gap", gap)]
    for split, excerpts in (
        ("train", TRAIN),
        ("validation", VALIDATION),
        ("test", TEST),
    ):
        sections.append((f"di_{split}", di(excerpts, SECONDS[split])))
    sections.append(("markers_end", markers()))
    layout, position = [], 0
    for name, x in sections:
        if name != "gap":
            layout.append({"section": name, "start": position, "samples": len(x)})
        position += len(x)
    signal = np.concatenate([x for _, x in sections])
    assert np.max(np.abs(signal)) < 1.0
    return signal.astype(np.float32), {"sample_rate": RATE, "sections": layout}


def self_check(signal: np.ndarray) -> int:
    """The first blip must be found at a known delay in a simulated, noisy capture."""
    delay = 777
    rng = np.random.default_rng(1)
    capture = np.concatenate([np.zeros(delay), 0.3 * signal])
    capture += 1e-4 * rng.standard_normal(len(capture))
    start = int(GAP_S * RATE)
    window = capture[start : start + RATE]
    baseline = capture[start - 9_600 : start]
    threshold = 10 * np.std(baseline)
    return int(np.nonzero(np.abs(window - np.median(baseline)) > threshold)[0][0])


KNOBS = ("G", "C", "B", "M", "T")  # gain, contour, bass, middle, treble (OD2)
VALUES = (0.0, 2.5, 5.0, 7.5, 10.0)


def name(values: dict[str, float]) -> str:
    return "_".join(f"{k}{round(v * 10):03d}" for k, v in values.items())


def settings() -> list[dict]:
    center = dict.fromkeys(KNOBS, 5.0)
    oat = [
        {**center, knob: value} for knob in KNOBS for value in VALUES if value != 5.0
    ]
    test = [
        dict(zip(KNOBS, v, strict=True))
        for v in (
            (5, 0, 10, 0, 10),
            (5, 10, 0, 10, 0),
            (10, 5, 10, 10, 0),
            (2.5, 5, 0, 0, 10),
            (7.5, 2.5, 7.5, 2.5, 5),
            (2.5, 7.5, 2.5, 7.5, 7.5),
            (5, 7.5, 7.5, 2.5, 2.5),
            (7.5, 2.5, 2.5, 7.5, 2.5),
        )
    ]
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
        [("od2_one_knob", s, "train") for s in oat]
        + [("od2_combination_test", s, "test") for s in test]
        + [("od2_combination_train", s, "train") for s in train]
    )
    rng.shuffle(rows)
    middle = len(rows) // 2
    anchor = ("od2_center", center, "train")
    rows = [anchor, *rows[:middle], anchor, *rows[middle:], anchor]
    order = [
        {
            "block": b,
            "channel": "OD2",
            "power_dimension": "off",
            "tone_shift": "",
            "setting": name(s),
            "split": split,
            **{k: s[k] for k in KNOBS},
        }
        for b, s, split in rows
    ]
    clean = [
        ({"B": 5, "M": 5, "T": 5}, "off"),
        ({"B": 5, "M": 5, "T": 5}, "on"),
        ({"B": 10, "M": 5, "T": 0}, "off"),
        ({"B": 0, "M": 5, "T": 10}, "off"),
        ({"B": 7.5, "M": 2.5, "T": 7.5}, "off"),
    ]
    rng.shuffle(clean)
    order += [
        {
            "block": "clean",
            "channel": "Clean",
            "power_dimension": "off",
            "tone_shift": ts,
            "setting": name(s) + f"_S{100 if ts == 'on' else 0:03d}",
            "split": "control",
            "G": "",
            "C": "",
            **s,
        }
        for s, ts in clean
    ]
    power = [dict(center), test[0], test[4]]
    rng.shuffle(power)
    order += [
        {
            "block": "power_dimension",
            "channel": "OD2",
            "power_dimension": "on",
            "tone_shift": "",
            "setting": name(s) + "_P100",
            "split": "control",
            **{k: s[k] for k in KNOBS},
        }
        for s in power
    ]
    for i, row in enumerate(order, 1):
        row["order"] = i
        row["file"] = f"{row['setting']}.vs100r_{i:02d}.wav"
    return order


def knob(value) -> str:
    return "" if value == "" else f"{float(value):g}".replace(".", ",")


def session_sheet(rows: list[dict], layout: dict) -> str:
    lines = [
        "# Séance de capture : Marshall Valvestate VS100R",
        "",
        "Voie produit. Généré par `scripts/product_valvestate_session.py` (graine "
        f"{SEED}) ; ne pas éditer à la main, relancer le script.",
        "",
        "## Avant la première prise",
        "",
        f"- Source : `{layout['file']}`, {layout['seconds']} s, {RATE // 1000} kHz, "
        "24 bits, "
        f"sha256 `{layout['sha256'][:16]}…`. Sections et positions : `layout.json`.",
        f"- Interface : Antelope Zen Go Synergy Core, **{RATE // 1000} kHz**, 24 bits, "
        "taille de "
        "tampon fixe pour toute la séance. **Tous les effets Synergy Core et le "
        "monitoring direct coupés** sur les entrées enregistrées.",
        "- Deux voies (la Zen Go n'a que deux entrées), en mode **ligne** : entrée 1 = "
        "départ de boucle (FX Send), entrée 2 = sortie ligne émulée.",
        "- Sortie moniteur (jack TRS) → boîte de réamplification → entrée de l'ampli. "
        "**Jamais** la sortie haut-parleur vers l'interface.",
        "- **Prise de bouclage** au début et à la fin de la séance : sortie de la "
        "boîte de réamplification branchée sur l'entrée 2 à la place de l'ampli, mêmes "
        "réglages, source jouée en entier (`loopback_start.wav`, `loopback_end.wav`). "
        "Elle donne la latence propre de la chaîne ; les blips suffisent ensuite.",
        f"- **Reaper** : projet et interface à {RATE // 1000} kHz (le fichier source "
        "doit avoir la fréquence du projet, sinon Reaper le rééchantillonne à la "
        "lecture) ; enregistrement en WAV 24 bits. Piste source : volume 0 dB, "
        "aucun effet, envoi au master **décoché**, sortie matérielle 1 (jack gauche "
        "de la sortie moniteur) ; master sans effet. Deux pistes mono armées sur les "
        "entrées 1 et 2, monitoring d'entrée coupé, aucun effet d'entrée. Lancer "
        "l'enregistrement une seconde avant la lecture, l'arrêter après les blips de "
        "fin. Une prise par ligne du tableau, dans l'ordre ; noter toute prise "
        "refaite.",
        "- Reverb Clean et OD à **0**. Power Dimension **éteint** sauf pour le dernier "
        "bloc. Retour de boucle vide.",
        "- **Test de 2 minutes** : note tenue ; bouger le master, puis Power "
        "Dimension ; noter laquelle des deux sorties change.",
        "  - FX Send dépend du master : oui / non. De Power Dimension : oui / non.",
        "  - Sortie émulée dépend du master : oui / non. De Power Dimension : "
        "oui / non.",
        "- **Niveau d'entrée de l'ampli** : viser 0 dBFS du fichier = 0 dBu à l'entrée "
        "de l'ampli (crêtes DI vers 1 V, comme un humbucker attaqué franchement), "
        "fourchette acceptable -6 à 0 dBu.",
        "  - D'après les fiches : niveau (dBu) = +20 (Zen Go, 0 dBFS, volume moniteur "
        "au maximum) + volume moniteur (dB, lu dans le panneau de contrôle) + gain de "
        "la boîte (dB, fiche, à la position choisie). Viser volume + boîte = -20 dB. "
        "Valable seulement si le volume se lit en dB et si la boîte est à une "
        "position documentée ; sinon, méthode suivante.",
        "  - Sans fiche : enregistrer la guitare sur une entrée en mode Hi-Z, puis "
        "brancher la sortie de la boîte sur cette même entrée, même gain, et régler "
        "la boîte pour que le programme DI y crête au même niveau.",
        "- Puis, prise 1 (centre) : aucune voie enregistrée ne doit écrêter, crêtes "
        "vers -6 dBFS. **Ne plus toucher à aucun niveau** ensuite.",
        "",
        "Valeurs fixes, à noter une fois :",
        "",
        "| | valeur |",
        "|---|---|",
        "| Volume OD2 | |",
        "| Volume Clean | |",
        "| Master (s'il existe) | |",
        "| Niveau de départ de boucle (FX level) | |",
        "| Volume moniteur de la Zen Go (dB, panneau de contrôle) | |",
        "| Boîte de réamplification : modèle, position, gain de la fiche (dB) | |",
        "| Calibration : dBu à l'entrée de l'ampli pour 0 dBFS (visé 0 dBu) | |",
        "| Méthode de calibration (fiches / guitare) | |",
        "| Gains et modes des deux entrées de la Zen Go | |",
        "| Taille de tampon | |",
        "",
        "## Prises, dans cet ordre",
        "",
        "Boutons de 0 à 10. Revenir au 5 des prises « centre » **depuis un côté "
        "différent** à chaque fois. Une photo du panneau par prise. Nommer chaque "
        "fichier comme dans la dernière colonne (suffixe de voie à ajouter : `_send`, "
        "`_line`).",
        "",
        "| # | fait | canal | gain | contour | grave | médium | aigu | "
        "tone shift | power dim. | rôle | fichier |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['order']} | ☐ | {r['channel']} | {knob(r['G'])} | {knob(r['C'])} | "
            f"{knob(r['B'])} | {knob(r['M'])} | {knob(r['T'])} | {r['tone_shift']} | "
            f"{r['power_dimension']} | {r['block'].replace('_', ' ')} ({r['split']}) | "
            f"`{r['file']}` |"
        )
    lines += [
        "",
        "Rôles : *one knob* et *combination train* servent à l'entraînement ; "
        "*combination test* n'y entre jamais ; *center* trois fois pour la dérive ; "
        "*clean* et *power dimension* sont des contrôles.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    signal, layout = reamp()
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIO_DIR / f"reamp_valvestate_{RATE // 1000}k.wav"
    sf.write(path, signal, RATE, subtype="PCM_24")
    written, _ = sf.read(path, dtype="float32")
    layout |= {
        "file": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seconds": round(len(written) / RATE, 3),
        "di_source": "ToneTwist AFx DRY-with-markers (Zenodo 10455730, CC BY-NC 4.0)",
    }
    found = self_check(written)
    assert found == 777, f"self-check found the blip at {found}, not 777"
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    (SHEET_DIR / "layout.json").write_text(json.dumps(layout, indent=1) + "\n")
    rows = settings()
    fields = [
        "order",
        "block",
        "channel",
        "G",
        "C",
        "B",
        "M",
        "T",
        "tone_shift",
        "power_dimension",
        "split",
        "setting",
        "file",
    ]
    with (SHEET_DIR / "settings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (SHEET_DIR / "SESSION.md").write_text(session_sheet(rows, layout), encoding="utf-8")
    print(f"{path} : {layout['seconds']} s, sha256 {layout['sha256'][:12]}")
    print(f"{len(rows)} reglages -> {SHEET_DIR / 'settings.csv'}")


if __name__ == "__main__":
    main()
