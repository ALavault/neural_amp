#!/usr/bin/env python3
"""Settings and session sheet for the mic'd VS100R session. Product lane.

About 20 OD2 settings in about 3 h, with the source of product_valvestate_mic_source.py.
Inputs: 1 = loopback (model input, as in September), 2 = microphone (target). With the
microphone, bass, middle and treble act on the target (the September FX send was taken
before the tone stack). Fifteen training settings (centre three times for drift, gain,
contour and each tone knob one at a time, two more at gain 2.5, one combination) and
five test settings never trained on: September's take 16 setting, recorded twice, and
the four gain-contour pairs September never recorded. Added 2026-10-07: three takes of
the short probe source at the centre setting: cold (first take, amp just switched on),
warm (after the last centre take, valve at equilibrium), then hot (right after a repeat
of the loudest setting), to separate thermal drift (valve, speaker, power amp) from the
settings. The repeat is also a second take of a test setting.

Writes demo/capture/valvestate_mic/settings.csv and SESSION.md.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import product_valvestate_session as session

ROOT = session.ROOT
SHEET_DIR = ROOT / "demo/capture/valvestate_mic"
SEED = 20261006
KNOBS = session.KNOBS


def setting(g, c, b=5, m=5, t=5) -> dict:
    return dict(zip(KNOBS, (g, c, b, m, t), strict=True))


CENTER = setting(5, 5)
TRAIN = [
    ("gain", setting(2.5, 5)),
    ("gain", setting(7.5, 5)),
    ("gain", setting(10, 5)),
    ("contour", setting(5, 0)),
    ("contour", setting(5, 10)),
    ("tone", setting(5, 5, b=0)),
    ("tone", setting(5, 5, b=10)),
    ("tone", setting(5, 5, m=0)),
    ("tone", setting(5, 5, m=10)),
    ("tone", setting(5, 5, t=0)),
    ("tone", setting(5, 5, t=10)),
    ("gain_2.5", setting(2.5, 2.5)),
    ("gain_2.5", setting(2.5, 5, b=7.5, m=2.5, t=7.5)),
    ("combination", setting(7.5, 2.5, b=7.5, m=2.5, t=5)),
]
REPEATED_TEST = setting(2.5, 7.5, b=2.5, m=7.5, t=7.5)  # September take 16
TEST = [
    ("unseen_pair", setting(2.5, 10)),
    ("unseen_pair", setting(7.5, 7.5)),
    ("unseen_pair", setting(10, 10)),
    ("unseen_pair", setting(10, 2.5)),
]


def settings() -> list[dict]:
    september = {
        r["setting"]
        for r in csv.DictReader((session.SHEET_DIR / "settings.csv").open())
    }
    rows = [(b, s, "train") for b, s in TRAIN] + [(b, s, "test") for b, s in TEST]
    rng = random.Random(SEED)
    rng.shuffle(rows)
    half = len(rows) // 2
    anchor = ("center", CENTER, "train")
    repeat = ("sept_take16", REPEATED_TEST, "test")
    first, second = rows[:half], rows[half:]
    first.insert(rng.randrange(1, len(first)), repeat)
    second.insert(rng.randrange(1, len(second)), repeat)
    ordered = [anchor, *first, anchor, *second, anchor]
    # Probe takes (short source, centre), added after the draw, so the order of the
    # other takes is unchanged: cold first; warm, a repeat of the loudest setting and
    # hot at the end, with the valve at equilibrium for both warm and hot.
    ordered = [
        ("probe_cold", CENTER, "probe"),
        *ordered,
        ("probe_warm", CENTER, "probe"),
        ("loud_repeat", setting(10, 10), "test"),
        ("probe_hot", CENTER, "probe"),
    ]
    out = []
    for i, (block, s, split) in enumerate(ordered, 1):
        name = session.name(s)
        probe = split == "probe"
        out.append(
            {
                "order": i,
                "block": block,
                "split": split,
                "source": "probe" if probe else "main",
                "setting": name,
                "in_september": name in september,
                **s,
                "file": f"{name}.vs100r_mic_{i:02d}{'_probe' if probe else ''}.wav",
            }
        )
    return out


def sheet(rows: list[dict], layout: dict, probe: dict) -> str:
    rate = layout["sample_rate"] // 1000
    loudest = next(
        r["order"] for r in rows if r["setting"] == "G100_C100_B050_M050_T050"
    )
    takes = sum(r["source"] == "main" for r in rows)
    probes = {r["block"]: r["order"] for r in rows if r["source"] == "probe"}
    lines = [
        "# Séance de capture au micro : Marshall Valvestate VS100R",
        "",
        "Voie produit. Généré par `scripts/product_valvestate_mic_session.py` (graine "
        f"{SEED}) ; ne pas éditer à la main, relancer le script.",
        "",
        f"{takes} prises de {layout['seconds'] / 60:.1f} min sur "
        f"{len({r['setting'] for r in rows})} réglages, plus {len(probes)} prises de "
        "la "
        f"sonde courte ({probe['seconds'] / 60:.1f} min) à froid, tiède et à chaud ; "
        "environ 3 h 20 avec les changements de réglage et l'installation.",
        "",
        "## Chaîne",
        "",
        f"- Source : `{layout['file']}`, {layout['seconds']} s, {rate} kHz, 24 bits, "
        f"sha256 `{layout['sha256'][:16]}…`. Sections et positions : "
        "`demo/capture/valvestate_mic/layout.json`. Les 212 premières secondes sont la "
        "source de septembre, identique ; suivent la polarité, les niveaux, les "
        "salves et les deux sons (100 Hz + 2 kHz à 4:1, 1000 + 1100 Hz, à -20, -10 et "
        "-3 dBFS).",
        f"- Sonde courte : `{probe['file']}`, {probe['seconds']} s, sha256 "
        f"`{probe['sha256'][:16]}…` (`layout_probe.json`) : mêmes marqueurs, silence, "
        "balayages, deux sons et pincements de la source principale.",
        f"- Interface : Antelope Zen Go Synergy Core, **{rate} kHz**, 24 bits, taille "
        "de tampon fixe pour toute la séance. **Tous les effets Synergy Core et le "
        "monitoring direct coupés** sur les entrées enregistrées.",
        "- **Entrée 1 = bouclage**, mode ligne, 0 dB, comme en septembre : c'est "
        "l'entrée des modèles. Noter d'où il est pris (sortie de la Zen Go avant la "
        "boîte de réamplification, ou sortie de la boîte).",
        "- **Entrée 2 = micro**, mode micro (alimentation fantôme si micro statique). "
        "C'est la cible.",
        "- Sortie moniteur (jack TRS) → boîte de réamplification → entrée de l'ampli, "
        "comme en septembre. Noter le modèle et les réglages de la boîte (non notés en "
        "septembre).",
        "- **Reaper** : projet à la fréquence de la source (sinon Reaper la "
        "rééchantillonne) ; WAV 24 bits ; piste source à 0 dB, sans effet, envoi au "
        "master décoché, sortie matérielle 1 ; deux pistes mono armées sur les entrées "
        "1 et 2, monitoring d'entrée coupé. Lancer lecture et enregistrement ensemble, "
        "arrêter après la fin du fichier.",
        "",
        "## Ampli, micro et pièce",
        "",
        "- Canal **OD2**. Reverb à **0**. Power Dimension **éteint**. Retour de boucle "
        "vide.",
        "- **Volume OD2 et master fixes pour toute la séance**, à un niveau de jeu "
        "habituel. Le haut-parleur et l'ampli de puissance font partie de la cible : "
        "un autre volume serait un autre appareil.",
        "- **Micro** : position fixe pour toute la séance. Mesurer la distance à la "
        "grille (cm), la position sur le cône (centre, mi-rayon, bord) et l'angle ; "
        "marquer l'emplacement au ruban, photographier. Retard acoustique attendu : "
        "distance / 343 m/s (10 cm ≈ 0,29 ms ≈ 56 échantillons à 192 kHz).",
        "- **Pièce** : fermée, mêmes conditions toute la séance ; noter les bruits "
        "notables (chauffage, circulation) et à quelle prise. Chaque prise contient "
        "déjà 5 s de silence pour le bruit de fond.",
        "- Si le micro ou l'ampli bouge pendant la séance : le noter, remettre en "
        "place sur les marques, et refaire une prise centre avant de continuer.",
        "",
        "## Froid, chaud, tiède : la sonde courte",
        "",
        "La sonde, au réglage centre, trois fois : à froid (prise "
        f"{probes['probe_cold']}), tiède en fin de séance (prise "
        f"{probes['probe_warm']}, lampe à l'équilibre), puis à chaud (prise "
        f"{probes['probe_hot']}) juste après une reprise du réglage le plus fort "
        f"(prise {probes['probe_hot'] - 1}). Froid face à tiède : la chauffe de "
        "l'ampli ; tiède "
        "face à chaud : l'échauffement du haut-parleur et de l'ampli de puissance par "
        "un passage fort.",
        "",
        "- **Ampli éteint depuis au moins 2 h** avant la séance. Noter la température "
        "de "
        "la pièce.",
        "- Le test des deux entrées et le niveau d'entrée de l'ampli se font **ampli "
        "éteint** : ils ne passent pas par l'ampli (mesurer à la sortie de la boîte).",
        "- **Allumer l'ampli et noter l'heure.** Après 2 min (chauffe minimale de la "
        "lampe), régler le gain du micro (15 s de jeu fort au plus), puis enregistrer "
        f"**aussitôt** la prise {probes['probe_cold']}.",
        f"- Prises {probes['probe_warm']} à {probes['probe_hot']} **enchaînées sans "
        f"pause** ; après la prise {probes['probe_hot'] - 1} (gain 10, contour 10), "
        "remettre le centre en moins d'une minute.",
        "- Noter l'heure de début de chaque prise de sonde dans le tableau.",
        "",
        "## Test des deux entrées, avant de régler les niveaux (5 min, ampli éteint)",
        "",
        "Les trois prises A, B et C de `demo/capture/valvestate/INTER_INPUT_TEST.md` "
        "(un signal partagé sur les deux entrées en mode ligne), jamais faites depuis "
        "septembre. Elles tranchent le décalage d'un échantillon des modèles de "
        "septembre. Il change le niveau de sortie : le faire avant le réglage des "
        "niveaux ci-dessous, puis remettre l'entrée 2 en mode micro.",
        "",
        "## Niveaux, réglés une fois",
        "",
        "- **Entrée de l'ampli** : mêmes réglages de sortie et de boîte qu'en "
        "septembre si on les retrouve (0 dBFS de la source vers 0 dBu à l'entrée de "
        "l'ampli, méthode dans `demo/capture/valvestate/SESSION.md`) ; sinon les "
        "régler ainsi et les noter.",
        "- **Gain du micro**, ampli allumé depuis 2 min, juste avant la première "
        "prise : ampli au réglage probablement le plus fort du plan (gain 10, contour "
        f"10, celui de la prise {loudest}), jouer la section « niveau +6 dB » de la "
        "source (255 à 265 s), 15 s au plus : crêtes de l'entrée 2 vers -10 dBFS, "
        "jamais d'écrêtage. **Ne plus toucher à aucun niveau ni volume** ensuite.",
        "",
        "Valeurs fixes, à noter une fois :",
        "",
        "| | valeur |",
        "|---|---|",
        "| Volume OD2 | |",
        "| Master | |",
        "| Micro : modèle, distance (cm), position sur le cône, angle | |",
        "| Haut-parleur et baffle (d'origine ?) | |",
        "| Pièce | |",
        "| Boîte de réamplification : modèle, réglages | |",
        "| Bouclage : pris avant ou après la boîte | |",
        "| Volume moniteur de la Zen Go (dB) | |",
        "| Gains et modes des deux entrées | |",
        "| Taille de tampon | |",
        "| Ampli éteint depuis (h), heure de mise sous tension | |",
        "| Température de la pièce, début et fin | |",
        "",
        "## Prises, dans cet ordre",
        "",
        "Boutons de 0 à 10. Revenir au 5 des prises « centre » **depuis un côté "
        "différent** à chaque fois. Une photo du panneau par prise. Nommer chaque "
        "fichier comme dans la dernière colonne (suffixe de voie : `_loop`, `_mic`). "
        "Noter toute prise refaite.",
        "",
        "| # | fait | source | gain | contour | grave | médium | aigu | rôle | "
        "en sept. | fichier | heure |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        source = "sonde" if r["source"] == "probe" else "principale"
        lines.append(
            f"| {r['order']} | ☐ | {source} | {session.knob(r['G'])} | "
            f"{session.knob(r['C'])} | {session.knob(r['B'])} | "
            f"{session.knob(r['M'])} | {session.knob(r['T'])} | "
            f"{r['block'].replace('_', ' ')} ({r['split']}) | "
            f"{'oui' if r['in_september'] else ''} | `{r['file']}` | |"
        )
    lines += [
        "",
        "Rôles : *train* sert à l'entraînement ; *test* n'y entre jamais. *center* "
        "trois fois pour la dérive. *sept take16* : le réglage de la prise 16 de "
        "septembre, deux fois, pour savoir si cette prise était représentative. "
        "*unseen pair* : couples gain-contour jamais enregistrés. « en sept. » : le "
        "même réglage existe dans la capture de septembre (départ de boucle). *probe "
        "cold / warm / hot* : la sonde courte, voir plus haut ; *loud repeat* : "
        "seconde "
        "prise du réglage le plus fort, juste avant la sonde chaude.",
        "",
        "## Après la séance",
        "",
        "Déposer les fichiers dans `datasets/amp_capture_mic/` (non commité), avec les "
        "photos et ce tableau rempli. Les scripts de traitement de septembre lisent "
        "l'ancien layout : à rediriger avant de traiter ces prises.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    layout = json.loads((SHEET_DIR / "layout.json").read_text())
    probe = json.loads((SHEET_DIR / "layout_probe.json").read_text())
    rows = settings()
    fields = ["order", "block", "split", "source", "setting", "in_september", *KNOBS]
    fields.append("file")
    with (SHEET_DIR / "settings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (SHEET_DIR / "SESSION.md").write_text(sheet(rows, layout, probe), encoding="utf-8")
    print(
        f"{len(rows)} prises, {len({r['setting'] for r in rows})} réglages, "
        f"{sum(r['in_september'] for r in rows)} prises à un réglage de septembre"
    )


if __name__ == "__main__":
    main()
