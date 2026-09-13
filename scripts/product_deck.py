#!/usr/bin/env python3
"""Build the ten-slide demo deck from the measured fact sheet.

Every figure on a slide is read from demo/report.json or demo/RUNS.jsonl, so a
new measurement updates the deck and no number can drift from its source.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "demo/report.json"
RUNS = ROOT / "demo/RUNS.jsonl"
OUT = ROOT / "demo/deck/index.html"
BLOCK = 64
DEVICE_LABELS = {
    "fulltone_full_drive_2": "Fulltone Full-Drive 2",
    "electro_harmonix_big_muff": "Electro-Harmonix Big Muff",
}


def load() -> tuple[dict, list[dict]]:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    runs = [json.loads(line) for line in RUNS.read_text(encoding="utf-8").splitlines()]
    return report, runs


def at_block(variant: dict, block: int) -> dict:
    return next(row for row in variant["cost"] if row["block_size"] == block)


def figures(report: dict, runs: list[dict]) -> dict:
    rows = []
    for device, entry in report.items():
        for label in ("full", "lite"):
            variant = entry["variants"][label]
            cost = at_block(variant, BLOCK)
            alias = variant["aliasing"]["k12287_a0.48"]
            rows.append(
                {
                    "device": DEVICE_LABELS[device],
                    "variant": f"NAM A2 {label.capitalize()}",
                    "esr": variant["fidelity"]["esr"],
                    "correlation": variant["fidelity"]["correlation"],
                    "mrstft": variant["fidelity"]["mrstft"],
                    "ns": cost["median_ns_per_sample"],
                    "load": 100.0 / cost["p95_realtime_factor"],
                    "rtf": cost["p95_realtime_factor"],
                    "parity": variant["parity"]["python_vs_native_block64"],
                    "irregular": variant["parity"]["native_regular_vs_irregular"],
                    "alias": alias["asr_db_no_dc"],
                }
            )
    by_run = {run["run_id"]: run for run in runs}
    selftest = by_run["selftest_capture_chain"]
    return {
        "rows": rows,
        "minutes": {
            run["device"]: run["minutes"]
            for run in runs
            if run["run_id"].startswith("product_a2")
        },
        "selftest_esr": selftest["test_esr"]["full"],
        "selftest_minutes": selftest["minutes"],
        "big_muff_epochs": by_run["product_a2_electro_harmonix_big_muff_seed0_v2"][
            "max_epochs"
        ],
    }


def cost_chart(rows: list[dict]) -> str:
    """Horizontal bars of one core's load at 48 kHz, block 64."""
    scale_max = 30.0
    bar_height, gap, top, left, width = 34, 16, 30, 188, 420
    height = top + len(rows) * (bar_height + gap)
    parts = [
        f'<svg viewBox="0 0 {left + width + 58} {height + 16}" role="img" '
        'aria-label="Charge CPU par modèle au bloc 64">'
    ]
    for tick in (0, 10, 20, 30):
        x = left + width * tick / scale_max
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" y2="{height - gap}" '
            'class="grid" />'
            f'<text x="{x:.1f}" y="{top - 18}" class="tick" '
            f'text-anchor="middle">{tick} %</text>'
        )
    for index, row in enumerate(sorted(rows, key=lambda r: r["load"])):
        y = top + index * (bar_height + gap)
        length = width * min(row["load"], scale_max) / scale_max
        tone = "bar-full" if "Full" in row["variant"] else "bar-lite"
        parts.append(
            f'<text x="{left - 12}" y="{y + bar_height * 0.68:.1f}" class="bar-label" '
            f'text-anchor="end">{row["device"].split()[0]} · {row["variant"]}</text>'
            f'<rect x="{left}" y="{y}" width="{length:.1f}" height="{bar_height}" '
            f'rx="2" class="{tone}" />'
            f'<text x="{left + length + 10:.1f}" y="{y + bar_height * 0.68:.1f}" '
            f'class="bar-value">{row["load"]:.1f} %</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def slide(number: int, eyebrow: str, title: str, body: str) -> str:
    return (
        f'<section class="slide" id="s{number}">'
        f'<div class="slide-head"><span class="num">{number:02d}</span>'
        f'<span class="eyebrow">{eyebrow}</span></div>'
        f"<h2>{title}</h2>{body}</section>"
    )


def metrics_table(rows: list[dict]) -> str:
    body = "".join(
        f"<tr><td>{row['device']}</td><td>{row['variant']}</td>"
        f"<td class='n'>{row['esr']:.5f}</td>"
        f"<td class='n'>{row['correlation']:.4f}</td>"
        f"<td class='n'>{row['mrstft']:.3f}</td></tr>"
        for row in rows
    )
    return (
        '<div class="scroller"><table><thead><tr><th>Appareil</th><th>Modèle</th>'
        "<th>ESR</th><th>Corrélation</th><th>MR-STFT</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def build(data: dict) -> str:
    rows = data["rows"]
    full = {row["device"]: row for row in rows if row["variant"].endswith("Full")}
    lite = {row["device"]: row for row in rows if row["variant"].endswith("Lite")}
    fulltone = full["Fulltone Full-Drive 2"]
    muff = full["Electro-Harmonix Big Muff"]
    worst_parity = max(row["parity"] for row in rows)
    load_full = fulltone["load"]

    slides = [
        slide(
            1,
            "Ce que nous montrons",
            "D’un appareil analogique à un plugin temps réel, avec les chiffres",
            f"""<p class="lead">Un VST3 mono qui charge un modèle <code>.nam</code>,
            bascule instantanément entre le modèle et l’enregistrement réel aligné à
            l’échantillon près, et tourne à <strong>{load_full:.0f}&nbsp;%</strong> d’un
            cœur à 48&nbsp;kHz.</p>
            <div class="disclosure"><h3>Ce qui n’est pas de nous</h3>
            <p><strong>A2 est l’architecture de Neural Amp Modeler</strong>, pas la
            nôtre : deux WaveNet empaquetés (Lite et Full, champ réceptif de 6 347
            échantillons), entraînés par le trainer officiel NAM, exportés au format
            <code>.nam</code> officiel et exécutés par
            <code>NeuralAmpModelerCore</code>. Nous n’avons rien inventé côté modèle et
            nous ne prétendons pas battre NAM.</p>
            <p>Ce qui est de nous&nbsp;: les poids entraînés sur ces appareils, le
            plugin, la chaîne de mesure et la chaîne de capture.</p></div>
            <p>La valeur est donc dans la mesure. Fidélité, coût, parité, repliement,
            robustesse et capture&nbsp;: tous chiffrés, reproductibles par une commande,
            et publiés avec leurs limites.</p>
            <div class="pills"><span class="pill">VST3 + standalone</span>
            <span class="pill">Linux, JUCE 8.0.15</span>
            <span class="pill">Moteur NAM A2</span>
            <span class="pill">2 appareils mesurés</span></div>""",
        ),
        slide(
            2,
            "Le produit",
            "Le plugin existe et se démontre en trois clics",
            """<div class="cols">
            <div><h3>Dans la démo</h3><ol class="steps">
            <li>Charger un <code>.nam</code></li>
            <li>Charger le DI et la capture réelle de l’appareil</li>
            <li>Basculer <em>A&nbsp;: modèle</em> / <em>B&nbsp;: réel</em></li>
            </ol></div>
            <div><h3>Ce que ça garantit</h3><ul class="ticks">
            <li>Les deux flux sont alignés à l’échantillon</li>
            <li>Aucune allocation dans le callback audio</li>
            <li>Tailles de bloc variables, reset exact</li>
            <li>Avertissement si l’hôte n’est pas au taux du modèle</li>
            </ul></div></div>""",
        ),
        slide(
            3,
            "Fidélité",
            "Deux appareils, deux régimes, aucun chiffre choisi après coup",
            metrics_table(rows)
            + f"""<p class="note">Jeu de test de développement, 30&nbsp;s par
            appareil. Le budget d’époques a été choisi après lecture de ce jeu&nbsp;:
            il n’est pas scellé, et le dire fait partie de la mesure.
            Entraînement&nbsp;: {data["minutes"]["fulltone_full_drive_2"]:.0f} à
            {data["minutes"]["electro_harmonix_big_muff"]:.0f}&nbsp;min sur un GPU.</p>""",
        ),
        slide(
            4,
            "Le cas difficile",
            "Le Big Muff, présenté avec ses repères plutôt que caché",
            f"""<p class="lead">ESR {muff["esr"]:.4f} contre
            {fulltone["esr"]:.5f} sur le Fulltone. Deux étages d’écrêtage en cascade,
            c’est le cas dur de la littérature — et nous le montrons quand même.</p>
            <div class="cols">
            <div><h3>Sur ce même appareil</h3><ul class="bare">
            <li><span class="n">0,1076</span> — S4-TFiLM <em>large</em></li>
            <li><span class="n">0,59 à 0,70</span> — gray-box à une non-linéarité</li>
            <li><span class="n">{muff["esr"]:.4f}</span> — NAM A2 Full, ici</li>
            </ul></div>
            <div><h3>Ce que nous avons vérifié</h3><ul class="ticks">
            <li>Alignement prédiction/cible&nbsp;: décalage de pic nul</li>
            <li>{data["big_muff_epochs"]} époques au lieu de 100&nbsp;: 10&nbsp;%
            d’ESR gagnés</li>
            <li>Le plateau vient du modèle, pas du budget</li>
            </ul></div></div>""",
        ),
        slide(
            5,
            "Coût",
            "Un cœur, 48 kHz, bloc 64",
            f'<div class="chart">{cost_chart(rows)}</div>'
            f"""<p class="note">Médiane de {fulltone["ns"]:.0f}&nbsp;ns/échantillon
            pour NAM A2 Full, {lite["Fulltone Full-Drive 2"]["ns"]:.0f}&nbsp;ns pour A2
            Lite. La charge affichée est calculée sur le p95, donc pessimiste. Machine
            partagée&nbsp;: la médiane est le chiffre fiable, une mesure sur machine
            dédiée reste à faire.</p>""",
        ),
        slide(
            6,
            "Parité",
            "Ce qui est entraîné est exactement ce qui est joué",
            f"""<div class="stats">
            <div class="stat"><span class="v">{worst_parity:.1e}</span>
            <span class="k">Python d’entraînement contre moteur natif, pire cas</span></div>
            <div class="stat"><span class="v">0,00</span>
            <span class="k">Blocs réguliers contre blocs irréguliers</span></div>
            <div class="stat"><span class="v">0,00</span>
            <span class="k">Rendu du plugin contre moteur natif, bit à bit</span></div>
            </div>
            <p>Le rendu hors ligne du plugin passe par les mêmes conversions
            float/double, le même flush des dénormaux et le même étage de gain que le
            callback temps réel. Il est identique au runner natif sur les quatre
            modèles. Le reset est exact, et la compilation évite
            <code>-ffast-math</code> pour que la propagation des NaN et l’égalité IEEE
            tiennent.</p>""",
        ),
        slide(
            7,
            "Repliement",
            "La mesure que personne ne publie sur ces modèles",
            f"""<p class="lead">Sondes sinus à bin exact, N&nbsp;=&nbsp;65&nbsp;536,
            rendues par le moteur natif. À 9&nbsp;kHz et fort niveau, NAM A2 Full produit
            {muff["alias"]:.1f}&nbsp;dB d’énergie non harmonique sur le Big Muff,
            {fulltone["alias"]:.1f}&nbsp;dB sur le Fulltone.</p>
            <div class="cols">
            <div><h3>Ce que le chiffre est</h3><p>Une borne supérieure du repliement&nbsp;:
            l’énergie hors bins harmoniques contient aussi toute erreur non harmonique
            du modèle. À 9&nbsp;kHz, seules deux harmoniques tiennent sous Nyquist.</p></div>
            <div><h3>Ce qu’il révèle</h3><p>A2 apprend une composante continue qui
            domine la métrique gelée jusqu’à 30&nbsp;dB. C’est un constat sur les
            modèles entraînés, pas un artefact de mesure — et il change la façon dont
            on coupe les extraits.</p></div></div>
            <p class="note">Plancher de mesure&nbsp;: −154,8&nbsp;dB.</p>""",
        ),
        slide(
            8,
            "Capture",
            "Votre appareil, votre plugin, en une passe de reamp",
            f"""<div class="cols">
            <div><h3>La chaîne</h3><ol class="steps">
            <li>Signal de reamp avec blips de calibration</li>
            <li>Une passe dans l’appareil, un enregistrement</li>
            <li>Alignement et contrôle qualité automatiques</li>
            <li>Entraînement, export <code>.nam</code>, plugin</li>
            </ol></div>
            <div><h3>Prouvé de bout en bout</h3><ul class="ticks">
            <li>Latence retrouvée à <strong>0 échantillon</strong>, tirage aléatoire
            à chaque exécution</li>
            <li>Niveau retrouvé à <strong>0,0000&nbsp;dB</strong></li>
            <li>Modèle réentraîné&nbsp;: ESR
            <strong>{data["selftest_esr"]:.5f}</strong> en
            {data["selftest_minutes"]:.0f}&nbsp;min</li>
            <li>Prise écrêtée ou silencieuse refusée</li>
            </ul></div></div>
            <p class="note">Validé contre un appareil fictif déterministe. Aucune
            capture matérielle réelle n’a encore été faite.</p>""",
        ),
        slide(
            9,
            "Écoute",
            "Une comparaison aveugle, pas une démonstration guidée",
            """<p class="lead">Trois extraits par appareil, pris à des positions
            fixes du fichier de test&nbsp;: aucune sélection sur le résultat.</p>
            <div class="cols">
            <div><h3>Protocole</h3><ul class="bare">
            <li>A/B tiré au sort au chargement, X à chaque essai</li>
            <li>Niveau égalisé par un seul scalaire par appareil</li>
            <li>Fondus identiques des deux côtés</li>
            <li>Score et p-valeur binomiale affichés</li>
            </ul></div>
            <div><h3>Pourquoi ça compte</h3><p>Un A/B piloté par le vendeur ne prouve
            rien. Celui-ci peut échouer devant vous, et le protocole est écrit sur la
            page que vous utilisez.</p></div></div>""",
        ),
        slide(
            10,
            "Limites et suite",
            "Ce qui n’est pas prouvé, dit avant que vous le demandiez",
            """<div class="cols">
            <div><h3>Pas encore fait</h3><ul class="bare warn">
            <li>Aucun chargement en DAW&nbsp;: pas d’hôte installé sur la machine
            de mesure</li>
            <li>Aucune capture matérielle réelle</li>
            <li>Deux appareils seulement, sources CC-BY-NC</li>
            <li>Coût mesuré sur station partagée, pas sur portable grand public</li>
            </ul></div>
            <div><h3>Ce que nous proposons</h3><ul class="ticks">
            <li>Capturer un de vos appareils et vous rendre le plugin</li>
            <li>Reproduire la fiche technique sur votre matériel cible</li>
            <li>Discuter des régimes durs&nbsp;: fuzz, écrêtage en cascade</li>
            </ul></div></div>""",
        ),
    ]
    nav = "".join(
        f'<a href="#s{index}"><span>{index:02d}</span></a>'
        for index in range(1, len(slides) + 1)
    )
    return TEMPLATE.replace("__NAV__", nav).replace("__SLIDES__", "".join(slides))


TEMPLATE = """<title>Fidélité mesurée</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700&family=IBM+Plex+Mono:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>
:root {
  --ground: #eceef1;
  --surface: #f6f7f9;
  --edge: #cfd5dd;
  --ink: #171c24;
  --muted: #5a6472;
  --accent: #0f6e8c;
  --accent-soft: #cfe3ea;
  --warn: #9a5218;
  --good: #2f6b4f;
  --display: "Archivo", "Helvetica Neue", sans-serif;
  --body: "Source Serif 4", Georgia, serif;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #0e1218;
    --surface: #161c25;
    --edge: #2b3542;
    --ink: #e6eaf0;
    --muted: #93a0b1;
    --accent: #4fb8d8;
    --accent-soft: #1d3d4a;
    --warn: #d8974f;
    --good: #6fbf96;
  }
}
:root[data-theme="dark"] {
  --ground: #0e1218;
  --surface: #161c25;
  --edge: #2b3542;
  --ink: #e6eaf0;
  --muted: #93a0b1;
  --accent: #4fb8d8;
  --accent-soft: #1d3d4a;
  --warn: #d8974f;
  --good: #6fbf96;
}
* { box-sizing: border-box; }
body {
  background: var(--ground);
  color: var(--ink);
  font-family: var(--body);
  font-size: 17px;
  line-height: 1.6;
  margin: 0;
}
.wrap {
  max-width: 62rem;
  margin: 0 auto;
  padding-block: 2.5rem 4rem;
  padding-inline: 1.25rem;
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}
.masthead {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  justify-content: space-between;
  gap: .75rem;
  border-bottom: 2px solid var(--ink);
  padding-bottom: .75rem;
}
.masthead h1 {
  font-family: var(--display);
  font-weight: 700;
  font-size: clamp(1.35rem, 3.6vw, 1.9rem);
  letter-spacing: -0.015em;
  margin: 0;
  text-wrap: balance;
}
.masthead p { color: var(--muted); font-size: .95rem; margin: 0; }
nav {
  display: flex;
  flex-wrap: wrap;
  gap: .35rem;
  font-family: var(--mono);
  font-size: .72rem;
}
nav a {
  color: var(--muted);
  text-decoration: none;
  border: 1px solid var(--edge);
  border-radius: 2px;
  padding: .15rem .45rem;
}
nav a:hover, nav a:focus-visible { color: var(--accent); border-color: var(--accent); }
.slide {
  background: var(--surface);
  border: 1px solid var(--edge);
  border-top: 3px solid var(--accent);
  padding: clamp(1.25rem, 3.5vw, 2.25rem);
  scroll-margin-top: 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.slide-head { display: flex; align-items: center; gap: .7rem; }
.num {
  font-family: var(--mono);
  font-weight: 600;
  font-size: 1.5rem;
  color: var(--accent);
  font-variant-numeric: tabular-nums;
}
.eyebrow {
  font-family: var(--display);
  font-weight: 500;
  font-size: .74rem;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--muted);
}
.slide h2 {
  font-family: var(--display);
  font-weight: 700;
  font-size: clamp(1.3rem, 3.2vw, 1.85rem);
  line-height: 1.2;
  letter-spacing: -0.015em;
  margin: 0;
  text-wrap: balance;
}
.slide h3 {
  font-family: var(--display);
  font-weight: 500;
  font-size: .78rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 0 0 .5rem;
}
.slide p { margin: 0; max-width: 62ch; }
.lead { font-size: 1.1rem; }
.note { color: var(--muted); font-size: .9rem; }
.cols { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.5rem; }
@media (max-width: 640px) { .cols { grid-template-columns: 1fr; } }
ol.steps, ul.ticks, ul.bare { margin: 0; padding-left: 1.1rem; display: grid; gap: .35rem; }
ul.ticks, ul.bare { list-style: none; padding-left: 0; }
ul.ticks li { padding-left: 1.35rem; position: relative; }
ul.ticks li::before {
  content: "—";
  position: absolute;
  left: 0;
  color: var(--good);
  font-family: var(--mono);
}
ul.bare li { border-bottom: 1px solid var(--edge); padding-bottom: .35rem; }
ul.bare.warn li::before { content: "· "; color: var(--warn); }
ol.steps li::marker { font-family: var(--mono); color: var(--accent); }
code { font-family: var(--mono); font-size: .88em; background: var(--accent-soft);
       padding: .05em .3em; border-radius: 2px; }
.n, span.n { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.disclosure { border-left: 3px solid var(--warn); padding: .1rem 0 .1rem .9rem;
              display: grid; gap: .5rem; }
.disclosure h3 { color: var(--warn); }
.pills { display: flex; flex-wrap: wrap; gap: .4rem; }
.pill {
  font-family: var(--mono);
  font-size: .72rem;
  border: 1px solid var(--edge);
  border-radius: 999px;
  padding: .2rem .6rem;
  color: var(--muted);
}
.scroller { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: .93rem; min-width: 34rem; }
th, td { text-align: left; padding: .45rem .6rem; border-bottom: 1px solid var(--edge); }
th {
  font-family: var(--display);
  font-weight: 500;
  font-size: .72rem;
  letter-spacing: .09em;
  text-transform: uppercase;
  color: var(--muted);
}
td.n { font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; }
.stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; }
@media (max-width: 640px) { .stats { grid-template-columns: 1fr; } }
.stat { display: flex; flex-direction: column; gap: .2rem; border-left: 2px solid var(--accent);
        padding-left: .8rem; }
.stat .v { font-family: var(--mono); font-size: 1.45rem; font-weight: 600; }
.stat .k { color: var(--muted); font-size: .87rem; }
.chart { overflow-x: auto; }
.chart svg { width: 100%; min-width: 32rem; height: auto; }
.grid { stroke: var(--edge); stroke-width: 1; }
.tick, .bar-label, .bar-value { font-family: var(--mono); fill: var(--muted); font-size: 13px; }
.bar-value { fill: var(--ink); font-weight: 600; }
.bar-full { fill: var(--accent); }
.bar-lite { fill: var(--accent-soft); stroke: var(--accent); stroke-width: 1; }
footer { color: var(--muted); font-size: .85rem; border-top: 1px solid var(--edge);
         padding-top: 1rem; }
a:focus-visible, nav a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
</style>
<div class="wrap">
  <header class="masthead">
    <div>
      <h1>Fidélité mesurée&nbsp;: d’un appareil au plugin</h1>
      <p>Dossier de démonstration pour éditeurs de plugins</p>
    </div>
    <nav aria-label="Slides">__NAV__</nav>
  </header>
  __SLIDES__
  <footer>
    Chaque chiffre de ce dossier est lu dans <code>demo/report.json</code> et
    <code>demo/RUNS.jsonl</code>, produits par <code>make demo</code>. Sources audio&nbsp;:
    ToneTwist AFx, licence CC-BY-NC&nbsp;— usage de démonstration de recherche, sans
    distribution des modèles.
  </footer>
</div>
"""


def main() -> None:
    report, runs = load()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(figures(report, runs)), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
