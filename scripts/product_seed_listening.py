#!/usr/bin/env python3
"""Blind A/B/X page for the two extreme seeds: is a factor 5 in ESR audible?

The butterfly page asked whether a one-float32-step divergence could be heard and the
answer was no, on a gap reaching 0.93 of the model-to-device error. This asks the question
that matters for the bench: pilot C separates seed 42 (test ESR 0.036) from seed 49
(0.178), and on the three most divergent test segments their gap reaches 2.8 times the
error of the good model, sitting only 3.7 to 5.7 dB under the signal
(diagnosis/seeds/seed_gap_where.json). That should be plainly audible; this measures it.

Three pairs per excerpt: the device against the good seed, the device against the bad one,
and the two seeds against each other.

The ladder is an INTERPOLATION, not an extrapolation, because the gap is expected to be
audible rather than inaudible. Side B is (1 - f) * seed42 + f * seed49, so f = 1 is the
real seed-to-seed difference and smaller f walks toward seed 42. Each rung is a real
signal whose ESR against the device is measured and printed on the page, so the rung at
which the listener falls to chance reads directly as "audible down to this ESR".

Excerpts stay strictly inside one test segment, the protocol resetting the model state at
each boundary. Each model side gets its own least-squares gain against the target so a
level difference cannot give the answer away, and one normalisation per excerpt keeps the
comparison untouched.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import product_nablafx_bench as bench  # noqa: E402
from product_butterfly_listening import TEMPLATE, fade  # noqa: E402
from product_fork_pilot_analysis import outputs  # noqa: E402

OUT_DIR = ROOT / "demo/listening"
AUDIO_DIR = OUT_DIR / "audio"
SAMPLE_RATE = 48_000
WINDOW_SECONDS = 1.5
PEAK = 10 ** (-1 / 20)
STAMP = int(time.time())
MODELS = {
    "graine42": "pilotC_decide_seed42",
    "graine49": "pilotC_decide_seed49",
}
PAIRS = (
    ("reel", "graine42", "appareil réel contre la meilleure graine (ESR 0,036)"),
    ("reel", "graine49", "appareil réel contre la pire graine (ESR 0,178)"),
    ("graine42", "graine49", "la meilleure contre la pire : la question"),
)
LADDER = (1.0, 0.71, 0.5, 0.35, 0.25, 0.18)
LADDER_EXCERPT = 1

TITLE = "Écoute aveugle A/B/X : la meilleure graine contre la pire"
INTRO = """<h1>Écoute aveugle A/B/X : la meilleure graine contre la pire</h1>
<p>Le pilote C sépare la graine 42 (ESR de test <b>0,036</b>) de la graine 49
(<b>0,178</b>), un facteur 5. Sur les trois segments les plus divergents, l'écart entre
les deux modèles atteint <b>2,8 fois</b> l'erreur du bon modèle contre l'appareil et ne
siège qu'à <b>3,7 à 5,7 dB</b> sous le signal. Chez les enfants du papillon, ce rapport
plafonnait à 0,93 et l'écart vivait 16 à 25 dB plus bas : rien n'y était audible.</p>
<p>Trois paires par extrait. <b>Réel contre graine 42</b> dit si la meilleure graine se
distingue de l'appareil. <b>Réel contre graine 49</b> demande si la pire s'en distingue.
<b>Graine 42 contre graine 49</b> est la question : l'écart entre deux tirages de graine
s'entend-il ? Dix essais par bloc ; 20 justes sur 30 donnent p &lt; 0,05 pour un extrait.</p>
<p class="meta">Les extraits sont choisis <b>sur</b> la divergence, segment et fenêtre,
d'après <code>diagnosis/seeds/seed_gap_where.json</code>. C'est donc un meilleur cas : si
l'écart est inaudible ici, il l'est partout sur cet appareil. La première page d'écoute
avait pris ses segments par position seule et tombait sur les moins divergents.</p>
<p class="meta"><b>L'échelle descend au lieu de monter</b>, parce que l'écart est attendu
audible. Chaque barreau joue la graine 42 contre une interpolation
(1 − f) × graine 42 + f × graine 49 : f = 1 est l'écart réel entre les deux graines, et f
plus petit s'en rapproche. L'ESR de chaque barreau contre l'appareil est indiquée, donc le
barreau où vous retombez au hasard se lit directement comme « audible jusqu'à cette ESR ».</p>
<p class="meta">Le bouton <b>Différence</b> n'appartient pas à l'épreuve : il joue
graine 42 moins graine 49, normalisé, pour entendre <i>ce qui</i> sépare les deux.</p>
"""


def esr(signal: np.ndarray, target: np.ndarray) -> float:
    return float(((signal - target) ** 2).sum() / (target**2).sum())


def main() -> None:
    torch.set_num_threads(8)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    measured = {
        row["segment"]: row
        for row in json.loads(
            (ROOT / "diagnosis/seeds/seed_gap_where.json").read_text()
        )
    }
    segments = tuple(
        sorted(measured, key=lambda s: -measured[s]["gap_over_err_good"])[:3]
    )

    data = bench.data_module("test")
    data.setup("test")
    inputs = torch.stack([x for x, _ in data.test_dataset])
    targets = torch.stack([y for _, y in data.test_dataset])
    rendered = {
        name: outputs(run, inputs).numpy()[:, 0] for name, run in MODELS.items()
    }
    rendered["reel"] = targets.numpy()[:, 0]

    excerpts, ladder_clips = [], {}
    for index, segment in enumerate(segments, start=1):
        start = min(measured[segment]["best_second_at_s"], 5.0 - WINDOW_SECONDS - 0.05)
        span = slice(
            int(start * SAMPLE_RATE), int((start + WINDOW_SECONDS) * SAMPLE_RATE)
        )
        reference = rendered["reel"][segment][span].astype(np.float64)
        clips, gains = {}, {}
        for name, signal in rendered.items():
            clip = signal[segment][span].astype(np.float64)
            gain = 1.0 if name == "reel" else float(clip @ reference / (clip @ clip))
            clips[name] = fade(clip * gain)
            gains[name] = float(20.0 * np.log10(gain))
        loud = PEAK / max(float(np.max(np.abs(c))) for c in clips.values())
        clips = {name: clip * loud for name, clip in clips.items()}
        difference = clips["graine42"] - clips["graine49"]
        clips["difference"] = difference * (PEAK / float(np.max(np.abs(difference))))
        if index == LADDER_EXCERPT:
            ladder_clips = dict(clips)

        files = {}
        for name, clip in clips.items():
            files[name] = f"audio/seeds_{index}_{name}.wav"
            sf.write(OUT_DIR / files[name], clip, SAMPLE_RATE, subtype="PCM_16")
        files = {name: f"{path}?{STAMP}" for name, path in files.items()}
        row = measured[segment]
        for left, right, label in PAIRS:
            gap = clips[left] - clips[right]
            excerpts.append(
                {
                    "key": f"{index}_{left}_vs_{right}",
                    "index": index,
                    "segment": segment,
                    "window": f"segment {segment}, {start:.2f}-"
                    f"{start + WINDOW_SECONDS:.2f} s ; écart entre modèles à"
                    f" {row['gap_to_signal_db']:.1f} dB sous le signal, soit"
                    f" {row['gap_over_err_good']:.2f} fois l'erreur de la graine 42",
                    "label": label,
                    "left": files[left],
                    "right": files[right],
                    "names": [left, right],
                    "esr_between_sides": float(
                        gap @ gap / (clips[right] @ clips[right])
                    ),
                    "gain_db": [gains[left], gains[right]],
                    "difference": (
                        files["difference"] if right == "graine49" else None
                    ),
                }
            )

    good, bad = ladder_clips["graine42"], ladder_clips["graine49"]
    device = ladder_clips["reel"]
    for factor in LADDER:
        blend = (1.0 - factor) * good + factor * bad
        name = f"audio/seeds_ladder_f{factor:g}.wav"
        sf.write(OUT_DIR / name, blend, SAMPLE_RATE, subtype="PCM_16")
        excerpts.append(
            {
                "key": f"echelle_f{factor:g}",
                "index": LADDER_EXCERPT,
                "segment": segments[LADDER_EXCERPT - 1],
                "window": f"interpolation f = {factor:g} ; ESR du côté B contre"
                f" l'appareil {esr(blend, device):.3f}, contre"
                f" {esr(good, device):.3f} pour la graine 42",
                "label": f"échelle : graine 42 contre interpolation f = {factor:g}",
                "left": f"audio/seeds_{LADDER_EXCERPT}_graine42.wav?{STAMP}",
                "right": f"{name}?{STAMP}",
                "names": ["graine42", f"interpolation f = {factor:g}"],
                "esr_between_sides": float(
                    (good - blend) @ (good - blend) / (blend @ blend)
                ),
                "gain_db": [0.0, 0.0],
                "difference": None,
            }
        )

    page = TEMPLATE.replace(
        "<title>Écoute aveugle A/B/X : les enfants du papillon</title>",
        f"<title>{TITLE}</title>",
    )
    head, tail = page.split('<div id="excerpts"></div>', 1)
    page = head[: head.index("<h1>")] + INTRO + '<div id="excerpts"></div>' + tail
    OUT_DIR.joinpath("seeds.html").write_text(
        page.replace("__EXCERPTS__", json.dumps(excerpts, ensure_ascii=False)),
        encoding="utf-8",
    )
    for excerpt in excerpts:
        print(
            f"{excerpt['key']}: ESR entre les deux côtés"
            f" {excerpt['esr_between_sides']:.3e}"
            f"  gains {excerpt['gain_db'][0]:+.2f} / {excerpt['gain_db'][1]:+.2f} dB"
        )
    print(f"\nsegments retenus : {segments}")
    print(f"ecrit {OUT_DIR / 'seeds.html'}")


if __name__ == "__main__":
    sys.exit(main())
