#!/usr/bin/env python3
"""Blind A/B/X page for the butterfly children: is a divergence of ESR audible?

Nobody has ever listened to these models. Fork 100 separates the control (decide k = 0,
test ESR 0.034) from its most divergent child (decide k = 1, 0.051) by 41 % of relative
error, and from a replay child (replay k = 1, 0.034) by nothing measurable. Three pairs
per excerpt:

- real device against the control, which says whether anything at all is audible here;
- control against decide k = 1, the question;
- control against replay k = 1, the null: their outputs differ by an ESR of 5e-4.

Excerpts sit strictly inside one test segment (0.3-4.7 s of a 5 s segment), because the
protocol resets the model state at the start of every segment: a window crossing a
boundary would carry that transient. Each model side gets its own least-squares gain
against the target, so a level difference cannot give the answer away; the page reports
the correction applied. Audio goes to demo/listening/audio/ (git-ignored), the page to
demo/listening/butterfly.html.

Second version. The first one took segments 2, 6 and 10, chosen by position alone, and
the listener heard no difference anywhere. Measurement afterwards showed those were among
the least divergent segments of the twelve: the model-to-model error reaches 0.93 of the
model-to-device error on segment 0 and falls to 0.17 on segment 10. The excerpts are now
chosen ON that divergence, which makes this a best-case probe rather than a blind sample -
if the divergence is inaudible here, it is inaudible anywhere on this device. Levels are
normalised per excerpt, and the difference signal is offered on its own, outside the test,
because on this material it sits only 16 to 21 dB under the signal in the same half-octaves
and masking is the likely reason nothing was heard.
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

from product_fork_pilot_analysis import outputs  # noqa: E402

import product_nablafx_bench as bench  # noqa: E402

OUT_DIR = ROOT / "demo/listening"
AUDIO_DIR = OUT_DIR / "audio"
SAMPLE_RATE = 48_000
FADE_SECONDS = 0.020
# The three most divergent test segments (model-to-model ESR 0.93, 0.65 and 0.43 of the
# model-to-device error), and a window inside one segment.
SEGMENTS = (0, 6, 1)
WINDOW = (0.3, 4.7)
PEAK = 10 ** (-1 / 20)
STAMP = int(time.time())
MODELS = {
    "temoin": "butterfly_ssm_seed42_f100_decide_k0",
    "decide_k1": "butterfly_ssm_seed42_f100_decide_k1",
    "rejoue_k1": "butterfly_ssm_seed42_f100_replay_k1",
}
PAIRS = (
    ("reel", "temoin", "appareil réel contre témoin : quelque chose est-il audible ?"),
    ("temoin", "decide_k1", "témoin contre enfant divergent : la question"),
    ("temoin", "rejoue_k1", "témoin contre enfant « rejoue » : le témoin négatif"),
)


def fade(signal: np.ndarray) -> np.ndarray:
    length = int(FADE_SECONDS * SAMPLE_RATE)
    ramp = np.linspace(0.0, 1.0, length, dtype=np.float64)
    faded = signal.astype(np.float64).copy()
    faded[:length] *= ramp
    faded[-length:] *= ramp[::-1]
    return faded


def main() -> None:
    torch.set_num_threads(8)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    data = bench.data_module("test")
    data.setup("test")
    inputs = torch.stack([x for x, _ in data.test_dataset])
    targets = torch.stack([y for _, y in data.test_dataset])
    rendered = {name: outputs(run, inputs).numpy()[:, 0] for name, run in MODELS.items()}
    rendered["reel"] = targets.numpy()[:, 0]

    span = slice(int(WINDOW[0] * SAMPLE_RATE), int(WINDOW[1] * SAMPLE_RATE))
    excerpts = []
    for index, segment in enumerate(SEGMENTS, start=1):
        reference = rendered["reel"][segment][span].astype(np.float64)
        clips, gains = {}, {}
        for name, signal in rendered.items():
            clip = signal[segment][span].astype(np.float64)
            # One least-squares scalar per model and per excerpt, against the target.
            gain = 1.0 if name == "reel" else float(clip @ reference / (clip @ clip))
            clips[name] = fade(clip * gain)
            gains[name] = float(20.0 * np.log10(gain))
        # One normalisation per excerpt, the same for every side, so the comparison is
        # untouched and the material is not played 25 dB below full scale.
        loud = PEAK / max(float(np.max(np.abs(c))) for c in clips.values())
        clips = {name: clip * loud for name, clip in clips.items()}
        # The difference, on its own and outside the test: what separates the two models.
        difference = clips["temoin"] - clips["decide_k1"]
        clips["difference"] = difference * (PEAK / float(np.max(np.abs(difference))))
        files = {}
        for name, clip in clips.items():
            files[name] = f"audio/butterfly_{index}_{name}.wav"
            sf.write(OUT_DIR / files[name], clip, SAMPLE_RATE, subtype="PCM_16")
        # The names do not change between renders, so the page asks for this one.
        files = {name: f"{path}?{STAMP}" for name, path in files.items()}
        for left, right, label in PAIRS:
            written = {
                side: sf.read(OUT_DIR / files[side].split("?")[0], dtype="float64")[0]
                for side in (left, right)
            }
            gap = written[left] - written[right]
            excerpts.append(
                {
                    "key": f"{index}_{left}_vs_{right}",
                    "index": index,
                    "segment": segment,
                    "window": f"segment {segment}, {WINDOW[0]:g}-{WINDOW[1]:g} s",
                    "label": label,
                    "left": files[left],
                    "right": files[right],
                    "names": [left, right],
                    "esr_between_sides": float(
                        gap @ gap / (written[right] @ written[right])
                    ),
                    "gain_db": [gains[left], gains[right]],
                    "difference": files["difference"] if right == "decide_k1" else None,
                }
            )

    OUT_DIR.joinpath("butterfly.html").write_text(
        TEMPLATE.replace("__EXCERPTS__", json.dumps(excerpts, ensure_ascii=False)),
        encoding="utf-8",
    )
    for excerpt in excerpts:
        print(
            f"{excerpt['key']}: ESR entre les deux côtés {excerpt['esr_between_sides']:.2e}"
            f" gains {excerpt['gain_db'][0]:+.2f} / {excerpt['gain_db'][1]:+.2f} dB"
        )
    print(f"wrote {OUT_DIR / 'butterfly.html'} and {len(SEGMENTS) * 5} clips")


TEMPLATE = """<!doctype html>
<html lang="fr">
<meta charset="utf-8">
<title>Écoute aveugle A/B/X : les enfants du papillon</title>
<style>
 body { font: 15px/1.5 system-ui, sans-serif; max-width: 46rem; margin: 2rem auto;
        padding: 0 1rem; }
 h1 { font-size: 1.4rem; }
 .excerpt { border: 1px solid #ccc; border-radius: 6px; padding: 1rem;
            margin: 1rem 0; }
 button { font: inherit; padding: .4rem .9rem; margin-right: .4rem; }
 .play { min-width: 3rem; }
 .on { background: #222; color: #fff; }
 .score { font-variant-numeric: tabular-nums; }
 .meta { color: #555; font-size: .9rem; }
 textarea { width: 100%; height: 10rem; font: 12px monospace; }
</style>
<h1>Écoute aveugle A/B/X : les enfants du papillon</h1>
<p>Trois paires par extrait. <b>Appareil réel contre témoin</b> dit si quoi que ce soit
est audible ici ; <b>témoin contre décide k = 1</b> est la question, ces deux modèles
diffèrent de 41 % d'ESR relatif ; <b>témoin contre rejoue k = 1</b> est le témoin
négatif, leurs sorties diffèrent d'un ESR de 5e-4.</p>
<p>Si le réel contre le témoin est au hasard, l'épreuve ne mesure rien et l'écart d'ESR
est hors de portée de cette écoute. Si le réel est trivial et que le témoin contre
décide k = 1 reste au hasard, alors l'écart d'ESR de 41 % est inaudible sur ce matériel.
Dix essais par bloc : 20 justes sur 30 donnent p &lt; 0,05 pour un extrait.</p>
<p class="meta"><b>Deuxième version.</b> La première prenait les segments 2, 6 et 10,
choisis par position seule, et rien n'y était audible. Mesure faite après coup : c'étaient
les segments les moins divergents des douze. L'écart entre modèles y vaut 0,17 à 0,65 fois
l'erreur au réel, contre 0,93 sur le segment 0. Les extraits sont donc maintenant choisis
<b>sur</b> cette divergence — c'est une épreuve au meilleur endroit possible, plus un
échantillon aveugle : si rien n'est audible ici, rien ne l'est ailleurs sur cet appareil.</p>
<p class="meta">Extraits pris dans un seul segment de test (0,3-4,7 s des 5 s). Chaque
modèle est corrigé d'un gain des moindres carrés contre la cible, propre à l'extrait, pour
qu'une différence de niveau ne donne pas la réponse ; l'épreuve porte donc sur ce qui reste
au-delà du niveau. Une seule normalisation par extrait, identique pour tous les côtés,
amène le crête à -1 dBFS. Le tirage est dans cette page : elle sert à écouter honnêtement,
pas à résister à quelqu'un qui ouvre les outils de développement. Au casque.</p>
<p class="meta">Le bouton <b>Différence</b> n'appartient pas à l'épreuve : il joue le signal
témoin moins décide k1, normalisé, pour entendre <i>ce qui</i> sépare les deux modèles. Sur
ce matériel cet écart est à 16 à 21 dB sous le signal dans les mêmes demi-octaves, ce qui
est la configuration de masquage la plus défavorable : une fuzz à fond masque sa propre
erreur.</p>
<div id="excerpts"></div>
<h2>Résultats</h2>
<textarea id="results" readonly></textarea>

<script>
const EXCERPTS = __EXCERPTS__;

function randomBit() {
  const buffer = new Uint32Array(1);
  crypto.getRandomValues(buffer);
  return buffer[0] % 2;
}

function pValue(correct, trials) {
  let logFactorial = [0];
  for (let i = 1; i <= trials; i++) logFactorial[i] = logFactorial[i - 1] + Math.log(i);
  let total = 0;
  for (let k = correct; k <= trials; k++) {
    total += Math.exp(logFactorial[trials] - logFactorial[k] - logFactorial[trials - k]
                      - trials * Math.log(2));
  }
  return total;
}

const state = JSON.parse(localStorage.getItem('abx-butterfly') || '{}');
const container = document.getElementById('excerpts');
const results = document.getElementById('results');

function save() {
  localStorage.setItem('abx-butterfly', JSON.stringify(state));
  results.value = JSON.stringify(EXCERPTS.map(e => {
    const s = state[e.key] || { trials: 0, correct: 0 };
    return { bloc: e.key, paire: e.names.join(' / '), essais: s.trials,
             justes: s.correct,
             p: s.trials ? Number(pValue(s.correct, s.trials).toFixed(4)) : null };
  }), null, 1);
}

for (const excerpt of EXCERPTS) {
  if (!state[excerpt.key]) state[excerpt.key] = { trials: 0, correct: 0 };
  // A/B assignment is redrawn each time the page loads, X at each trial.
  const aIsLeft = randomBit() === 0;
  const sources = {
    A: aIsLeft ? excerpt.left : excerpt.right,
    B: aIsLeft ? excerpt.right : excerpt.left,
  };
  let xIsA = randomBit() === 0;

  const node = document.createElement('div');
  node.className = 'excerpt';
  node.innerHTML = `<b>${excerpt.label}</b><br>
    <span class="meta">extrait ${excerpt.index} (${excerpt.window})</span><br>
    <button class="play" data-k="A">A</button>
    <button class="play" data-k="B">B</button>
    <button class="play" data-k="X">X</button>
    <button data-guess="A">X = A</button>
    <button data-guess="B">X = B</button>
    <span class="score"></span>
    <button data-reveal="1" hidden>Révéler</button>
    ${excerpt.difference ? '<button data-difference="1">Différence</button>' : ''}`;
  container.appendChild(node);

  const gap = node.querySelector('[data-difference]');
  if (gap) {
    const player = new Audio(excerpt.difference);
    gap.onclick = () => { player.currentTime = 0; player.play(); };
  }

  const audio = { A: new Audio(sources.A), B: new Audio(sources.B), X: new Audio() };
  audio.X.src = xIsA ? sources.A : sources.B;
  const score = node.querySelector('.score');
  const reveal = node.querySelector('[data-reveal]');

  function refresh() {
    const s = state[excerpt.key];
    score.textContent = ` ${s.correct}/${s.trials}`
      + (s.trials ? ` (p = ${pValue(s.correct, s.trials).toFixed(3)})` : '');
    reveal.hidden = s.trials < 10;
    save();
  }

  node.querySelectorAll('.play').forEach(button => {
    button.onclick = () => {
      // Keep the playhead when switching, so the comparison is instantaneous.
      let position = 0;
      for (const player of Object.values(audio)) {
        if (!player.paused) { position = player.currentTime; player.pause(); }
      }
      node.querySelectorAll('.play').forEach(other => other.classList.remove('on'));
      button.classList.add('on');
      const player = audio[button.dataset.k];
      player.currentTime = position;
      player.play();
    };
  });

  node.querySelectorAll('[data-guess]').forEach(button => {
    button.onclick = () => {
      const s = state[excerpt.key];
      s.trials += 1;
      if ((button.dataset.guess === 'A') === xIsA) s.correct += 1;
      xIsA = randomBit() === 0;
      audio.X.src = xIsA ? sources.A : sources.B;
      refresh();
    };
  });

  reveal.onclick = () => {
    const [left, right] = excerpt.names;
    reveal.textContent = `A = ${aIsLeft ? left : right}, B = ${aIsLeft ? right : left}`
      + ` — ESR entre les deux côtés ${excerpt.esr_between_sides.toExponential(2)}`;
    reveal.disabled = true;
  };

  refresh();
}
</script>
</html>
"""


if __name__ == "__main__":
    main()
