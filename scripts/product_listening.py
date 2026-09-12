#!/usr/bin/env python3
"""Build the local blind A/B/X listening page from the native renders.

The model side is rendered by the native engine, which is bit-exact with the
plugin, so what the page plays is what the demo plays. Page and audio stay
local: the source captures are CC-BY-NC.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

from product_report import MANIFEST, ROOT, RUNS, build_tools, run_native

from fssr_nam.metrics.time import gain_error, time_metrics

OUT_DIR = ROOT / "demo/listening"
AUDIO_DIR = OUT_DIR / "audio"
SAMPLE_RATE = 48_000
FADE_SECONDS = 0.020
# Three fixed windows covering the 30 s test file, chosen by position only.
WINDOWS = ((1.0, 9.0), (10.5, 18.5), (20.0, 28.0))


def fade(signal: np.ndarray) -> np.ndarray:
    """Fade both edges so the learnt DC offset does not click."""
    length = int(FADE_SECONDS * SAMPLE_RATE)
    ramp = np.linspace(0.0, 1.0, length, dtype=np.float64)
    faded = signal.astype(np.float64).copy()
    faded[:length] *= ramp
    faded[-length:] *= ramp[::-1]
    return faded


def main() -> None:
    _, runner = build_tools()
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    from fssr_nam.product.data import device_pairs

    excerpts = []
    for device, run_id in RUNS.items():
        test_input, test_target = device_pairs(MANIFEST, device, root=ROOT)["test"]
        x, rate = sf.read(test_input, dtype="float32")
        target, _ = sf.read(test_target, dtype="float32")
        if rate != SAMPLE_RATE:
            raise RuntimeError(f"{device} test pair is at {rate} Hz")
        model_path = ROOT / "demo/runs" / run_id / "model_full.nam"
        prediction, _ = run_native(runner, model_path, x, "64")

        # One least-squares scalar per device on the model side, so a level cue
        # cannot give the answer away. Identical for every excerpt.
        correction = 1.0 / (1.0 + gain_error(prediction, target))
        prediction = prediction * correction

        for index, (start, stop) in enumerate(WINDOWS, start=1):
            span = slice(int(start * SAMPLE_RATE), int(stop * SAMPLE_RATE))
            pair = {"real": fade(target[span]), "model": fade(prediction[span])}
            peak = max(float(np.max(np.abs(signal))) for signal in pair.values())
            if peak >= 1.0:
                raise RuntimeError(f"{device} excerpt {index} peaks at {peak:.3f}")
            names = {}
            for side, signal in pair.items():
                name = f"{device}_{index}_{side}.wav"
                sf.write(AUDIO_DIR / name, signal, SAMPLE_RATE, subtype="PCM_24")
                names[side] = f"audio/{name}"
            # Score the files as written, so the page shows what it plays.
            written = {
                side: sf.read(AUDIO_DIR / Path(name).name, dtype="float64")[0]
                for side, name in names.items()
            }
            metrics = time_metrics(written["model"], written["real"])
            excerpts.append(
                {
                    "device": device,
                    "index": index,
                    "window": f"{start:g}-{stop:g} s",
                    "real": names["real"],
                    "model": names["model"],
                    "esr": metrics["esr"],
                    "gain_correction_db": float(20.0 * np.log10(correction)),
                }
            )

    OUT_DIR.joinpath("index.html").write_text(page(excerpts), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'index.html'} and {len(excerpts) * 2} clips")


def page(excerpts: list[dict]) -> str:
    return TEMPLATE.replace("__EXCERPTS__", json.dumps(excerpts, ensure_ascii=False))


TEMPLATE = """<!doctype html>
<html lang="fr">
<meta charset="utf-8">
<title>Écoute aveugle A/B/X</title>
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
 textarea { width: 100%; height: 8rem; font: 12px monospace; }
</style>
<h1>Écoute aveugle A/B/X : appareil réel contre modèle</h1>
<p>Trois extraits par appareil, pris à des positions fixes du fichier de test
(1-9 s, 10,5-18,5 s, 20-28 s) : aucune sélection sur le résultat. A et B sont
tirés au sort à chaque chargement, X est tiré à chaque essai. Le modèle est
rendu par le moteur natif du plugin, et corrigé d'un seul gain par appareil
pour qu'une différence de niveau ne donne pas la réponse.</p>
<p class="meta">L'ESR affiché est celui de l'extrait, mesuré sur les fichiers tels
qu'ils sont joués, donc après correction de gain : il ne coïncide pas avec
l'ESR global de <code>REPORT.md</code>, qui porte sur les 30 s non corrigées.</p>
<p class="meta">Le tirage est dans cette page : elle sert à écouter honnêtement,
pas à résister à quelqu'un qui ouvre les outils de développement. Au casque,
au moins 10 essais par extrait.</p>
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

// One-sided binomial tail: probability of at least `correct` out of `trials`
// by guessing.
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

const state = JSON.parse(localStorage.getItem('abx') || '{}');
const container = document.getElementById('excerpts');
const results = document.getElementById('results');

function save() {
  localStorage.setItem('abx', JSON.stringify(state));
  results.value = JSON.stringify(EXCERPTS.map(e => {
    const key = e.device + '_' + e.index;
    const s = state[key] || { trials: 0, correct: 0 };
    return { extrait: key, fenetre: e.window, essais: s.trials, justes: s.correct,
             p: s.trials ? Number(pValue(s.correct, s.trials).toFixed(4)) : null,
             esr: Number(e.esr.toFixed(5)) };
  }), null, 1);
}

for (const excerpt of EXCERPTS) {
  const key = excerpt.device + '_' + excerpt.index;
  if (!state[key]) state[key] = { trials: 0, correct: 0 };
  // A/B assignment is redrawn each time the page loads.
  const aIsReal = randomBit() === 0;
  const sources = {
    A: aIsReal ? excerpt.real : excerpt.model,
    B: aIsReal ? excerpt.model : excerpt.real,
  };
  let xIsA = randomBit() === 0;

  const node = document.createElement('div');
  node.className = 'excerpt';
  node.innerHTML = `<b>${excerpt.device}</b> — extrait ${excerpt.index}
    (${excerpt.window}) <span class="meta">ESR ${excerpt.esr.toFixed(5)}</span><br>
    <button class="play" data-k="A">A</button>
    <button class="play" data-k="B">B</button>
    <button class="play" data-k="X">X</button>
    <button data-guess="A">X = A</button>
    <button data-guess="B">X = B</button>
    <span class="score"></span>
    <button data-reveal="1" hidden>Révéler</button>`;
  container.appendChild(node);

  const audio = { A: new Audio(sources.A), B: new Audio(sources.B), X: new Audio() };
  audio.X.src = xIsA ? sources.A : sources.B;
  const score = node.querySelector('.score');
  const reveal = node.querySelector('[data-reveal]');

  function refresh() {
    const s = state[key];
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
      const s = state[key];
      s.trials += 1;
      if ((button.dataset.guess === 'A') === xIsA) s.correct += 1;
      xIsA = randomBit() === 0;
      audio.X.src = xIsA ? sources.A : sources.B;
      refresh();
    };
  });

  reveal.onclick = () => {
    reveal.textContent = `A = ${aIsReal ? 'réel' : 'modèle'}, `
      + `B = ${aIsReal ? 'modèle' : 'réel'}`;
    reveal.disabled = true;
  };

  refresh();
}
</script>
</html>
"""


if __name__ == "__main__":
    main()
