#!/usr/bin/env python3
"""Post hoc descriptive checks of two listening remarks (2026-10-02). Read-only, CPU.

1. VS100R take 16 sounded louder than the real amp: model - real RMS level (dB) and
   per-octave levels on the 8 unseen test takes (same scored region as the eval).
2. Marshall B0 M10 T10: "one arm eats the body, the other stays on the attacks".
   Per arm, model - real level per octave, separately in attack windows (0-30 ms after
   onsets of the real envelope) and sustain windows (50-300 ms, cut at the next onset),
   on the excerpt listened to and on every test segment of that setting; and the
   residual energy relative to the real one in each window class.
No prediction was written: these numbers describe, they do not test.
Writes demo/listening/listening_checks.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import product_nablafx_bench as bench
import product_results_listening as page
import product_vs100r_eval as ev
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
RATE = 48_000
OCTAVES = (125, 250, 500, 1000, 2000, 4000, 8000, 16000)
FRAME, HOP = 512, 128


def db(x: float) -> float:
    return 10 * math.log10(x + 1e-30)


def octave_energy(x: np.ndarray) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / RATE)
    return np.array(
        [spectrum[(freqs >= c / 2**0.5) & (freqs < c * 2**0.5)].sum() for c in OCTAVES]
    )


@torch.no_grad()
def vs100r() -> list[dict]:
    model = ev.load("vs100r_gc_film_joint_s925_r2")
    rows = []
    for take in ev.takes():
        if take["split"] != "test" or take["low_signal"]:
            continue
        loop, fx = ev.audio(take)
        x48, y48 = resample_poly(loop, 1, 4), resample_poly(fx, 1, 4)
        controls = torch.tensor([v / 10 for v in take["gain_contour"]])
        m, r = ev.scored(ev.run(model, x48, controls)), ev.scored(y48)
        em, er = octave_energy(m), octave_energy(r)
        rows.append(
            {
                "order": take["order"],
                "gain_contour": take["gain_contour"],
                "pair_takes_in_training": take["takes_same_pair_in_training"],
                "level_db": db((m**2).sum()) - db((r**2).sum()),
                "octave_db": {
                    str(c): db(a) - db(b)
                    for c, a, b in zip(OCTAVES, em, er, strict=True)
                },
            }
        )
        print(take["order"], round(rows[-1]["level_db"], 2), flush=True)
    return rows


def frames(x: np.ndarray) -> np.ndarray:
    count = 1 + (len(x) - FRAME) // HOP
    index = np.arange(FRAME)[None] + HOP * np.arange(count)[:, None]
    return x[index] * np.hanning(FRAME)


def classes(real: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frame masks (attack, sustain) from onsets of the real envelope."""
    env = 10 * np.log10((frames(real) ** 2).mean(1) + 1e-12)
    times = (np.arange(len(env)) * HOP + FRAME / 2) / RATE
    floor = env.max() - 40
    look = int(0.05 * RATE / HOP)
    onsets = []
    for i in range(look, len(env)):
        if env[i] > floor and env[i] - env[i - look : i].min() > 9:
            if not onsets or times[i] - onsets[-1] > 0.1:
                onsets.append(times[i])
    attack = np.zeros(len(env), bool)
    sustain = np.zeros(len(env), bool)
    for k, t in enumerate(onsets):
        end = onsets[k + 1] if k + 1 < len(onsets) else times[-1] + 1
        attack |= (times >= t) & (times < t + 0.03)
        sustain |= (times >= t + 0.05) & (times < min(t + 0.3, end))
    return attack, sustain


def band_energy(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    spectra = np.abs(np.fft.rfft(frames(x)[mask], axis=1)) ** 2
    freqs = np.fft.rfftfreq(FRAME, 1 / RATE)
    return np.array(
        [
            spectra[:, (freqs >= c / 2**0.5) & (freqs < c * 2**0.5)].sum()
            for c in OCTAVES
        ]
    )


def windows(real: np.ndarray, arms: dict[str, np.ndarray]) -> dict:
    attack, sustain = classes(real)
    out = {"attack_frames": int(attack.sum()), "sustain_frames": int(sustain.sum())}
    for name, mask in (("attack", attack), ("sustain", sustain)):
        er = band_energy(real, mask)
        for arm, y in arms.items():
            ey = band_energy(y, mask)
            res = band_energy(real - y, mask)
            out[f"{arm}_{name}"] = {
                "octave_db": {
                    str(c): db(a) - db(b)
                    for c, a, b in zip(OCTAVES, ey, er, strict=True)
                },
                "residual_db": db(res.sum()) - db(er.sum()),
            }
    return out


@torch.no_grad()
def marshall(setting: str = "B000_M100_T100_G050") -> dict:
    data = bench.data_module("test", device="marshall")
    data.setup("test")
    dataset = data.test_dataset
    arm_a = page.load_param("param_ssm_wavenet_marshall", "marshall")
    arm_d = page.load_param("g2_poles_additive_s925", "marshall")
    listened = page.loudest(dataset, setting)
    reals, films, poles = [], [], []
    out = {}
    for i, sample in enumerate(dataset.samples):
        if Path(sample["target_file"]).parent.name != setting:
            continue
        x, y, c = dataset[i]
        real = y[0].double().numpy()
        film, pole = page.render(arm_a, x, c), page.render(arm_d, x, c)
        if i == listened:
            out["listened"] = windows(
                page.excerpt(real),
                {"film": page.excerpt(film), "poles": page.excerpt(pole)},
            )
        cut = slice(RATE // 10, -RATE // 10)
        reals.append(real[cut])
        films.append(film[cut])
        poles.append(pole[cut])
    out["all_segments"] = windows(
        np.concatenate(reals),
        {"film": np.concatenate(films), "poles": np.concatenate(poles)},
    )
    out["segments"] = len(reals)
    return out


def main() -> None:
    torch.set_num_threads(16)
    result = {"vs100r": vs100r(), "marshall_B000_M100_T100": marshall()}
    path = ROOT / "demo/listening/listening_checks.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    m = result["marshall_B000_M100_T100"]
    for scope in ("listened", "all_segments"):
        for arm in ("film", "poles"):
            print(
                scope,
                arm,
                {
                    k: round(m[scope][f"{arm}_{k}"]["residual_db"], 1)
                    for k in ("attack", "sustain")
                },
            )


if __name__ == "__main__":
    main()
