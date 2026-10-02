#!/usr/bin/env python3
"""Audio of the listening page for the results of 2026-09-23 to 2026-10-02.

Product lane, read-only, CPU. Four sections. Within a setting, the excerpt is chosen
by the real signal's level only; settings follow fixed rules, except the Marshall ones,
which are picked from the recorded per-setting test ESRs (illustrative, not
representative):
- VS100R: vs100r_gc_film_joint_s925_r2 on the unseen test takes of lowest, median and
  highest gain; the 6 s window of highest real RMS.
- Marshall: arm A (param_ssm_wavenet_marshall, FiLM) and arm D (g2_poles_additive_s925,
  poles) at three test settings picked from the recorded per-setting ESRs (largest D
  advantage, smallest gap, worst for both; the mislabelled setting is left out).
- Rate transfer: paired_ssm_wavenet_s925 (ZOH) and paired_ssm_wavenet_foh_s925 (FOH),
  Δ rescaled to 192 kHz and brought back, against the same model at 48 kHz.
- F-Fuzz: param_ssm_wavenet_ffuzz_r2 (before) and _adv4_r2 (after, delayed back by its
  4-sample target advance) at F000, F070 (held out) and F100.
Bench excerpts are 0.5-4.5 s of the test segment of highest real RMS for that setting.
Each group shares one gain, so residuals keep their true level. Writes 16-bit WAV
files to demo/listening/audio/results_*.wav (git-ignored: derived from CC BY-NC 4.0
ToneTwist audio) and demo/listening/results.json.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import statistics
from pathlib import Path

import numpy as np

# Run as a script: its own directory is on sys.path.
import product_nablafx_bench as bench
import product_sr_transfer as transfer
import product_vs100r_eval as ev
import soundfile as sf
import torch
from nablafx.core import BlackBoxModel
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]

AUDIO = ROOT / "demo/listening/audio"
RATE = 48_000
FFUZZ_ADVANCE = 4


def load_param(run_id: str, device: str) -> BlackBoxModel:
    record = json.loads((bench.RESULTS_DIR / f"{run_id}.json").read_text())
    args = argparse.Namespace(model=record["model"], device=device, **record["ssm"])
    processor = bench.build_processor(args)
    path = bench.RUNS_DIR / f"nablafx_{run_id}/checkpoints/last.ckpt"
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    prefix = "model.processor."
    processor.load_state_dict(
        {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    )
    return BlackBoxModel(processor).eval()


def by_setting(run_id: str) -> dict[str, float]:
    record = json.loads((bench.RESULTS_DIR / f"{run_id}.json").read_text())
    table = record["test_last_by_setting"]
    table = ast.literal_eval(table) if isinstance(table, str) else table
    return {k: v["esr"] for k, v in table.items()}


def loudest(dataset, setting: str) -> int:
    """Index of the test segment of `setting` with the highest real RMS."""
    best, index = -1.0, -1
    for i, sample in enumerate(dataset.samples):
        if Path(sample["target_file"]).parent.name != setting:
            continue
        rms = float(dataset[i][1].pow(2).mean())
        if rms > best:
            best, index = rms, i
    return index


@torch.no_grad()
def render(model, x: torch.Tensor, c: torch.Tensor | None) -> np.ndarray:
    model.reset_states()
    return model(x[None], None if c is None else c[None])[0, 0].double().numpy()


def esr(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(((ref - pred) ** 2).sum() / (ref**2).sum())


def level_db(x: np.ndarray, ref: np.ndarray) -> float:
    return 10 * math.log10((x**2).sum() / (ref**2).sum() + 1e-30)


def write_group(name: str, tracks: dict[str, np.ndarray]) -> dict[str, str]:
    """One gain for the whole group: peak 0.89 (-1 dBFS), 10 ms fades."""
    gain = 0.89 / max(np.abs(t).max() for t in tracks.values())
    fade = np.linspace(0, 1, RATE // 100)
    files = {}
    for key, track in tracks.items():
        y = track * gain
        y[: len(fade)] *= fade
        y[-len(fade) :] *= fade[::-1]
        path = AUDIO / f"results_{name}_{key}.wav"
        sf.write(path, y.astype(np.float32), RATE, subtype="PCM_16")
        files[key] = f"audio/{path.name}"
    return files


def excerpt(x: np.ndarray) -> np.ndarray:
    return x[RATE // 2 : RATE * 9 // 2]


def vs100r() -> list[dict]:
    model = ev.load("vs100r_gc_film_joint_s925_r2")
    chosen = [t for t in ev.takes() if t["split"] == "test" and not t["low_signal"]]
    chosen.sort(key=lambda t: (t["gain_contour"][0], t["order"]))
    picks = [chosen[0], chosen[len(chosen) // 2], chosen[-1]]
    out = []
    for take in picks:
        loop, fx = ev.audio(take)
        x48, y48 = resample_poly(loop, 1, 4), resample_poly(fx, 1, 4)
        controls = torch.tensor([v / 10 for v in take["gain_contour"]])
        model.reset_states()
        m48 = ev.run(model, x48, controls)
        window = 6 * RATE
        starts = range(RATE // 2, len(y48) - window - RATE // 10, RATE)
        a = max(starts, key=lambda s: float((y48[s : s + window] ** 2).mean()))
        real, mod = y48[a : a + window], m48[a : a + window]
        g, c = take["gain_contour"]
        name = f"vs100r_take{take['order']}"
        out.append(
            {
                "label": f"Prise {take['order']} : gain {g:g}, contour {c:g}",
                "window_s": [a / RATE, (a + window) / RATE],
                "esr": esr(mod, real),
                "residual_db": level_db(real - mod, real),
                "files": write_group(
                    name, {"real": real, "model": mod, "residual": real - mod}
                ),
            }
        )
        print(name, out[-1]["esr"], flush=True)
    return out


def marshall() -> list[dict]:
    a_id, d_id = "param_ssm_wavenet_marshall", "g2_poles_additive_s925"
    esr_a, esr_d = by_setting(a_id), by_setting(d_id)
    usable = [s for s in esr_a if s != "B030_M070_T100_G050"]
    ratio = {s: math.log(esr_a[s] / esr_d[s]) for s in usable}
    picks = [
        ("Plus grand avantage du bras pôles", max(usable, key=lambda s: ratio[s])),
        ("Plus petit écart entre bras", min(usable, key=lambda s: abs(ratio[s]))),
        ("Réglage le plus difficile", max(usable, key=lambda s: esr_a[s] + esr_d[s])),
    ]
    data = bench.data_module("test", device="marshall")
    data.setup("test")
    dataset = data.test_dataset
    arm_a, arm_d = load_param(a_id, "marshall"), load_param(d_id, "marshall")
    out = []
    for rule, setting in picks:
        i = loudest(dataset, setting)
        x, y, c = dataset[i]
        real = excerpt(y[0].double().numpy())
        ya = excerpt(render(arm_a, x, c))
        yd = excerpt(render(arm_d, x, c))
        b, m, t, g = (int(setting[k + 1 : k + 4]) / 10 for k in (0, 5, 10, 15))
        out.append(
            {
                "label": f"Basses {b:g}, médiums {m:g}, aigus {t:g}, gain {g:g}",
                "rule": rule,
                "setting": setting,
                "recorded_esr": {"film": esr_a[setting], "poles": esr_d[setting]},
                "esr": {"film": esr(ya, real), "poles": esr(yd, real)},
                "files": write_group(
                    f"marshall_{setting}",
                    {
                        "real": real,
                        "film": ya,
                        "poles": yd,
                        "residual_film": real - ya,
                        "residual_poles": real - yd,
                    },
                ),
            }
        )
        print(setting, out[-1]["esr"], flush=True)
    return out


def rate_transfer() -> list[dict]:
    data = bench.data_module("test")
    data.setup("test")
    dataset = data.test_dataset
    rms = [float(dataset[i][1].pow(2).mean()) for i in range(len(dataset))]
    picks = sorted(range(len(dataset)), key=lambda i: -rms[i])[:2]
    x48 = torch.stack([dataset[i][0] for i in picks])
    out = {"segments": picks, "models": {}}
    for kind, run_id in (
        ("zoh", "paired_ssm_wavenet_s925"),
        ("foh", "paired_ssm_wavenet_foh_s925"),
    ):
        model, _ = transfer.load_model(run_id, "last")
        base = {
            id(m): (m.log_dt.detach().clone(), m.C.detach().clone())
            for m in transfer.ssm_modules(model)
        }
        transfer.set_rate(model, RATE, base)
        y48 = transfer.run(model, x48)
        transfer.set_rate(model, 192_000, base)
        y192 = transfer.run(model, transfer.whole(x48, RATE, 192_000))
        back = transfer.whole(y192, 192_000, RATE)
        rows = []
        for k, i in enumerate(picks):
            ref = excerpt(y48[k, 0].double().numpy())
            moved = excerpt(back[k, 0].double().numpy())
            real = excerpt(dataset[i][1][0].double().numpy())
            rows.append(
                {
                    "segment": i,
                    "esr_vs_48k": esr(moved, ref),
                    "residual_db": level_db(ref - moved, ref),
                    "files": write_group(
                        f"rate_{kind}_seg{i}",
                        {
                            "real": real,
                            "model48": ref,
                            "model192": moved,
                            "residual": ref - moved,
                        },
                    ),
                }
            )
            print(kind, i, rows[-1]["esr_vs_48k"], flush=True)
        out["models"][kind] = {"run_id": run_id, "excerpts": rows}
    return out


def ffuzz() -> list[dict]:
    before_id, after_id = (
        "param_ssm_wavenet_ffuzz_r2",
        "param_ssm_wavenet_ffuzz_adv4_r2",
    )
    esr_b, esr_a = by_setting(before_id), by_setting(after_id)
    data = bench.data_module("test", device="ffuzz")
    data.setup("test")
    dataset = data.test_dataset
    before, after = load_param(before_id, "ffuzz"), load_param(after_id, "ffuzz")
    out = []
    for setting in ("F000_V100", "F070_V100", "F100_V100"):
        i = loudest(dataset, setting)
        x, y, c = dataset[i]
        real = excerpt(y[0].double().numpy())
        yb = excerpt(render(before, x, c))
        # The after model learnt the target 4 samples early; delay it back.
        ya_full = render(after, x, c)
        ya = excerpt(np.concatenate([np.zeros(FFUZZ_ADVANCE), ya_full])[: len(ya_full)])
        fuzz = int(setting[1:4]) / 10
        out.append(
            {
                "label": f"Fuzz {fuzz:g}"
                + (" (réglage non vu)" if setting == "F070_V100" else ""),
                "setting": setting,
                "recorded_esr": {"before": esr_b[setting], "after": esr_a[setting]},
                "esr": {"before": esr(yb, real), "after": esr(ya, real)},
                "files": write_group(
                    f"ffuzz_{setting}",
                    {
                        "real": real,
                        "before": yb,
                        "after": ya,
                        "residual_before": real - yb,
                        "residual_after": real - ya,
                    },
                ),
            }
        )
        print(setting, out[-1]["esr"], flush=True)
    return out


def main() -> None:
    torch.set_num_threads(16)
    AUDIO.mkdir(parents=True, exist_ok=True)
    result = {
        "vs100r": vs100r(),
        "marshall": marshall(),
        "rate": rate_transfer(),
        "ffuzz": ffuzz(),
    }
    path = ROOT / "demo/listening/results.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print(statistics.median(r["esr"] for r in result["vs100r"]))


if __name__ == "__main__":
    main()
