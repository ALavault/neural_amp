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
Each group shares one gain, so residuals keep their true level. Writes 24-bit WAV
files to demo/listening/audio/results_*.wav (git-ignored: derived from CC BY-NC 4.0
ToneTwist audio) and demo/listening/results.json.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import statistics
import sys
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
_spec = importlib.util.spec_from_file_location(
    "floor", ROOT / "demo/sr_transfer/floor.py"
)
floor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(floor)
OCTAVES = (63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000)

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


def octave_db(x: np.ndarray) -> np.ndarray:
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / RATE)
    return np.array(
        [
            10 * math.log10(power[(freqs >= c / 2**0.5) & (freqs < c * 2**0.5)].sum())
            for c in OCTAVES
        ]
    )


def write_group(name: str, tracks: dict[str, np.ndarray], ref: str = "real") -> dict:
    """One gain for the whole group: peak 0.89 (-1 dBFS), 10 ms fades, 24-bit WAV.
    Metrics are computed on the files read back, as they are played: ESR of each
    non-residual track against `ref`, per-octave level minus the real track, and each
    residual's level relative to `ref`."""
    gain = 0.89 / max(np.abs(t).max() for t in tracks.values())
    fade = np.linspace(0, 1, RATE // 100)
    files, played = {}, {}
    for key, track in tracks.items():
        y = track * gain
        y[: len(fade)] *= fade
        y[-len(fade) :] *= fade[::-1]
        path = AUDIO / f"results_{name}_{key}.wav"
        sf.write(path, y, RATE, subtype="PCM_24")
        files[key] = f"audio/{path.name}"
        played[key] = sf.read(path, dtype="float64")[0]
    signals = [k for k in tracks if not k.startswith("residual")]
    real = octave_db(played["real"])
    return {
        "files": files,
        "ref": ref,
        "esr": {
            k: esr(played[k], played[ref]) for k in signals if k not in (ref, "real")
        },
        "octaves": {
            k: (octave_db(played[k]) - real).round(2).tolist()
            for k in signals
            if k != "real"
        },
        "residual_db": {
            k: 10 * math.log10((played[k] ** 2).sum() / (played[ref] ** 2).sum())
            for k in tracks
            if k.startswith("residual")
        },
    }


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
                **write_group(
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
                **write_group(
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


def rate_transfer() -> dict:
    """Kaiser-64 resampling both ways (floor.hq), as in brightness.py. Tracks: the
    model at 48 kHz, at 192 kHz brought back, the 48 kHz output through the resampler
    alone (control), 48 - 192, and for ZOH 48 - 192 advanced by the checkpoint's
    median fractional delay at 192 kHz (kappa.json)."""
    data = bench.data_module("test")
    data.setup("test")
    dataset = data.test_dataset
    rms = [float(dataset[i][1].pow(2).mean()) for i in range(len(dataset))]
    picks = sorted(range(len(dataset)), key=lambda i: -rms[i])[:2]
    x48 = torch.stack([dataset[i][0] for i in picks])
    kappa = json.loads((ROOT / "demo/sr_transfer/kappa.json").read_text())["bigmuff"]
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
        control = floor.hq(floor.hq(y48, RATE, 192_000), 192_000, RATE)
        transfer.set_rate(model, 192_000, base)
        y192 = transfer.run(model, floor.hq(x48, RATE, 192_000))
        back = floor.hq(y192, 192_000, RATE)
        rows = []
        for k, i in enumerate(picks):
            ref = excerpt(y48[k, 0].double().numpy())
            moved = excerpt(back[k, 0].double().numpy())
            tracks = {
                "real": excerpt(dataset[i][1][0].double().numpy()),
                "model48": ref,
                "model192": moved,
                "control": excerpt(control[k, 0].double().numpy()),
                "residual": ref - moved,
            }
            row = {"segment": i}
            if kind == "zoh":
                delay = kappa[run_id]["192000"]["delay_median"]
                full = back[k, 0].double().numpy()
                spectrum = np.fft.rfft(full)
                phase = np.exp(2j * np.pi * np.fft.rfftfreq(len(full)) * delay)
                tracks["residual_compensated"] = ref - excerpt(
                    np.fft.irfft(spectrum * phase, len(full))
                )
                row["delay_samples"] = delay
            rows.append(
                {**row, **write_group(f"rate_{kind}_seg{i}", tracks, "model48")}
            )
            print(kind, i, rows[-1]["esr"], flush=True)
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
                **write_group(
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


RATE_PAIRS = [
    (42, "vs100r_gc_film_joint_s925_r2", "vs100r192_film_s42"),
    (43, "vs100r48_film_s43", "vs100r192_film_s43"),
    (44, "vs100r48_film_s44", "vs100r192_film_s44"),
]


@torch.no_grad()
def vs100r_rates() -> list[dict]:
    """predictions_192k.md, seeds 42, 43 and 44: the model trained at 48 kHz against the
    one trained at 192 kHz, whose output is decimated by resample_poly(., 1, 4), the
    decimation that made the 48 kHz real reference. Same takes and window rule as
    vs100r(). Both models are level-matched to the real excerpt (RMS) and their
    original offsets recorded. Listening only; the readings are in eval_192k.json."""
    chosen = [t for t in ev.takes() if t["split"] == "test" and not t["low_signal"]]
    chosen.sort(key=lambda t: (t["gain_contour"][0], t["order"]))
    picks = [chosen[0], chosen[len(chosen) // 2], chosen[-1]]
    audio = {t["order"]: ev.audio(t) for t in picks}
    out = []
    for seed, run48, run192 in RATE_PAIRS:
        model48, model192 = ev.load(run48), ev.load(run192)
        for take in picks:
            loop, fx = audio[take["order"]]
            controls = torch.tensor([v / 10 for v in take["gain_contour"]])
            y48 = resample_poly(fx, 1, 4)
            m48 = ev.run(model48, resample_poly(loop, 1, 4), controls)
            m192 = resample_poly(ev.run(model192, loop, controls), 1, 4)
            window = 6 * RATE
            starts = range(RATE // 2, len(y48) - window - RATE // 10, RATE)
            a = max(starts, key=lambda s: float((y48[s : s + window] ** 2).mean()))
            cut = slice(a, a + window)
            real, a48, b192 = y48[cut], m48[cut], m192[cut]
            # Level-matched to the real excerpt (RMS), so a louder model is not
            # heard as a different one; the original offsets are kept for display.
            rms = lambda v: float(np.sqrt(np.mean(v**2)))  # noqa: E731
            offsets = {
                "model48": 20 * math.log10(rms(a48) / rms(real)),
                "model192": 20 * math.log10(rms(b192) / rms(real)),
            }
            a48 = a48 * rms(real) / rms(a48)
            b192 = b192 * rms(real) / rms(b192)
            g, c = take["gain_contour"]
            out.append(
                {
                    "seed": seed,
                    "run48": run48,
                    "run192": run192,
                    "level_offset_db": offsets,
                    "label": f"Graine {seed}, prise {take['order']} : gain {g:g}, "
                    f"contour {c:g}",
                    "window_s": [a / RATE, (a + window) / RATE],
                    **write_group(
                        f"vs100r_rates_s{seed}_take{take['order']}",
                        {
                            "real": real,
                            "model48": a48,
                            "model192": b192,
                            "residual_48": real - a48,
                            "residual_192": real - b192,
                        },
                    ),
                }
            )
            print("vs100r_rates", seed, take["order"], out[-1]["esr"], flush=True)
    return out


@torch.no_grad()
def vs100r_gain25() -> list[dict]:
    """Blind ABX at gain 2.5 (diagnosis/hammerstein/NOTE.md): the 48 kHz model as is
    against its odd part (f(x) - f(-x)) / 2, which has no even harmonics, with the real
    amp as reference. Every non-low-signal take at gain 2.5, di_test (audio never
    trained on), 6 s window of highest real RMS as in vs100r()."""
    model = ev.load("vs100r_gc_film_joint_s925_r2")
    takes = [
        t for t in ev.takes() if t["gain_contour"][0] == 2.5 and not t["low_signal"]
    ]
    out = []
    for take in sorted(takes, key=lambda t: t["order"]):
        loop, fx = ev.audio(take)
        controls = torch.tensor([v / 10 for v in take["gain_contour"]])
        y48 = resample_poly(fx, 1, 4)
        x48 = resample_poly(loop, 1, 4)
        plus, minus = ev.run(model, x48, controls), ev.run(model, -x48, controls)
        odd = (plus - minus) / 2
        window = 6 * RATE
        starts = range(RATE // 2, len(y48) - window - RATE // 10, RATE)
        a = max(starts, key=lambda s: float((y48[s : s + window] ** 2).mean()))
        cut = slice(a, a + window)
        g, c = take["gain_contour"]
        out.append(
            {
                "label": f"Prise {take['order']} : gain {g:g}, contour {c:g}",
                "window_s": [a / RATE, (a + window) / RATE],
                **write_group(
                    f"vs100r_g25_take{take['order']}",
                    {
                        "real": y48[cut],
                        "model": plus[cut],
                        "odd": odd[cut],
                        "residual_even": plus[cut] - odd[cut],
                    },
                ),
            }
        )
        print("vs100r_gain25", take["order"], out[-1]["esr"], flush=True)
    return out


def main() -> None:
    torch.set_num_threads(16)
    AUDIO.mkdir(parents=True, exist_ok=True)
    path = ROOT / "demo/listening/results.json"
    if "--only-gain25" in sys.argv:
        result = json.loads(path.read_text())
        result["vs100r_gain25"] = vs100r_gain25()
        path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
        return
    if "--only-rates" in sys.argv:
        # Recompute the 48 against 192 kHz section only, keep the others as written.
        result = json.loads(path.read_text())
        result["vs100r_rates"] = vs100r_rates()
        path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
        return
    result = {
        "vs100r": vs100r(),
        "vs100r_rates": vs100r_rates(),
        "marshall": marshall(),
        "rate": rate_transfer(),
        "ffuzz": ffuzz(),
    }
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print(statistics.median(r["esr"]["model"] for r in result["vs100r"]))


if __name__ == "__main__":
    main()
