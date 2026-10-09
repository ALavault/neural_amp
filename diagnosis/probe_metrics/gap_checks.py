#!/usr/bin/env python3
"""Post hoc checks of two real gaps, before any ad hoc metric (NOTE.md).

jvm: why does Hammerstein 5 leave ~0.75 of the JVM410H differences unrepresented?
  - capacity on the reference itself: r_ref / RMS(y_ref) of H5 (256 taps) at the centre;
  - on three ladder pairs: H5 with 1024 taps (memory), Volterra-Laguerre 3 (frozen v2;
    interactions), and the coherence between x and the unrepresented difference
    (near 0: noise-like or not linearly tied to x).
sweep: why does the harmonic signature stop separating at -3 dBFS on the VS100R?
  orders 1-9 on one repeat pair and one gain pair, at -10 and -3 dBFS.
Writes gap_checks_<which>.json. CPU only. Post hoc: diagnosis, not a protocol change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import coherence

from fssr_nam.product.fidelity import (
    HarmonicProtocol,
    Protocol,
    analyze_pair,
    analyze_sweep_pair,
)
from fssr_nam.product.probes import Probe, fit, render

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibrate import frozen_probes  # noqa: E402
from real_pairs import SWEEPS, common_section, jvm_audio, lag  # noqa: E402

PAIRS = ("B050_M050_T050_G040", "B080_M050_T050_G050", "B050_M050_T020_G050")


def capacity(probe: Probe, rate: int, x: dict, y: dict) -> float:
    q = render(fit(probe, x["A"], y["A"]), x["B"])
    skip = round(0.05 * rate)
    return float(
        np.sqrt(np.mean((y["B"] - q)[skip:] ** 2) / np.mean(y["B"][skip:] ** 2))
    )


def pair_check(rate, x, y_ref, y_eval, probes) -> dict:
    out = {}
    for name, probe in probes.items():
        protocol = Protocol(probe=probe, estimation=("A",), reading=("B",))
        reading = analyze_pair(x, y_ref, y_eval, rate, protocol)["readings"]["B"]
        out[name] = {
            "captured": reading["d_probe"] / reading["d_audio"],
            "unrepresented": reading["unrepresented"] / reading["d_audio"],
        }
    skip = round(0.05 * rate)
    diff = (y_eval["B"] - y_ref["B"])[skip:]
    freqs, coh = coherence(x["B"][skip:], diff, fs=rate, nperseg=8192)
    band = (freqs > 80) & (freqs < 8000)
    out["coherence_x_diff_median_80_8k"] = float(np.median(coh[band]))
    return out


def jvm() -> dict:
    v2 = frozen_probes("v2")
    probes = {
        "hammerstein_5": frozen_probes("v1")["hammerstein_5"],
        "hammerstein_5_1024": Probe(5, 1024, 1e-4),
        "volterra_laguerre_3": v2["volterra_laguerre_3"],
    }
    rate, x, y_ref = jvm_audio("PreampOut", "B050_M050_T050_G050")
    out = {
        "capacity_centre": {
            "hammerstein_5": capacity(probes["hammerstein_5"], rate, x, y_ref),
            "hammerstein_5_1024": capacity(
                probes["hammerstein_5_1024"], rate, x, y_ref
            ),
        }
    }
    for setting in PAIRS:
        _, _, y_eval = jvm_audio("PreampOut", setting)
        out[setting] = pair_check(rate, x, y_ref, y_eval, probes)
        print(setting, flush=True)
    return out


def sweep() -> dict:
    from fssr_nam.product import vs100r
    from fssr_nam.product.hammerstein import Sweep

    sheet = {t["order"]: t for t in vs100r.sheet()}
    sweep_def = Sweep(f0=20.0, f1=20_000.0, seconds=5.0)
    protocol = HarmonicProtocol(order=9)
    out = {}
    for ref, ev, label in ((20, 39, "repeat"), (39, 38, "gain_7.5")):
        for name in ("sweep_-10dBFS", "sweep_-3dBFS"):
            s = common_section(sheet[ref], sheet[ev], name)
            result = analyze_sweep_pair(
                s["x"],
                s["y_ref"],
                s["y_eval"],
                48_000.0,
                sweep_def,
                protocol,
                s["x_eval"],
            )
            out[f"{label}|{name}"] = {
                k: {"complex_rel": v["complex_rel"], "level_ref_db": v["level_ref_db"]}
                for k, v in result.items()
                if k.startswith("order_")
            }
    assert set(SWEEPS) >= {"sweep_-10dBFS", "sweep_-3dBFS"}
    return out


def main() -> None:
    which = sys.argv[1]
    result = {
        "jvm": jvm,
        "sweep": sweep,
        "jvm_saturating": jvm_saturating,
        "jvm_causes": jvm_causes,
        "jvm_parallel": jvm_parallel,
        "slip_harm": slip_harm,
    }[which]()
    (HERE / f"gap_checks_{which}.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


def saturating_capacity(rate: int, x: dict, y: dict, gains=(1.0, 3.0, 10.0, 30.0)):
    """Capacity of a Hammerstein probe whose branches are x, x^2 and tanh(g x / s) for
    several slopes g (s = RMS of x on A), each through a 256-tap FIR, ridge 1e-4: a
    saturating basis instead of powers. Diagnostic only."""
    from scipy.linalg import solve
    from scipy.signal import fftconvolve

    from fssr_nam.product.probes import normal_equations, regularized

    scale = float(np.sqrt(np.mean(x["A"] ** 2)))

    def basis(v: np.ndarray) -> list:
        signals = [v / scale, (v / scale) ** 2] + [
            np.tanh(g * v / scale) for g in gains
        ]
        return signals

    probe = Probe(degree=len(gains) + 2, taps=256, ridge=1e-4)
    ua = basis(x["A"])
    norms = [float(np.sqrt(np.mean(s**2))) for s in ua]
    ua = [s / n for s, n in zip(ua, norms, strict=True)]
    gram, rhs = normal_equations(probe, ua, y["A"])
    theta = solve(regularized(probe, gram), rhs, assume_a="pos")
    ub = [s / n for s, n in zip(basis(x["B"]), norms, strict=True)]
    q = np.full(len(x["B"]), theta[0])
    for k, signal in enumerate(ub):
        q += fftconvolve(signal, theta[1 + 256 * k : 1 + 256 * (k + 1)])[: len(signal)]
    skip = round(0.05 * rate)
    return float(
        np.sqrt(np.mean((y["B"] - q)[skip:] ** 2) / np.mean(y["B"][skip:] ** 2))
    )


def jvm_saturating() -> dict:
    out = {}
    for setting in (
        "B050_M050_T050_G050",
        "B050_M050_T050_G000",
        "B050_M050_T050_G100",
    ):
        rate, x, y = jvm_audio("PreampOut", setting)
        out[setting] = {
            "hammerstein_5": capacity(frozen_probes("v1")["hammerstein_5"], rate, x, y),
            "saturating_basis": saturating_capacity(rate, x, y),
        }
        print(setting, out[setting], flush=True)
    return out


def jvm_input() -> tuple[np.ndarray, int]:
    import soundfile as sf
    from real_pairs import JVM

    x, rate = sf.read(JVM / "DRY/trainval/trainval.input.wav", dtype="float64")
    return x, rate


def jvm_target(setting: str) -> np.ndarray:
    import soundfile as sf
    from real_pairs import JVM

    path = JVM / f"PreampOut/trainval/{setting}/{setting}.trainval.target.wav"
    return sf.read(path, dtype="float64")[0]


def frame_rms(v: np.ndarray, frame: int) -> np.ndarray:
    n = len(v) // frame
    return np.sqrt(np.mean(v[: n * frame].reshape(n, frame) ** 2, axis=1))


def residual_ratio(y: np.ndarray, q: np.ndarray, rate: int) -> float:
    skip = round(0.05 * rate)
    return float(np.sqrt(np.mean((y - q)[skip:] ** 2) / np.mean(y[skip:] ** 2)))


def jvm_in_sample(x, y, rate) -> dict:
    probe = frozen_probes("v1")["hammerstein_5"]
    out = {}
    for name, span in (("0-150 s", 150), ("0-30 s", 30)):
        part = slice(0, span * rate)
        out[name] = residual_ratio(
            y[part], render(fit(probe, x[part], y[part]), x[part]), rate
        )
    return out


def jvm_silence_floor(x, rate) -> dict:
    frame = round(0.05 * rate)
    quiet = frame_rms(x, frame) < 10 ** (-60 / 20)
    out = {"quiet_frames": int(quiet.sum())}
    for gain in ("G000", "G050", "G100"):
        y = jvm_target(f"B050_M050_T050_{gain}")
        level = np.median(frame_rms(y, frame)[quiet]) / np.sqrt(np.mean(y**2))
        out[gain] = float(20 * np.log10(level))
    return out


def jvm_level_history(x, y, rate) -> dict:
    a, b = slice(0, 150 * rate), slice(215 * rate, 300 * rate)
    q = render(fit(frozen_probes("v1")["hammerstein_5"], x[a], y[a]), x[b])
    frame = round(0.05 * rate)
    xr, rr, yr = (frame_rms(v, frame) for v in (x[b], y[b] - q, y[b]))
    level = 20 * np.log10(xr + 1e-12)
    out = {}
    for lo, hi in ((-90, -50), (-50, -40), (-40, -30), (-30, -20), (-20, 0)):
        m = (level >= lo) & (level < hi)
        out[f"{lo}..{hi} dBFS"] = {
            "frames": int(m.sum()),
            "residual_ratio": float(np.sqrt(np.sum(rr[m] ** 2) / np.sum(yr[m] ** 2))),
        }
    mid = (level >= -30) & (level < -20)
    before = np.array(
        [np.max(level[max(0, i - 10) : i]) if i else -99.0 for i in range(len(level))]
    )
    for name, cond in (
        ("after_louder_6db", before > level + 6),
        ("no_louder", before <= level + 6),
    ):
        m = mid & cond
        out[name] = float(np.sqrt(np.sum(rr[m] ** 2) / np.sum(yr[m] ** 2)))
    return out


def jvm_drift(x, y, rate) -> dict:
    """Best integer lag of y against the H5 rendering, 10 s windows over the file."""
    q = render(
        fit(frozen_probes("v1")["hammerstein_5"], x[: 150 * rate], y[: 150 * rate]), x
    )
    out = {}
    for t0 in range(0, 300, 30):
        w = slice(t0 * rate, (t0 + 10) * rate)
        out[f"{t0}-{t0 + 10} s"] = lag(q[w], y[w], span=40)
    return out


def jvm_wiener_hammerstein(x, y, rate) -> dict:
    from fssr_nam.product.probes import WienerHammerstein, fit_wh, wh_render

    probe = WienerHammerstein(taps_in=32, taps_out=128, starts=3, iterations=25)
    a, b = slice(30 * rate, 31 * rate), slice(230 * rate, 231 * rate)
    state = fit_wh(probe, x[a], y[a]).states[0]
    return {
        "in_sample_1s": residual_ratio(y[a], wh_render(state, probe, x[a]), rate),
        "reading_1s": residual_ratio(y[b], wh_render(state, probe, x[b]), rate),
    }


def jvm_causes() -> dict:
    x, rate = jvm_input()
    y = jvm_target("B050_M050_T050_G050")
    out = {
        "in_sample_h5": jvm_in_sample(x, y, rate),
        "output_in_input_silence_db": jvm_silence_floor(x, rate),
        "residual_by_level_and_history": jvm_level_history(x, y, rate),
        "best_lag_samples_by_window": jvm_drift(x, y, rate),
        "wiener_hammerstein": jvm_wiener_hammerstein(x, y, rate),
    }
    return out


def parallel_capacity(x, y, rate, pole: float, gains=(1.0, 4.0, 16.0), taps=64) -> dict:
    """Parallel Wiener-Hammerstein probe with FIXED input filters: branches x and
    tanh(g z_j), z_j the 8 frozen Laguerre outputs (pole), each through a FIR; linear
    in its parameters. A = 0-60 s, B = 215-245 s."""
    from scipy.linalg import solve
    from scipy.signal import fftconvolve

    from fssr_nam.product.probes import laguerre, normal_equations, regularized

    a, b = slice(0, 60 * rate), slice(215 * rate, 245 * rate)
    scale = float(np.sqrt(np.mean(x[a] ** 2)))

    z_rms = np.sqrt(np.mean(laguerre(x[a] / scale, 8, pole) ** 2, axis=0)) + 1e-30

    def basis(v: np.ndarray) -> list:
        z = laguerre(v / scale, 8, pole) / z_rms  # normalised on A only, as the probe
        return [v / scale] + [np.tanh(g * z[:, j]) for j in range(8) for g in gains]

    ua = basis(x[a])
    norms = [float(np.sqrt(np.mean(s**2))) for s in ua]
    probe = Probe(degree=len(ua), taps=taps, ridge=1e-4)
    gram, rhs = normal_equations(
        probe, [s / n for s, n in zip(ua, norms, strict=True)], y[a]
    )
    theta = solve(regularized(probe, gram), rhs, assume_a="pos")
    out = {}
    for name, part in (("in_sample", a), ("reading", b)):
        ub = [s / n for s, n in zip(basis(x[part]), norms, strict=True)]
        q = np.full(len(ub[0]), theta[0])
        for k, sig in enumerate(ub):
            q += fftconvolve(sig, theta[1 + taps * k : 1 + taps * (k + 1)])[: len(sig)]
        out[name] = residual_ratio(y[part], q, rate)
    return out


def jvm_parallel() -> dict:
    x, rate = jvm_input()
    out = {}
    for gain in ("G050", "G100"):
        y = jvm_target(f"B050_M050_T050_{gain}")
        out[gain] = {f"pole_{p}": parallel_capacity(x, y, rate, p) for p in (0.8, 0.95)}
        print(gain, out[gain], flush=True)
    return out


def slip_harm() -> dict:
    """Harm of 24-sample slips the guard does not flag: delete 24 samples in the
    loopback AND the FX send of intact take 39 (as real slips behave), on a grid of
    times; for each, whether the guard flags it and the harmonic deviation per order
    against the intact take, as a fraction of the 20-39 repeat at the same order."""
    import real_pairs as rp
    from scipy.signal import resample_poly

    from fssr_nam.product import vs100r
    from fssr_nam.product.hammerstein import Sweep

    sheet = {t["order"]: t for t in vs100r.sheet()}
    takes = {t["order"]: t for p in rp.vs100r_pairs() for t in (p["ref"], p["eval"])}
    sweep_def, protocol = Sweep(f0=20.0, f1=20_000.0, seconds=5.0), HarmonicProtocol()
    out = {}
    for name in ("sweep_-10dBFS",):
        source = rp.sweep_source(name)
        lags = {
            o: rp.sweep_profile(source, vs100r.audio(t, name)[0])
            for o, t in takes.items()
        }
        modal = np.median(np.array(list(lags.values())), axis=0)
        loop, fx = vs100r.audio(sheet[39], name)
        x48, y48 = resample_poly(loop, 1, 4), resample_poly(fx, 1, 4)
        repeat = common_section(sheet[20], sheet[39], name)
        floor = analyze_sweep_pair(
            repeat["x"],
            repeat["y_ref"],
            repeat["y_eval"],
            48_000.0,
            sweep_def,
            protocol,
            repeat["x_eval"],
        )
        rows = {}
        for t in (0.5, 0.8, 1.0, 1.1, 1.21, 1.26, 1.5, 1.8, 2.5, 3.5, 4.76, 4.81, 4.9):
            i = round(t * vs100r.RATE)
            l2, f2 = (np.r_[v[:i], v[i + 24 :], np.zeros(24)] for v in (loop, fx))
            result = analyze_sweep_pair(
                x48,
                y48,
                resample_poly(f2, 1, 4),
                48_000.0,
                sweep_def,
                protocol,
                resample_poly(l2, 1, 4),
            )
            rows[f"{t}s"] = {
                "guard_flags": rp.broken(rp.sweep_profile(source, l2), modal),
                **{
                    o: {
                        "deviation": result[o]["complex_rel"],
                        "of_repeat": result[o]["complex_rel"] / floor[o]["complex_rel"],
                    }
                    for o in result
                    if o.startswith("order_")
                },
            }
        out[name] = {
            "repeat_20_39": {
                o: floor[o]["complex_rel"] for o in floor if o.startswith("order_")
            },
            "slips": rows,
        }
    return out


if __name__ == "__main__":
    main()
