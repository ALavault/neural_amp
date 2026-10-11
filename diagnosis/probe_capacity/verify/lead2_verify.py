#!/usr/bin/env python3
"""Skeptic verification of lead 2 (WH features fitted by gradient, then frozen).

Independent of the lead's code: the saved state dicts are read with torch.load only
and the cascade is re-implemented in numpy float64 (`wh_channels`, `wh_output`); the
readout is the harness's h.FeatureProbe (full Gram, scipy solve), not the lead's
cached-Cholesky `Readout`. Checks: full-model B ratio of the 9 take-20 runs and the 4
per-take models, frozen take-20 readout on B for 8 takes, linearity in y on a pair
whose reference is not take 20 or its repeats, the JVM centre number, and derived
checks from lead2_wh_gradient_frozen.json and the saved training curves.

Run from the repository root, CPU, 2 threads:
    uv run python diagnosis/probe_capacity/verify/lead2_verify.py
Writes diagnosis/probe_capacity/verify/lead2_verify.json.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.signal import oaconvolve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness as h  # noqa: E402

torch.set_num_threads(2)
LEAD = json.loads((h.HERE / "lead2_wh_gradient_frozen.json").read_text())
BASE = json.loads((h.HERE / "baseline.json").read_text())["probes"]
STORE = h.CACHE / "lead2"
OUT = HERE / "lead2_verify.json"
T0 = time.time()


def log(*args) -> None:
    print(f"{time.time() - T0:7.1f}s", *args, flush=True)


def load_state(name: str):
    saved = torch.load(STORE / name, weights_only=False)
    state = {k: v.double().numpy() for k, v in saved["state"].items()}
    y_scale = np.atleast_1d(np.asarray(saved["y_scale"], dtype=float))
    return state, float(saved["x_scale"]), y_scale, saved["info"]


def conv(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Causal convolution, zero initial state, first len(a) samples."""
    return oaconvolve(a, k)[: len(a)]


def wh_channels(st: dict, z: np.ndarray) -> list[np.ndarray]:
    """Last-stage channels of the cascade for an already scaled input z."""
    stages = sum(k.startswith("mix.") for k in st)
    zs = [np.asarray(z, dtype=float)]
    for s in range(stages):
        mix, fir = st[f"mix.{s}"], st[f"fir.{s}"]
        a, b, w, o = (st[f"{n}.{s}"] for n in ("gain", "bias", "weight", "offset"))
        new = []
        for c in range(mix.shape[0]):
            u = conv(sum(mix[c, j] * zs[j] for j in range(len(zs))), fir[c])
            v = sum(w[c, k] * np.tanh(a[c, k] * u + b[c, k]) for k in range(a.shape[1]))
            new.append(v + o[c])
        zs = new
    return zs


def wh_output(st: dict, x_scale: float, y_scale, x: np.ndarray, head: int = 0):
    xn = np.asarray(x, dtype=float) / x_scale
    zs = wh_channels(st, xn)
    out = sum(conv(zc, st["post"][head, c]) for c, zc in enumerate(zs))
    out = out + conv(xn, st["direct"][head, 0])
    return (out + np.asarray(st["constant"]).reshape(-1)[head]) * y_scale[head]


class Feats:
    """[x] + last-stage channels of x / x_scale; reads inputs['x'] only."""

    def __init__(self, st: dict, x_scale: float) -> None:
        self.st, self.x_scale, self.memo = st, x_scale, {}

    def __call__(self, inputs: dict) -> list[np.ndarray]:
        x = np.asarray(inputs["x"])
        key = (len(x), float(x[:48000].sum()), float(np.dot(x, x)))
        if key not in self.memo:
            self.memo[key] = [x, *wh_channels(self.st, x / self.x_scale)]
        return self.memo[key]


def g5_and_composition() -> dict:
    """Claims on the take-20 frozen readout over takes (chosen_B, from the JSON)."""
    by_take = {
        int(t): v[h.READ] for t, v in LEAD["chosen_B"]["capacity_by_take"].items()
    }
    h5 = {
        int(t): v["value"] for t, v in BASE["hammerstein_5"]["capacity_by_take"].items()
    }
    usable = [t for t in h.USABLE if t not in (1, 20, 39)]
    vals = [by_take[t] for t in usable]
    g5 = [t for t in usable if float(h.ROWS[t]["G"]) == 5.0]
    g5_cab_off = [t for t in g5 if h.ROWS[t]["cab"] == "off"]
    below = [t for t in usable if by_take[t] < 0.3]
    non_g5_off = [
        t for t in usable if float(h.ROWS[t]["G"]) != 5.0 and h.ROWS[t]["cab"] == "off"
    ]
    return {
        "usable_without_1_20_39": {
            "count": len(usable),
            "median": float(np.median(vals)),
            "min": min(vals),
            "max": max(vals),
            "below_0_3": len(below),
            "below_0_3_takes": below,
            "below_0_3_by_gain_cab": dict(
                Counter(
                    f"G{float(h.ROWS[t]['G']):g}_cab_{h.ROWS[t]['cab']}" for t in below
                )
            ),
            "excludes_1_20_39": not {1, 20, 39} & set(usable),
        },
        "other_g5_usable": {str(t): by_take[t] for t in g5},
        "other_g5_usable_range": [
            min(by_take[t] for t in g5),
            max(by_take[t] for t in g5),
        ],
        "other_g5_cab_off_range": [
            min(by_take[t] for t in g5_cab_off),
            max(by_take[t] for t in g5_cab_off),
        ],
        "g5_cab_off_above_0_3": [t for t in g5_cab_off if by_take[t] >= 0.3],
        "h5_on_other_g5_usable_range": [min(h5[t] for t in g5), max(h5[t] for t in g5)],
        "non_g5_cab_off_usable": {str(t): by_take[t] for t in non_g5_off},
        "non_g5_cab_off_usable_range": [
            min(by_take[t] for t in non_g5_off),
            max(by_take[t] for t in non_g5_off),
        ],
        "non_g5_cab_off_below_0_3": sum(by_take[t] < 0.3 for t in non_g5_off),
        "median_non_g5_cab_off": float(np.median([by_take[t] for t in non_g5_off])),
    }


def curves() -> dict:
    """Best V step from the saved .pt infos (not the JSON) and the V slope at the end
    of training; cosine schedule to 0.03 LR, so 'best at the last eval' is expected."""
    out = {}
    for path in sorted(STORE.glob("*.pt")):
        info = torch.load(path, weights_only=False)["info"]
        curve = info["curve"]
        v = {c["step"]: c["v_ratio"] for c in curve}
        last = curve[-1]["step"]
        out[path.stem] = {
            "best_step": info["best_step"],
            "last_eval_step": last,
            "best_is_last": info["best_step"] == last,
            "v_1000": v.get(1000),
            "v_1500": v.get(1500),
            "v_drop_1000_to_1500": v.get(1000) - v.get(1500),
            "v_drop_1400_to_1500": v.get(1400) - v.get(1500),
            "v_min_before_1500": min(c["v_ratio"] for c in curve[:-1]),
        }
    best_last = [k for k, r in out.items() if r["best_is_last"]]
    return {
        "per_model": out,
        "best_is_last_count": len(best_last),
        "models": len(out),
        "nolly_best_is_last": sum(not k.startswith("jvm") for k in best_last),
        "nolly_models": sum(not k.startswith("jvm") for k in out),
    }


def json_checks() -> dict:
    runs = LEAD["runs"]
    out = {}
    out["take20_full_gaps"] = {
        k: {
            "V_minus_A": r["full_model_take20"][h.VALID]
            - r["full_model_take20"][h.FIT],
            "B_minus_V": r["full_model_take20"][h.READ]
            - r["full_model_take20"][h.VALID],
        }
        for k, r in runs.items()
    }
    out["max_abs_V_minus_A"] = max(
        abs(v["V_minus_A"]) for v in out["take20_full_gaps"].values()
    )
    out["B_minus_V_range"] = [
        min(v["B_minus_V"] for v in out["take20_full_gaps"].values()),
        max(v["B_minus_V"] for v in out["take20_full_gaps"].values()),
    ]
    # depth: readout B (frozen, take-20 features) by config at each gain
    depth = {}
    for g in ("0", "2.5", "5", "7.5", "10"):
        depth[g] = {
            c: [
                min(
                    r["readout_B"][g][h.READ] for r in runs.values() if r["config"] == c
                ),
                max(
                    r["readout_B"][g][h.READ] for r in runs.values() if r["config"] == c
                ),
            ]
            for c in ("wh1", "wh2", "wh3")
        }
    out["readout_B_by_depth"] = depth
    # transfer matrix: B ratio against gain distance, off-diagonal
    gains = {30: 0.0, 16: 2.5, 20: 5.0, 6: 7.5, 7: 10.0}
    mono = {}
    for src, row in LEAD["transfer"]["matrix"].items():
        pts = sorted(
            (abs(float(g) - gains[int(src)]), v[h.READ])
            for g, v in row["readout_by_gain"].items()
            if float(g) != gains[int(src)]
        )
        dist = np.array([p[0] for p in pts])
        val = np.array([p[1] for p in pts])
        rank_d = np.argsort(np.argsort(dist))
        rank_v = np.argsort(np.argsort(val))
        mono[src] = {
            "distance_value": pts,
            "spearman_like": float(np.corrcoef(rank_d, rank_v)[0, 1]),
        }
    out["transfer_vs_gain_distance"] = mono
    chosen = runs[LEAD["selection"]["chosen"]]["readout_B"]
    out["drive_bank_gain_over_chosen"] = {
        g: chosen[g][h.READ] - v[h.READ]
        for g, v in LEAD["drive_bank"]["capacity_by_gain"].items()
    }
    up = LEAD["upper_per_take"]
    out["upper_full_model_B"] = {t: u["full_model"][h.READ] for t, u in up.items()}
    out["upper_own_readout_B"] = {
        t: u["readout_own_features"][h.READ] for t, u in up.items()
    }
    multi = LEAD["multi_take"]["capacity_by_take"]
    seen = {30, 16, 20, 6, 7, 1, 39}
    frozen35 = {t: v[h.READ] for t, v in multi.items() if int(t) not in seen}
    out["multi_take_35_frozen_max"] = [len(frozen35), max(frozen35.values())]
    out["multi_take_35_frozen_argmax"] = max(frozen35, key=frozen35.get)
    shares = LEAD["transfer"]["gain_pair_shares"]
    clean = [s for s in shares if s["clean"]]
    out["clean_gain_pair_shares"] = {
        f"{s['ref']}->{s['eval']}_from_{s['features_from_take']}": s["share_from_x"]
        for s in clean
    }
    out["clean_share_range"] = [
        min(s["share_from_x"] for s in clean),
        max(s["share_from_x"] for s in clean),
    ]
    return out


def main() -> None:
    result = {
        "protocol": "fit di_train, read di_test, 48 kHz, SKIP 2400, floor 1e-4; "
        "numpy float64 re-implementation of the saved WH models; readout = "
        "h.FeatureProbe([x] + channels, taps 256, ridge 1e-4)",
        "json_g5_and_composition": g5_and_composition(),
        "json_checks": json_checks(),
        "curves": curves(),
    }
    OUT.write_text(json.dumps(h.nr.finite(result), indent=1) + "\n")
    log("json checks written")

    x = h.load("x", (h.FIT, h.VALID, h.READ))
    y20 = h.load(20, (h.FIT, h.VALID, h.READ))
    # 1. full model, take 20, 9 runs, on B (and A, V for wh3_s0)
    full = {}
    for cfg in ("wh1", "wh2", "wh3"):
        for seed in (0, 1, 2):
            st, xs, ys, _ = load_state(f"take_20_{cfg}_s{seed}.pt")
            sections = (
                (h.FIT, h.VALID, h.READ) if (cfg, seed) == ("wh3", 0) else (h.READ,)
            )
            mine = {s: h.ratio(y20[s], wh_output(st, xs, ys, x[s])) for s in sections}
            lead = LEAD["runs"][f"{cfg}_s{seed}"]["full_model_take20"]
            full[f"{cfg}_s{seed}"] = {
                "mine": mine,
                "lead": {s: lead[s] for s in sections},
                "max_abs_gap": max(abs(mine[s] - lead[s]) for s in sections),
            }
            log("full", cfg, seed, mine)
    result["full_model_take20"] = full
    # 2. per-take models (upper), full model on B
    upper = {}
    for take in (30, 16, 6, 7):
        st, xs, ys, _ = load_state(f"take_{take:02d}_wh3_s0.pt")
        y = h.load(take, (h.READ,))[h.READ]
        mine = h.ratio(y, wh_output(st, xs, ys, x[h.READ]))
        lead = LEAD["upper_per_take"][str(take)]["full_model"][h.READ]
        upper[str(take)] = {"mine": mine, "lead": lead, "gap": abs(mine - lead)}
        log("upper", take, mine, lead)
    result["upper_full_model_B"] = upper
    OUT.write_text(json.dumps(h.nr.finite(result), indent=1) + "\n")

    # 3. frozen take-20 (wh3_s0) features, readout through h.FeatureProbe
    st, xs, ys, _ = load_state("take_20_wh3_s0.pt")
    feats = Feats(st, xs)
    probe = h.FeatureProbe(feats, taps=256, ridge=1e-4)
    readout = {}
    lead_by_take = LEAD["chosen_B"]["capacity_by_take"]
    for take in (3, 14, 16, 6, 7, 30, 42, 37, 20):
        t1 = time.time()
        mine = h.output_capacity(probe, take)
        lead = lead_by_take[str(take)][h.READ]
        readout[str(take)] = {
            "G": h.ROWS[take]["G"],
            "setting": h.ROWS[take]["setting"],
            "mine_B": mine,
            "lead_B": lead,
            "gap": abs(mine - lead),
            "label": "training take (nonlinear fit)"
            if take == 20
            else "frozen reading",
        }
        log("readout", take, mine, lead, f"{time.time() - t1:.0f}s")
        result["frozen_readout_B"] = readout
        OUT.write_text(json.dumps(h.nr.finite(result), indent=1) + "\n")
    # V of the 5 gain takes -> selection score of wh3_s0 (V only)
    v_scores = {}
    for g, take in h.GAIN_TAKES.items():
        v_scores[f"{g:g}"] = h.output_capacity(probe, take, read=h.VALID)
    result["selection_score_wh3_s0"] = {
        "mine": float(np.mean(list(v_scores.values()))),
        "lead": LEAD["selection"]["scores"]["wh3_s0"],
        "by_gain_V": v_scores,
    }
    log("selection score", result["selection_score_wh3_s0"])
    OUT.write_text(json.dumps(h.nr.finite(result), indent=1) + "\n")

    # 4. linearity in y: one pair without take 20 or its repeats, and the lead's (1, 7)
    lin = {}
    for ref, ev in ((16, 7), (1, 7)):
        lin[f"{ref}->{ev}"] = h.linearity_check(probe, ref, ev)
        log("linearity", ref, ev, lin[f"{ref}->{ev}"])
    lin["lead_1_7"] = LEAD["chosen_B"]["linearity_check_1_7"]
    result["linearity"] = lin
    # repeats of take 20 on B
    yb = {t: h.load(t, (h.READ,))[h.READ] for t in (1, 20, 39)}
    result["repeat_distance_B"] = {
        f"{t}_vs_20": h.nr.rms((yb[t] - yb[20])[h.SKIP :]) / h.nr.rms(yb[20][h.SKIP :])
        for t in (1, 39)
    }
    OUT.write_text(json.dumps(h.nr.finite(result), indent=1) + "\n")

    # 5. JVM centre: full model and readout on B (features trained on this setting)
    from real_pairs import jvm_audio

    st, xs, ys, _ = load_state("jvm_preampout_B050_M050_T050_G050_wh3_s0.pt")
    rate, xj, yj = jvm_audio("PreampOut", "B050_M050_T050_G050")
    skip = round(0.05 * rate)

    def res(y, q):
        return h.nr.rms((y - q)[skip:]) / max(h.nr.rms(y[skip:]), h.FLOOR)

    full_b = res(yj["B"], wh_output(st, xs, ys, xj["B"]))
    jp = h.FeatureProbe(Feats(st, xs), taps=256, ridge=1e-4).fit(
        {"x": xj["A"]}, yj["A"]
    )
    read_b = res(yj["B"], jp.predict({"x": xj["B"]}))
    lead = LEAD["jvm"]
    result["jvm_centre"] = {
        "full_model_B": {"mine": full_b, "lead": lead["full_model_centre"]["B"]},
        "readout_B": {
            "mine": read_b,
            "lead": lead["readout"]["B050_M050_T050_G050"]["readout_B"],
        },
        "label": "features trained on this setting's A: a nonlinear fit, other device",
    }
    log("jvm", result["jvm_centre"])
    result["runtime_s"] = time.time() - T0
    OUT.write_text(json.dumps(h.nr.finite(result), indent=1) + "\n")
    log("done")


def levels() -> None:
    """Level-stratified B residual of the frozen take-20 readout at G2.5 and G10
    (claim 4: the H5 quiet-bin excess disappears), merged into the JSON."""
    result = json.loads(OUT.read_text())
    st, xs, _, _ = load_state("take_20_wh3_s0.pt")
    probe = h.FeatureProbe(Feats(st, xs), taps=256, ridge=1e-4)
    x_b = h.section_inputs(h.READ)["x"]
    out = {}
    for g, take in ((2.5, 16), (10.0, 7)):
        q = h.fit_read(probe, h.load(take, (h.FIT,))[h.FIT])
        mine = h.level_capacity_from(x_b, h.load(take, (h.READ,))[h.READ], q)
        lead = LEAD["chosen_B"]["capacity_by_level"][f"{g:g}"]
        out[f"{g:g}"] = {
            "mine": [b["capacity"] for b in mine],
            "lead": [b["capacity"] for b in lead],
            "h5": [
                b["capacity"]
                for b in BASE["hammerstein_5"]["capacity_by_level"][f"{g:g}"]
            ],
            "max_abs_gap": max(
                abs(a["capacity"] - b["capacity"])
                for a, b in zip(mine, lead, strict=True)
            ),
        }
        log("levels", g, out[f"{g:g}"])
    result["level_capacity_B"] = out
    OUT.write_text(json.dumps(h.nr.finite(result), indent=1) + "\n")


if __name__ == "__main__":
    levels() if sys.argv[1:] == ["levels"] else main()
