#!/usr/bin/env python3
"""Lead 1, output-referenced probe: model the CHANGE between two outputs, not the amp.

Exploration (failure-mining), CPU only, at most 3 processes or threads. A probe gets
the reference take's output y_ref as an input and is fitted on A = di_train to the
difference d = y_eval - y_ref, read on B = di_test (Nolly renders, 48 kHz, harness
protocol). Every variant has features fixed before seeing the target, so it stays
linear in the target. The x probes of the frozen reading are the baselines: their
shares come from the harness's cached predictions (fitted on A, read on B).

Selection: one variant is chosen on V = di_validation (median share over the listed
non-repeat pairs), then B is reported for every variant. B never enters a fit. The
prediction of the fragile hypothesis (PREDICTION) is written into the JSON before B.

Known identity (checked in `checks`): when y_ref is a feature and its FIR can carry the
identity, the residual of fitting d equals the residual of fitting y_eval with the
same features, up to the ridge. The probe of the change and the conditional probe of
the eval output are the same probe; only the normalisation differs. Its fit of y_ref
is then near exact (r_ref about 0), so the frozen bound |d_audio - d_probe| <=
r_ref + r_eval reduces to the triangle inequality: the bound is vacuous here.

VS100R (discriminating observation, defect unknown): model outputs (arm C SSM-WaveNet
FiLM and the D-0014 grey boxes, last checkpoints, CPU) against the device's FX send,
on a few test takes. The probe is fitted on di_validation and read on di_test (no
hyper-parameter is selected there: variants are those of the Nolly stage), both
directions: ref = device (d = y_model - y_device) and ref = model (d = y_device -
y_model, the part a filter after the model cannot remove). Model outputs are cached
in tmp/probe_capacity/lead1/vs100r/.

    uv run python diagnosis/probe_capacity/lead1_output_referenced.py --stage nolly
    uv run python diagnosis/probe_capacity/lead1_output_referenced.py --stage checks
    uv run python diagnosis/probe_capacity/lead1_output_referenced.py --stage vs100r
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as h  # noqa: E402

OUT = HERE / "lead1_output_referenced.json"
WORK = h.CACHE / "lead1"
LEAD = 16  # samples of y_ref ahead of the current sample in the *_lead16 variant
SAT = (3.0, 10.0, 30.0)  # tanh(a y_ref), fixed slopes (output RMS about 0.1)
RIDGE = 1e-4
X_PROBES = ("fir", "hammerstein_3", "hammerstein_5", "volterra_laguerre_2")
X_PROBES += ("volterra_laguerre_3",)
LEVEL_VARIANTS = ("yfir", "yfir1024", "yh5", "yh5_1024", "x5_yfir", "x5_yh5")


def powers(s: np.ndarray, k: int) -> list[np.ndarray]:
    return [np.asarray(s) ** j for j in range(1, k + 1)]


def lead(s: np.ndarray, n: int = LEAD) -> np.ndarray:
    """s advanced by n samples (zeros at the end): its causal FIR covers lags -n.."""
    return np.concatenate([np.asarray(s[n:]), np.zeros(n)])


def f_yfir(i: dict) -> list[np.ndarray]:
    return [i["y_ref"]]


def f_yfir_lead(i: dict) -> list[np.ndarray]:
    return [lead(i["y_ref"])]


def f_yh5(i: dict) -> list[np.ndarray]:
    return powers(i["y_ref"], 5)


def f_ysat(i: dict) -> list[np.ndarray]:
    return [i["y_ref"]] + [np.tanh(a * np.asarray(i["y_ref"])) for a in SAT]


def f_x5_yfir(i: dict) -> list[np.ndarray]:
    return [*powers(i["x"], 5), i["y_ref"]]


def f_x5_yh5(i: dict) -> list[np.ndarray]:
    return powers(i["x"], 5) + powers(i["y_ref"], 5)


# name -> (features, taps, uses x, description); ridge RIDGE (frozen H5's) for all
VARIANTS = {
    "yfir": (f_yfir, 256, False, "FIR 256 on y_ref"),
    "yfir1024": (f_yfir, 1024, False, "FIR 1024 on y_ref"),
    "yfir_lead16": (f_yfir_lead, 256, False, "FIR 256 on y_ref, lags -16..239"),
    "yh5": (f_yh5, 256, False, "Hammerstein 5 on y_ref (y_ref^1..5, FIR 256 each)"),
    "yh5_1024": (f_yh5, 1024, False, "Hammerstein 5 on y_ref, FIR 1024 each"),
    "ysat": (f_ysat, 256, False, "y_ref and tanh(a y_ref), a = 3, 10, 30, FIR 256"),
    "x5_yfir": (f_x5_yfir, 256, True, "x^1..5 and y_ref, FIR 256 each"),
    "x5_yh5": (f_x5_yh5, 256, True, "x^1..5 and y_ref^1..5, FIR 256 each"),
}

PREDICTION = {
    "written": (
        "2026-10-11, in this script before its first reading of B. The scratch study "
        "(session scratchpad, run_b.json) had already read B for FIR 256 on y_ref on "
        "pairs 1->2, 1->4, 1->7, 39->30, 39->35, 39->36; the other pairs, the other "
        "variants, the reversed pairs and VS100R are unseen."
    ),
    "fragile_hypothesis": (
        "An output-referenced probe represents a change acting after the main "
        "clipping (tone knobs, cabinet) and fails for a gain change (before clipping)."
    ),
    "metric": "share = RMS(d_B - q_B) / RMS(d_B), selected variant (chosen on V)",
    "classes": {
        "tone": "one-knob B, M, T, P pairs and combinations with G unchanged",
        "gain": "one-knob G pairs and combinations with G changed",
        "cab": "cabinet on vs off, same amp settings: positive control",
        "repeat": "same setting rendered twice: floor, not interpreted",
    },
    "rule": {
        "supported": "median(gain) >= 0.6 and median(tone) <= 0.5 and "
        "median(gain) - median(tone) >= 0.15",
        "refuted": "median(gain) - median(tone) < 0.05 or median(gain) < 0.5",
        "partial": "otherwise",
    },
    "expected_controls": {"cab": "median <= 0.2", "repeat": "median >= 0.9"},
    "secondary": (
        "Gain pairs: nonlinear y_ref features (yh5, ysat) gain more over yfir when the "
        "reference has the LOWER gain (re-clipping a less clipped output is possible, "
        "un-clipping is not). Supposed mechanism, measured by the reversed pairs."
    ),
    "vs100r": {
        "written": (
            "2026-10-11, before any VS100R inference of this script, with "
            "demo/greybox_deep/evaluation_dry.json known: arm C needs its output "
            "delayed by about 1 sample (tau +0.95..+1.05; ln ESR -3.95 raw, -5.5 "
            "aligned, i.e. about 79 % of its error energy is that delay); GREY tau "
            "about -0.5 with ln ESR about -1.5 barely changed by alignment."
        ),
        "expected": {
            "C": "the 1-sample advance is a filter of y_ref only in the causal "
            "direction: share(ref_device, yfir) - share(ref_device, yfir_lead16) >= "
            "0.1, and share(ref_model, yfir) <= 0.6 on the seen-pair takes; the x "
            "probes leave more (H5 share >= share(ref_model, yfir) + 0.1)",
            "grey_boxes": "GREY, L, F3 errors are mostly a wrong nonlinearity, not a "
            "filter of either output: share(ref_model, yfir) >= 0.6",
            "take_16": "unseen gain-contour pair: the C error there is a wrong knob "
            "response, expected share(ref_model, yfir) higher than on the other takes",
        },
    },
}


def gain(take: int) -> float:
    return float(h.ROWS[take]["G"])


def classify(pair: dict) -> str:
    if pair["kind"] == "repeat":
        return "repeat"
    if pair["kind"] == "cab":
        return "cab"
    return "gain" if gain(pair["ref"]) != gain(pair["eval"]) else "tone"


def nolly_pairs(only: list[str] | None) -> list[dict]:
    """The 34 listed pairs, the 8 gate-4 pairs (flagged) and the 19 + 1 one-knob pairs
    reversed (eval as reference), kind one_knob_reversed."""
    out = h.pairs(excluded=True)
    out += [
        {
            "ref": p["eval"],
            "eval": p["ref"],
            "kind": "one_knob_reversed",
            **{"excluded": p["excluded"]},
        }
        for p in h.pairs(("one_knob",), excluded=True)
    ]
    for p in out:
        p["class"] = classify(p)
        p["knob"] = h.knob(p) if p["kind"].startswith("one_knob") else None
        p["G_ref"], p["G_eval"] = gain(p["ref"]), gain(p["eval"])
    if only:
        wanted = {tuple(int(v) for v in s.split(":")) for s in only}
        out = [p for p in out if (p["ref"], p["eval"]) in wanted]
    return out


def frozen_shares(ref: int, take: int) -> tuple[dict, dict, np.ndarray]:
    """Share of d_B left by each frozen x probe (cached predictions, linear in y:
    q(d) = q(y_eval) - q(y_ref)), and H5's level split."""
    d = h.difference(ref, take, h.READ)
    shares, q_h5 = {}, None
    for name in X_PROBES:
        q = h.frozen_prediction(name, take) - h.frozen_prediction(name, ref)
        shares[name] = h.ratio(d, q, floor=0.0)
        if name == "hammerstein_5":
            q_h5 = q
    level = h.level_capacity_from(h.section_inputs(h.READ)["x"], d, q_h5)
    return shares, {"hammerstein_5": level}, d


def reading_share(ref: int, take: int) -> dict | None:
    for p in h.READING["pairs"]:
        if (p["ref"], p["eval"]) == (ref, take):
            return {
                n: p["probes"][n][h.READ]["unrepresented"]
                / p["probes"][n][h.READ]["d_audio_norm"]
                for n in X_PROBES
            }
    return None


def nolly_job(pair: dict) -> dict:
    start = time.time()
    ref, take = pair["ref"], pair["eval"]
    with threadpool_limits(1):
        sections = (h.FIT, h.VALID, h.READ)
        d = {s: h.difference(ref, take, s) for s in sections}
        inputs = {s: h.section_inputs(s, ref) for s in sections}
        out = dict(pair)
        out["d_rel"] = {
            s: h.nr.rms(d[s][h.SKIP :]) / h.nr.rms(inputs[s]["y_ref"][h.SKIP :])
            for s in (h.VALID, h.READ)
        }
        frozen, level, _ = frozen_shares(ref, take)
        out["frozen_share_B"] = frozen
        listed = reading_share(ref, take)
        out["frozen_share_reading_B"] = listed
        out["share"], out["level_B"] = {}, level
        for name, (features, taps, _, _) in VARIANTS.items():
            probe = h.FeatureProbe(features, taps=taps, ridge=RIDGE)
            probe.fit(inputs[h.FIT], d[h.FIT])
            out["share"][name] = {}
            for s in (h.VALID, h.READ):
                q = probe.predict(inputs[s])
                out["share"][name][s] = h.ratio(d[s], q, floor=0.0)
                if s == h.READ and name in LEVEL_VARIANTS:
                    out["level_B"][name] = h.level_capacity_from(
                        inputs[s]["x"], d[s], q
                    )
    out["seconds"] = time.time() - start
    print(f"{ref}->{take} {out['seconds']:.0f} s", flush=True)
    return out


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def selection(rows: list[dict]) -> dict:
    """Variant with the lowest median share on V over the listed (not excluded,
    not reversed) one-knob, combination and cab pairs; also the best y_ref-only one."""
    pool = [
        r
        for r in rows
        if not r["excluded"]
        and r["class"] != "repeat"
        and r["kind"] != ("one_knob_reversed")
    ]
    med = {v: median([r["share"][v][h.VALID] for r in pool]) for v in VARIANTS}
    y_only = [v for v in VARIANTS if not VARIANTS[v][2]]
    return {
        "rule": "lowest median share on di_validation over the listed one-knob, "
        "combination and cab pairs (gate-4 pairs, reversed pairs, repeats excluded)",
        "pairs": len(pool),
        "median_V": med,
        "selected": min(med, key=med.get),
        "selected_y_only": min(y_only, key=med.get),
    }


def grouped(rows: list[dict]) -> dict[str, list[dict]]:
    """Pairs by class; reversed and gate-4 pairs in their own groups."""
    out = {}
    for r in rows:
        if r["excluded"]:
            key = f"{r['class']}_gate4_excluded"
        elif r["kind"] == "one_knob_reversed":
            key = f"{r['class']}_reversed"
        else:
            key = r["class"]
        out.setdefault(key, []).append(r)
    return dict(sorted(out.items()))


def summary(rows: list[dict], selected: str) -> dict:
    """Median B shares per class (listed pairs; reversed and gate-4 separately)."""
    out = {}
    for key, members in grouped(rows).items():
        out[key] = {
            "n": len(members),
            "pairs": [f"{r['ref']}->{r['eval']}" for r in members],
            "median_share_B": {
                v: median([r["share"][v][h.READ] for r in members]) for v in VARIANTS
            },
            "median_frozen_share_B": {
                n: median([r["frozen_share_B"][n] for r in members]) for n in X_PROBES
            },
            "median_d_rel_B": median([r["d_rel"][h.READ] for r in members]),
        }
    tone, gain_ = (out[k]["median_share_B"][selected] for k in ("tone", "gain"))
    if gain_ >= 0.6 and tone <= 0.5 and gain_ - tone >= 0.15:
        verdict = "supported"
    elif gain_ - tone < 0.05 or gain_ < 0.5:
        verdict = "refuted"
    else:
        verdict = "partial"
    return {
        "by_class": out,
        "verdict": {
            "variant": selected,
            "median_tone_B": tone,
            "median_gain_B": gain_,
            "median_cab_B": out["cab"]["median_share_B"][selected],
            "median_repeat_B": out["repeat"]["median_share_B"][selected],
            "label": verdict,
        },
    }


def direction(rows: list[dict]) -> list[dict]:
    """Gain pairs in both directions: share of yfir, yh5, ysat and the gain of the
    nonlinear y_ref features over yfir, by whether the reference has the lower gain."""
    out = []
    for r in rows:
        if r["class"] != "gain" or not r["kind"].startswith("one_knob"):
            continue
        names = ("yfir", "yfir_lead16", "yh5", "ysat", "x5_yh5")
        s = {v: r["share"][v][h.READ] for v in names}
        out.append(
            {
                "pair": f"{r['ref']}->{r['eval']}",
                "G_ref": r["G_ref"],
                "G_eval": r["G_eval"],
                "ref_lower_gain": r["G_ref"] < r["G_eval"],
                "excluded": r["excluded"],
                "share_B": s,
                "conditional_capacity_B": {
                    v: r["conditional_capacity_B"][v] for v in names
                }
                if "conditional_capacity_B" in r
                else None,
                "yfir_minus_yh5": s["yfir"] - s["yh5"],
                "yfir_minus_ysat": s["yfir"] - s["ysat"],
                "yfir_minus_yfir_lead16": s["yfir"] - s["yfir_lead16"],
            }
        )
    return out


def load_result(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save(path: Path, result: dict) -> None:
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(h.nr.finite(result), indent=1) + "\n")
    partial.replace(path)


def stage_nolly(args, result: dict) -> None:
    start = time.time()
    todo = nolly_pairs(args.only)
    rows = []
    with ProcessPoolExecutor(args.jobs) as pool:
        for row in pool.map(nolly_job, todo):
            rows.append(row)
    chosen = selection(rows)
    block = {
        "pairs": rows,
        "selection": chosen,
        "direction_gain_pairs": direction(rows),
        "seconds": time.time() - start,
    }
    if not args.only:
        block |= summary(rows, chosen["selected"])
        block["summary_yfir"] = summary(rows, "yfir")["verdict"]
    result["nolly"] = block


def stage_post(args, result: dict) -> None:
    """Post hoc (after the B reading): conditional capacity = residual / RMS(y_eval)
    on B, i.e. the share times RMS(d) / RMS(y_eval), floor 1e-4. With y_ref among
    the features it is the output capacity of a probe that also receives y_ref (the
    `checks` stage measures the equality), comparable with the frozen output
    capacity of the eval take (baseline.json), unlike the share, whose denominator
    RMS(d) depends on the direction when the two outputs differ in level."""
    base = json.loads((HERE / "baseline.json").read_text())["probes"]
    keep = slice(h.SKIP, None)
    rows = result["nolly"]["pairs"]
    for r in rows:
        y = {
            t: h.nr.rms(h.load(t, (h.READ,))[h.READ][keep])
            for t in (r["ref"], r["eval"])
        }
        rms_d = r["d_rel"][h.READ] * y[r["ref"]]
        scale = max(y[r["eval"]], h.FLOOR)
        r["eval_rel_B"] = rms_d / scale
        r["conditional_capacity_B"] = {
            v: s[h.READ] * rms_d / scale for v, s in r["share"].items()
        }
        r["frozen_capacity_eval_B"] = {
            n: base[n]["capacity_by_take"][str(r["eval"])]["value"] for n in X_PROBES
        }
    by_class = {}
    for key, members in grouped(rows).items():
        by_class[key] = {
            "n": len(members),
            "median_conditional_capacity_B": {
                v: median([r["conditional_capacity_B"][v] for r in members])
                for v in VARIANTS
            },
            "median_frozen_capacity_eval_B": {
                n: median([r["frozen_capacity_eval_B"][n] for r in members])
                for n in X_PROBES
            },
        }
    ladder = {}
    for g, take in h.GAIN_TAKES.items():
        found = [
            r
            for r in rows
            if r["eval"] == take and r["kind"] == "one_knob" and not r["excluded"]
        ]
        for r in found:
            ladder[f"{g:g}"] = {
                "pair": f"{r['ref']}->{r['eval']}",
                "conditional_capacity_B": r["conditional_capacity_B"],
                "frozen_capacity_eval_B": r["frozen_capacity_eval_B"],
            }
    rows_dir = direction(rows)
    lower = [d["yfir_minus_yh5"] for d in rows_dir if d["ref_lower_gain"]]
    higher = [d["yfir_minus_yh5"] for d in rows_dir if not d["ref_lower_gain"]]
    result["nolly"]["direction_gain_pairs"] = rows_dir
    result["nolly"]["conditional"] = {
        "post_hoc": True,
        "by_class": by_class,
        "gain_ladder": ladder,
        "gain_ladder_note": "gain 5 (take 20) has no one-knob pair from another "
        "setting: its only reference is a repeat (oracle, ref = same setting)",
        "secondary_prediction": {
            "median_yfir_minus_yh5_ref_lower_gain": median(lower),
            "median_yfir_minus_yh5_ref_higher_gain": median(higher),
            "max_abs": max(abs(v) for v in lower + higher),
        },
    }


CHECK_PAIRS = ((1, 2), (1, 3), (1, 4), (1, 5), (1, 7), (1, 20), (20, 15), (32, 40))


def check_job(job: tuple[str, int, int]) -> dict:
    """Linearity in y and the bound terms of one variant on one pair (5 fits)."""
    name, ref, take = job
    features, taps, _, _ = VARIANTS[name]
    with threadpool_limits(1):
        probe = h.FeatureProbe(features, taps=taps, ridge=RIDGE)
        gap = h.linearity_check(probe, ref, take)
        y = {t: h.load(t, (h.FIT, h.READ)) for t in (ref, take)}
        q = {t: h.fit_read(probe, y[t][h.FIT], ref) for t in (ref, take)}
        d = y[take][h.READ] - y[ref][h.READ]
        q_d = h.fit_read(probe, y[take][h.FIT] - y[ref][h.FIT], ref)
        keep = slice(h.SKIP, None)
        rms = h.nr.rms
        r_ref = rms((y[ref][h.READ] - q[ref])[keep])
        r_eval = rms((y[take][h.READ] - q[take])[keep])
        d_audio, d_probe = rms(d[keep]), rms((q[take] - q[ref])[keep])
        unrepresented = rms((d - q_d)[keep])
    return {
        "variant": name,
        "pair": f"{ref}->{take}",
        "linearity_gap": gap,
        "r_ref_rel": r_ref / rms(y[ref][h.READ][keep]),
        "r_eval_rel": r_eval / rms(y[take][h.READ][keep]),
        "r_eval_over_unrepresented": r_eval / unrepresented,
        "bound": {
            "gap": abs(d_audio - d_probe),
            "limit": r_ref + r_eval,
            "holds": abs(d_audio - d_probe) <= (r_ref + r_eval) * (1 + 1e-9) + 1e-15,
            "r_ref_share_of_limit": r_ref / (r_ref + r_eval),
        },
    }


def stage_checks(args, result: dict) -> None:
    start = time.time()
    names = sorted({"yfir", result["nolly"]["selection"]["selected"]})
    jobs = [(n, r, e) for n in names for r, e in CHECK_PAIRS]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(check_job, jobs))
    result["checks"] = {
        "pairs": [f"{r}->{e}" for r, e in CHECK_PAIRS],
        "rows": rows,
        "max_linearity_gap": max(r["linearity_gap"] for r in rows),
        "bound_holds_everywhere": all(r["bound"]["holds"] for r in rows),
        "seconds": time.time() - start,
    }


# ---------------------------------------------------------------- VS100R

VS_ARMS = ("C", "GREY", "L", "F3")


def vs_path(*parts: str) -> Path:
    return WORK / "vs100r" / Path(*parts)


def vs_job(job: tuple[str, int]) -> dict:
    """Shares of y_model - y_device left by the x probes and by every y_ref variant,
    both directions; fitted on di_validation, read on di_test."""
    run_id, take = job
    start = time.time()
    sections = (h.VALID, h.READ)
    with threadpool_limits(1):
        x, dev, mod = (
            {
                s: np.load(vs_path(folder, f"take_{take:02d}_{s}{suffix}.npy"))
                for s in sections
            }
            for folder, suffix in (("device", "_x"), ("device", "_y"), (run_id, ""))
        )
        d = {s: mod[s] - dev[s] for s in sections}
        keep = slice(h.SKIP, None)
        out = {
            "run_id": run_id,
            "take": take,
            "d_rel_B": h.nr.rms(d[h.READ][keep]) / h.nr.rms(dev[h.READ][keep]),
            "frozen_x_share_B": {},
            "share_B": {"ref_device": {}, "ref_model": {}},
        }
        for name in ("fir", "hammerstein_5", "volterra_laguerre_3"):
            probe = h.FrozenProbe(name).fit({"x": x[h.VALID]}, d[h.VALID])
            q = probe.predict({"x": x[h.READ]})
            out["frozen_x_share_B"][name] = h.ratio(d[h.READ], q, floor=0.0)
        for side, (ref, ev) in {
            "ref_device": (dev, mod),
            "ref_model": (mod, dev),
        }.items():
            inputs = {s: {"x": x[s], "y_ref": ref[s]} for s in sections}
            for name, (features, taps, _, _) in VARIANTS.items():
                probe = h.FeatureProbe(features, taps=taps, ridge=RIDGE)
                probe.fit(inputs[h.VALID], ev[h.VALID] - ref[h.VALID])
                q = probe.predict(inputs[h.READ])
                out["share_B"][side][name] = h.ratio(
                    ev[h.READ] - ref[h.READ], q, floor=0.0
                )
    out["seconds"] = time.time() - start
    print(f"{run_id} take {take}: {out['seconds']:.0f} s", flush=True)
    return out


def vs_device(vs100r, takes: dict, chosen: list[int]) -> dict:
    """Cache the device's x and y (48 kHz) on V and B; return its noise floor: FX
    send RMS during the layout's silence over its di_test RMS."""
    from scipy.signal import resample_poly

    noise = {}
    for o in chosen:
        for s in (h.VALID, h.READ):
            target = vs_path("device", f"take_{o:02d}_{s}_y.npy")
            if not target.exists():
                loop, fx = vs100r.audio(takes[o], name=s)
                target.parent.mkdir(parents=True, exist_ok=True)
                x48 = resample_poly(loop, 1, 4)
                np.save(vs_path("device", f"take_{o:02d}_{s}_x.npy"), x48)
                np.save(target, resample_poly(fx, 1, 4))
        _, quiet = vs100r.audio(takes[o], name="silence")
        quiet = resample_poly(quiet, 1, 4)[h.FS // 10 : -h.FS // 10]
        y_b = np.load(vs_path("device", f"take_{o:02d}_{h.READ}_y.npy"))
        noise[o] = h.nr.rms(quiet) / h.nr.rms(y_b[h.SKIP :])
    return noise


def vs_infer(gd, takes: dict, args) -> dict:
    """Model outputs on V and B, cached; runs without a record are listed absent."""
    runs = {}
    for arm in args.arms:
        for seed in args.seeds:
            run_id = gd.attempt(gd.VS100R[arm].format(seed))
            if not gd.exists(run_id):
                runs[run_id] = {"arm": arm, "seed": seed, "absent": True}
                continue
            record = json.loads((gd.bench.RESULTS_DIR / f"{run_id}.json").read_text())
            entry = {"arm": arm, "seed": seed, "global_step": record["global_step"]}
            model, spent = None, 0.0
            for o, s in ((o, s) for o in args.vs100r_takes for s in (h.VALID, h.READ)):
                path = vs_path(run_id, f"take_{o:02d}_{s}.npy")
                if path.exists():
                    continue
                model = model or gd.load(run_id, "vs100r")
                x = np.load(vs_path("device", f"take_{o:02d}_{s}_x.npy"))
                t0 = time.time()
                y = gd.cd.run(model, x, gd.cd.controls(takes[o]))
                spent += time.time() - t0
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, y)
            runs[run_id] = entry | {"inference_seconds_this_call": spent}
            print(f"{run_id}: inference {spent:.0f} s", flush=True)
    return runs


def vs_by_arm(rows: list[dict], runs: dict) -> dict:
    out = {}
    for arm in VS_ARMS:
        mine = [r for r in rows if runs[r["run_id"]]["arm"] == arm]
        if not mine:
            continue
        out[arm] = {
            "n": len(mine),
            "median_d_rel_B": median([r["d_rel_B"] for r in mine]),
            "median_frozen_x_share_B": {
                n: median([r["frozen_x_share_B"][n] for r in mine])
                for n in mine[0]["frozen_x_share_B"]
            },
            "median_share_B": {
                side: {
                    v: median([r["share_B"][side][v] for r in mine]) for v in VARIANTS
                }
                for side in ("ref_device", "ref_model")
            },
        }
    return out


def stage_vs100r(args, result: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only: a GPU queue is running
    import torch

    from fssr_nam.product import vs100r

    torch.set_num_threads(3)
    sys.path.insert(0, str(h.ROOT / "demo/greybox_deep"))
    import evaluate as gd

    assert gd.cd.DEVICE.type == "cpu"
    start = time.time()
    takes = {t["order"]: t for t in gd.ev.takes()}
    noise = vs_device(vs100r, takes, args.vs100r_takes)
    runs = vs_infer(gd, takes, args)
    jobs = [
        (r, o)
        for r, v in runs.items()
        if not v.get("absent")
        for o in args.vs100r_takes
    ]
    # spawn: the parent holds torch's threads, which a fork would copy
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.jobs, mp_context=spawn) as pool:
        rows = list(pool.map(vs_job, jobs))
    result["vs100r"] = {
        "protocol": "fit di_validation, read di_test (VS100R, 48 kHz by "
        "resample_poly(., 1, 4) as in the evaluation path), SKIP 2400, CPU fp32",
        "takes": {
            o: {
                "setting": takes[o]["setting"],
                "gain_contour": takes[o]["gain_contour"],
                "split": takes[o]["split"],
                "takes_same_pair_in_training": takes[o]["takes_same_pair_in_training"],
                "device_noise_rel": noise[o],
            }
            for o in args.vs100r_takes
        },
        "runs": runs,
        "rows": rows,
        "by_arm": vs_by_arm(rows, runs),
        "seconds": time.time() - start,
    }


def lag(a: np.ndarray, b: np.ndarray, span: int = 32) -> float:
    """Lag of a against b in samples, positive when a is late: the cross-correlation
    peak over -span..span, refined by a parabola through its neighbours."""
    size = 2 * len(a)
    c = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    window = np.concatenate([c[-span:], c[: span + 1]])  # lags -span .. span
    k = int(np.argmax(window))
    lo, mid, hi = window[k - 1], window[k], window[k + 1]
    return float(k - span + 0.5 * (lo - hi) / (lo - 2 * mid + hi))


def stage_lag(args, result: dict) -> None:
    """Timing check of the VS100R reading on V only: lag of each model output against
    the device's FX send (positive: model late)."""
    block = result["vs100r"]
    out = {}
    for run_id, run in block["runs"].items():
        if run.get("absent"):
            continue
        out[run_id] = {
            take: lag(
                np.load(vs_path(run_id, f"take_{int(take):02d}_{h.VALID}.npy")),
                np.load(vs_path("device", f"take_{int(take):02d}_{h.VALID}_y.npy")),
            )
            for take in block["takes"]
        }
    block["lag_V"] = {
        "definition": "cross-correlation peak of y_model against y_device on "
        "di_validation, parabolic refinement, samples at 48 kHz, positive = model late",
        "by_run": out,
        "median_by_arm": {
            arm: median(
                [
                    v
                    for r, t in out.items()
                    if block["runs"][r]["arm"] == arm
                    for v in t.values()
                ]
            )
            for arm in VS_ARMS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage", choices=("nolly", "post", "checks", "vs100r", "lag"), required=True
    )
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--only", nargs="*", help="Nolly pairs REF:EVAL (smoke run)")
    parser.add_argument("--vs100r-takes", type=int, nargs="*", default=[3, 16, 30, 36])
    parser.add_argument("--arms", nargs="*", default=list(VS_ARMS))
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    result = load_result(args.out)
    result["protocol"] = {
        "nolly": "fit di_train (A), select on di_validation (V), read di_test (B), "
        "48 kHz, SKIP 2400; share = RMS(d - q)/RMS(d) after SKIP, no floor",
        "ridge": RIDGE,
        "b_never_fitted": True,
        "y_ref_fed": "every variant reads y_ref: the frozen bound's r_ref is vacuous",
    }
    result["prediction"] = PREDICTION
    result["variants"] = {
        n: {"taps": t, "uses_x": u, "features": desc}
        for n, (_, t, u, desc) in VARIANTS.items()
    }
    {
        "nolly": stage_nolly,
        "post": stage_post,
        "checks": stage_checks,
        "vs100r": stage_vs100r,
        "lag": stage_lag,
    }[args.stage](args, result)
    save(args.out, result)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
