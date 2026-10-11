#!/usr/bin/env python3
"""Skeptic verification of lead 6 (linear readout on frozen learned features).

Stages (CPU only, 2 threads; results merged into verify/lead6_verify.json):
- json:   table values recomputed from the per-unit JSONs (tmp/probe_capacity/lead6/
          units) and compared with the lead JSON and its report text;
- leak:   which sections the arm C networks vs100r48fast_film_s42-44 trained on
          (dataset manifest, layout, bench data module), audio identity of the lead's
          A' / B with the dataset WAVs, (G, C) pairs never trained, checkpoint dates;
- vs:     VS100R take 1, seed 42, centre knobs: features recomputed on CPU, readouts
          refitted with harness.FeatureProbe (fir_pen) and an independent dense least
          squares (static_blocks, static_pen), raw network, ridge re-selection on A',
          linearity identity fit(y26) - fit(y01) = fit(d01_26) on B;
- nolly:  Nolly, seed 42, level-matched, centre knobs (fir_pen y16, fir_pen d01_06) and
          gain-matched knobs (2.5, 5) (static_pen 2.616): values at the lead's ridges,
          then B read at every ridge as a LABELLED DIAGNOSTIC (B used to describe, never
          to select); FIR fed the measured y_ref on Nolly d01_06; H5 fitted on A';
- random: two further untrained random-init draws (seeds 43, 44) on VS100R take 1.

The features call the lead's thin model/hook helpers (net.load_model, net.features),
but every readout here is refitted outside the lead's readout classes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import solve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
import harness as h  # noqa: E402
import lead6_learned_features as l6  # noqa: E402
import lead6_learned_features_net as net  # noqa: E402

CACHE = ROOT.parent / "tmp" / "probe_capacity"
UNITS = CACHE / "lead6" / "units"
WORK = CACHE / "verify" / "lead6"
OUT = HERE / "lead6_verify.json"
LEAD = json.loads((HERE.parent / "lead6_learned_features.json").read_text())
APRIME = slice(4_320_000, 5_760_000)
SPLIT = 1_152_000
GRID = (1e-8, 1e-6, 1e-4, 1e-2, 1.0)
DIAG_GRID = (1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def unit(key: str) -> dict:
    return json.loads((UNITS / f"{key}.json").read_text())


def log(text: str) -> None:
    print(time.strftime("%H:%M:%S"), text, flush=True)


# ---------------------------------------------------------------- readouts (mine)


def dense_static(u_fit: np.ndarray, y_fit, ridges) -> dict:
    """q = c + sum w_k f_k / s_k, s_k = RMS on the fit rows, unpenalised constant,
    ridge * mean diagonal of the feature block: least squares written here (not the
    lead's StaticReadout)."""
    f = np.asarray(u_fit, dtype=np.float64)
    s = np.sqrt(np.mean(f * f, axis=1))
    s = np.maximum(s, 1e-300)
    z = f / s[:, None]
    phi = np.vstack([np.ones(z.shape[1]), z])
    gram = phi @ phi.T
    rhs = phi @ np.asarray(y_fit, dtype=np.float64)
    out = {}
    for ridge in ridges:
        g = gram.copy()
        g[1:, 1:] += ridge * np.mean(np.diag(gram)[1:]) * np.eye(len(z))
        out[ridge] = solve(g, rhs, assume_a="pos")
    return {"scales": s, "theta": out}


def dense_predict(model: dict, ridge: float, u: np.ndarray) -> np.ndarray:
    theta = model["theta"][ridge]
    z = np.asarray(u, dtype=np.float64) / model["scales"][:, None]
    return theta[0] + theta[1:] @ z


def fir_probe(u_fit, y_fit, ridge, taps=256) -> h.FeatureProbe:
    probe = h.FeatureProbe(
        lambda i: [np.asarray(f, dtype=np.float64) for f in i["u"]], taps, ridge
    )
    return probe.fit({"u": u_fit}, np.asarray(y_fit, dtype=np.float64))


def fir_multi_ridge(u_fit, y_fit, ridges, taps=256) -> tuple[h.FeatureProbe, dict]:
    """One Gram (library normal_equations, as FeatureProbe), several ridges."""
    from fssr_nam.product.probes import Probe, normal_equations, regularized

    probe = h.FeatureProbe(
        lambda i: [np.asarray(f, dtype=np.float64) for f in i["u"]], taps, 1.0
    )
    rows = [np.asarray(f, dtype=np.float64) for f in u_fit]
    probe.scales = [max(h.nr.rms(r), 1e-300) for r in rows]
    p = Probe(degree=len(rows), taps=taps, ridge=1.0)
    gram, rhs = normal_equations(
        p,
        [r / s for r, s in zip(rows, probe.scales, strict=True)],
        np.asarray(y_fit, dtype=np.float64),
    )
    thetas = {}
    for ridge in ridges:
        thetas[ridge] = solve(
            regularized(Probe(len(rows), taps, ridge), gram), rhs, assume_a="pos"
        )
    return probe, thetas


def fir_predict(probe: h.FeatureProbe, theta, u) -> np.ndarray:
    probe.theta = theta
    return probe.predict({"u": u})


# ---------------------------------------------------------------- features


def features(model, x, knobs) -> dict:
    t0 = time.time()
    f = net.features(model, np.asarray(x), net.controls(*knobs))
    log(f"forward {len(x)} samples {time.time() - t0:.0f} s")
    return f


def vs_y(take: int, section: str) -> np.ndarray:
    return np.asarray(net.vs100r_section(take, section)[1])


# ---------------------------------------------------------------- stage json


def stage_json() -> dict:
    out = {}
    takes = net.vs100r_takes()
    rows = {r["take"]: r for r in LEAD["vs100r"]["tables"]["output_capacity"]}
    # recompute mean/min/max of fir_pen and static_blocks from the units
    rec = {}
    for t in l6.OUTPUT_TAKES:
        knobs = tuple(takes[t]["gain_contour"])
        for mode, k in (("center", (5.0, 5.0)), ("own", knobs)):
            if mode == "own" and k == (5.0, 5.0):
                continue
            tag = f"{k[0]:g}-{k[1]:g}"
            for ro in ("fir_pen", "static_blocks", "fir_out"):
                vals = [
                    unit(f"vs100r_s{s}_x{t:02d}_k{tag}_raw")["readouts"][ro]["targets"][
                        f"y{t:02d}"
                    ]["B"]
                    for s in (42, 43, 44)
                ]
                lead_entry = rows[t][f"learned_{mode}"][ro]["B"]
                rec[f"y{t:02d}/{mode}/{ro}"] = {
                    "recomputed_mean": float(np.mean(vals)),
                    "min": min(vals),
                    "max": max(vals),
                    "lead_mean": lead_entry["mean"],
                    "abs_gap": abs(float(np.mean(vals)) - lead_entry["mean"]),
                }
    out["vs100r_tables_from_units"] = rec
    # ranges the report text quotes
    cc = [rec[f"y{t:02d}/center/fir_pen"]["recomputed_mean"] for t in (1, 39, 37)]
    g_fp = [rec[f"y{t:02d}/center/fir_pen"]["recomputed_mean"] for t in (26, 9, 16, 30)]
    g_sb = [
        rec[f"y{t:02d}/center/static_blocks"]["recomputed_mean"]
        for t in (26, 9, 16, 30)
    ]
    own = {
        ro: [
            rec[f"y{t:02d}/own/{ro}"]
            for t in l6.OUTPUT_TAKES
            if f"y{t:02d}/own/{ro}" in rec
        ]
        for ro in ("fir_pen", "static_blocks", "fir_out")
    }
    frozen = []
    for t in l6.OUTPUT_TAKES:
        for p, fits in rows[t]["frozen"].items():
            frozen += [(t, p, fit, v) for fit, v in fits.items()]
    out["text_ranges"] = {
        "center_fir_pen_takes_1_39_37": [min(cc), max(cc)],
        "center_fir_pen_G2.5_G10_takes_26_9_16_30": [min(g_fp), max(g_fp)],
        "center_static_blocks_G2.5_G10": [min(g_sb), max(g_sb)],
        "center_fir_pen_G2.5_G10_seed_range": [
            min(rec[f"y{t:02d}/center/fir_pen"]["min"] for t in (26, 9, 16, 30)),
            max(rec[f"y{t:02d}/center/fir_pen"]["max"] for t in (26, 9, 16, 30)),
        ],
        "own_max_mean": {
            ro: max(e["recomputed_mean"] for e in v) for ro, v in own.items()
        },
        "own_max_seed": {ro: max(e["max"] for e in v) for ro, v in own.items()},
        "frozen_A_range": [
            min(v for *_, f, v in frozen if f == "A"),
            max(v for *_, f, v in frozen if f == "A"),
        ],
        "frozen_Aprime_range": [
            min(v for *_, f, v in frozen if f == "Aprime"),
            max(v for *_, f, v in frozen if f == "Aprime"),
        ],
    }
    # Nolly gains
    base = json.loads((HERE.parent / "baseline.json").read_text())["probes"]
    nol = {}
    for gain, take in h.GAIN_TAKES.items():
        vals = [
            unit(f"nolly_s{s}_nolly_k5-5_matched")["readouts"]["fir_pen"]["targets"][
                f"y{take:02d}"
            ]["B"]
            for s in (42, 43, 44)
        ]
        lead_row = next(
            r
            for r in LEAD["nolly"]["tables"]["output_capacity_by_gain"]
            if r["take"] == take
        )
        raw42 = unit("nolly_s42_nolly_k5-5_raw")["readouts"]["fir_pen"]["targets"][
            f"y{take:02d}"
        ]["B"]
        nol[f"G{gain:g}"] = {
            "fir_pen_matched_mean": float(np.mean(vals)),
            "fir_pen_min_max": [min(vals), max(vals)],
            "lead_mean": lead_row["learned_center_matched"]["fir_pen"]["B"]["mean"],
            "fir_pen_s42_raw_level": raw42,
            "H5_A_baseline": base["hammerstein_5"]["capacity_by_take"][str(take)][
                "value"
            ],
            "VL3_A_baseline": base["volterra_laguerre_3"]["capacity_by_take"][
                str(take)
            ]["value"],
        }
        nol[f"G{gain:g}"]["H5_minus_fir_pen_mean"] = (
            nol[f"G{gain:g}"]["H5_A_baseline"]
            - nol[f"G{gain:g}"]["fir_pen_matched_mean"]
        )
    out["nolly_gains"] = nol
    # ridge choices on the grid edges, per readout and domain
    edges = {}
    for p in sorted(UNITS.glob("*.json")):
        u = json.loads(p.read_text())
        if "readouts" not in u:
            continue
        for ro, e in u["readouts"].items():
            key = f"{u['domain']}/{ro}"
            d = edges.setdefault(key, {})
            d[f"{e['ridge']:g}"] = d.get(f"{e['ridge']:g}", 0) + 1
    out["ridge_choices_by_domain"] = edges
    # Nolly differences: learned vs FIR fed y_ref
    diffs = []
    for r in LEAD["nolly"]["tables"]["difference_share_one_knob"]:
        fp = r["learned_center_matched"]["fir_pen"]["B"]["mean"]
        yref = r["output_referenced_fir_yref_fed"]["A"]
        diffs.append((r["ref"], r["eval"], fp, yref, fp > yref))
    out["nolly_diff_yref_beats_learned_all"] = all(d[4] for d in diffs)
    out["nolly_diff_ranges"] = {
        "fir_pen_mean": [min(d[2] for d in diffs), max(d[2] for d in diffs)],
        "yref_A": [min(d[3] for d in diffs), max(d[3] for d in diffs)],
    }
    return out


# ---------------------------------------------------------------- stage leak


def stage_leak() -> dict:
    import soundfile as sf

    out = {}
    layout = json.loads((ROOT / "demo/capture/valvestate/layout.json").read_text())
    sec = {s["section"]: s for s in layout["sections"]}
    trainval = (
        sec["di_train"]["start"],
        sec["di_validation"]["start"] + sec["di_validation"]["samples"],
    )
    test = (
        sec["di_test"]["start"],
        sec["di_test"]["start"] + sec["di_test"]["samples"],
    )
    out["layout_192k"] = {"trainval_span": trainval, "di_test_span": test}
    out["di_test_disjoint_from_trainval_span"] = test[0] >= trainval[1]
    data = ROOT / "datasets/raw/internal/valvestate_vs100r/explore_48k"
    manifest = json.loads((data / "manifest.json").read_text())
    by_split: dict[str, list] = {}
    for m in manifest:
        by_split.setdefault(m["split"], []).append(m["take"])
    out["manifest_splits"] = {k: sorted(v) for k, v in by_split.items()}
    used = sorted(set(l6.OUTPUT_TAKES) | set(l6.DIFF_EVALS) | {l6.REF})
    out["takes_used"] = {
        str(t): {
            "trainval_contains_di_train_and_di_validation": t in by_split["trainval"],
            "di_test_in_split": "test" if t in by_split["test"] else "test_seen",
        }
        for t in used
    }
    # (G, C) pairs never trained
    with (ROOT / "demo/capture/valvestate/settings.csv").open() as stream:
        rows = [
            r
            for r in csv.DictReader(stream)
            if r["channel"] == "OD2" and r["power_dimension"] == "off"
        ]
    pair = {int(r["order"]): (float(r["G"]), float(r["C"])) for r in rows}
    trained = [pair[int(r["order"])] for r in rows if r["split"] == "train"]
    out["test_takes_pair_count_in_training"] = {
        str(int(r["order"])): {
            "gc": pair[int(r["order"])],
            "n": trained.count(pair[int(r["order"])]),
        }
        for r in rows
        if r["split"] == "test"
    }
    gains_tr = sorted({g for g, _ in trained})
    contours_tr = sorted({c for _, c in trained})
    out["take16_knob_values_individually_trained"] = {
        "gain_2.5_trained": 2.5 in gains_tr,
        "contour_7.5_trained": 7.5 in contours_tr,
    }
    # audio identity: lead's A' / B (48 kHz cache) vs the dataset WAVs
    ident = {}
    for t in (1, 26, 9, 16):
        x_b = np.asarray(net.vs100r_section(t, h.READ)[0])
        split = "test" if t in by_split["test"] else "test_seen"
        d_te = sf.read(
            data / "dry" / split / f"take{t:02d}.input.wav", dtype="float64"
        )[0]
        entry = {
            "B_vs_dataset_di_test_dry_rel_rms": h.nr.rms(d_te - x_b) / h.nr.rms(x_b),
            "B_len": len(x_b),
            "dataset_di_test_len": len(d_te),
        }
        if t in by_split["trainval"]:
            d_tv = sf.read(
                data / "dry" / "trainval" / f"take{t:02d}.input.wav", dtype="float64"
            )[0]
            w_tv = sf.read(
                data
                / "wet"
                / "trainval"
                / next(m["setting"] for m in manifest if m["take"] == t)
                / f"{next(m['setting'] for m in manifest if m['take'] == t)}.take{t:02d}.target.wav",
                dtype="float64",
            )[0]
            x_a = np.asarray(net.vs100r_section(t, h.FIT)[0])
            y_a = np.asarray(net.vs100r_section(t, h.FIT)[1])
            mid = slice(APRIME.start, APRIME.stop - 2000)  # away from the A/V seam
            entry["Aprime_x_vs_trainval_rel_rms"] = h.nr.rms(
                d_tv[mid] - x_a[mid]
            ) / h.nr.rms(x_a[mid])
            entry["Aprime_y_vs_trainval_target_rel_rms"] = h.nr.rms(
                w_tv[mid] - y_a[mid]
            ) / h.nr.rms(y_a[mid])
            # does B audio appear anywhere in trainval? normalised cross-correlation
            # peak of a 1 s chunk of B input against the whole trainval input
            from scipy.signal import fftconvolve

            chunk = x_b[100_000:148_000]
            corr = fftconvolve(d_tv, chunk[::-1], mode="valid")
            c2 = np.concatenate(([0.0], np.cumsum(d_tv**2)))
            window = c2[len(chunk) :] - c2[: len(c2) - len(chunk)]
            energy = np.sqrt(np.maximum(window, 0.0)) * np.linalg.norm(chunk)
            ncc = corr / np.maximum(energy, 1e-30)
            entry["max_ncc_of_1s_B_chunk_in_trainval_input"] = float(
                np.max(np.abs(ncc))
            )
            entry["trainval_len"] = len(d_tv)
        ident[str(t)] = entry
    out["audio_identity"] = ident
    # checkpoint and dataset dates vs run records
    runs = {}
    for line in (ROOT / "demo/RUNS.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec.get("run_id", "").startswith("vs100r48fast_film_s4") and "commit" in rec:
            runs[rec["run_id"]] = rec
    dates = {}
    ds_mtime = os.path.getmtime(data / "manifest.json")
    for rid, rec in sorted(runs.items()):
        ck = ROOT / "demo/runs" / f"nablafx_{rid}" / "checkpoints" / "last.ckpt"
        stamp = datetime.fromisoformat(rec["timestamp"]).timestamp()
        dates[rid] = {
            "device": rec["device"],
            "held_out_settings": rec["held_out_settings"],
            "global_step": rec["global_step"],
            "max_steps": rec["max_steps"],
            "no_early_stopping": rec["no_early_stopping"],
            "val_indices": len(rec["val_indices"]),
            "last_ckpt_before_record": os.path.getmtime(ck) <= stamp,
            "last_ckpt_minutes_before_record": (stamp - os.path.getmtime(ck)) / 60,
            "dataset_built_before_run": ds_mtime < stamp,
            "test_last_esr_take16_setting": rec["test_last_by_setting"]
            .get("G025_C075_B025_M075_T075", {})
            .get("esr"),
        }
    out["runs"] = dates
    out["segments"] = {
        "trainval_seconds_per_take": 150,
        "segments_total": 31 * 150 * 48000 // 144000,
        "expected_val_10pct": round(0.1 * 31 * 150 * 48000 // 144000),
    }
    raw16 = {
        s: unit(f"vs100r_s{s}_x16_k2.5-7.5_raw")["network_raw"]["y16"]
        for s in (42, 43, 44)
    }
    out["take16_raw_network_vs_bench_esr"] = {
        s: {
            "lead_raw_ratio": raw16[s],
            "bench_test_last_sqrt_esr": float(
                np.sqrt(
                    dates[f"vs100r48fast_film_s{s}"]["test_last_esr_take16_setting"]
                )
            ),
        }
        for s in (42, 43, 44)
    }
    return out


# ---------------------------------------------------------------- stage vs


def stage_vs() -> dict:
    out = {}
    model = net.load_model(42)
    xa = np.asarray(net.vs100r_section(1, h.FIT)[0])[APRIME]
    xb = np.asarray(net.vs100r_section(1, h.READ)[0])
    fa, fb = features(model, xa, (5.0, 5.0)), features(model, xb, (5.0, 5.0))
    y01a, y01b = vs_y(1, h.FIT)[APRIME], vs_y(1, h.READ)
    y26a, y26b = vs_y(26, h.FIT)[APRIME], vs_y(26, h.READ)
    d_a, d_b = y26a - y01a, y26b - y01b
    lead = unit("vs100r_s42_x01_k5-5_raw")
    # pen really is the penultimate layer: out = tanh(conv(pen))
    head = model.processor.output_net[3]
    w = head.weight.detach().numpy()[0, :, 0].astype(np.float64)
    b = float(head.bias.detach().numpy()[0])
    rebuilt = np.tanh(w @ fb["pen"][:, :200_000].astype(np.float64) + b)
    out["pen_is_penultimate_max_abs_gap"] = float(
        np.max(np.abs(rebuilt - fb["out"][0, :200_000]))
    )
    out["raw_network_y01"] = {
        "recomputed": h.ratio(y01b, fb["out"][0].astype(np.float64)),
        "lead": lead["network_raw"]["y01"],
    }
    # fir_pen, harness.FeatureProbe, ridge chosen by the lead (1e-8): y01, y26, d01_26
    r = lead["readouts"]["fir_pen"]["ridge"]
    t0 = time.time()
    q = {}
    for name, ya in (("y01", y01a), ("y26", y26a), ("d01_26", d_a)):
        q[name] = fir_probe(fa["pen"], ya, r).predict({"u": fb["pen"]})
    log(f"fir_pen three fits {time.time() - t0:.0f} s")
    out["fir_pen"] = {
        "ridge": r,
        "y01_B": {
            "recomputed": h.ratio(y01b, q["y01"]),
            "lead": lead["readouts"]["fir_pen"]["targets"]["y01"]["B"],
        },
        "d01_26_B": {
            "recomputed": h.ratio(d_b, q["d01_26"], 0.0),
            "lead": lead["readouts"]["fir_pen"]["targets"]["d01_26"]["B"],
        },
        "linearity_gap_B": h.nr.rms((q["y26"] - q["y01"] - q["d01_26"])[h.SKIP :])
        / h.nr.rms(q["d01_26"][h.SKIP :]),
        "lead_linearity_gap": lead["readouts"]["fir_pen"]["linearity_gap"],
    }
    # same identity at ridge 1e-4 (conditioning vs linearity)
    q4 = {
        name: fir_probe(fa["pen"], ya, 1e-4).predict({"u": fb["pen"]})
        for name, ya in (("y01", y01a), ("y26", y26a), ("d01_26", d_a))
    }
    out["fir_pen"]["linearity_gap_B_ridge_1e-4"] = h.nr.rms(
        (q4["y26"] - q4["y01"] - q4["d01_26"])[h.SKIP :]
    ) / h.nr.rms(q4["d01_26"][h.SKIP :])
    out["fir_pen"]["y01_B_ridge_1e-4"] = h.ratio(y01b, q4["y01"])
    # fir_pen ridge re-selection on A' only (A1 fit, A2 score)
    probe, thetas = fir_multi_ridge(fa["pen"][:, :SPLIT], y01a[:SPLIT], GRID)
    curve = {}
    for ridge, th in thetas.items():
        pred = fir_predict(probe, th, fa["pen"])
        curve[f"{ridge:g}"] = h.ratio(y01a[SPLIT:], pred[SPLIT:])
    out["fir_pen"]["ridge_curve_A2_recomputed"] = curve
    out["fir_pen"]["ridge_curve_A2_lead"] = lead["readouts"]["fir_pen"][
        "ridge_curve_A2"
    ]
    # static_blocks and static_pen: dense least squares written here
    for ro, rep in (("static_blocks", "blocks"), ("static_pen", "pen")):
        ridge = lead["readouts"][ro]["ridge"]
        res = {"ridge": ridge}
        for name, ya, yb, fl in (
            ("y01", y01a, y01b, h.FLOOR),
            ("d01_26", d_a, d_b, 0.0),
            ("y26", y26a, y26b, h.FLOOR),
        ):
            m = dense_static(fa[rep], ya, (ridge,))
            res[f"{name}_pred"] = dense_predict(m, ridge, fb[rep])
            res[f"{name}_B"] = {
                "recomputed": h.ratio(yb, res[f"{name}_pred"], fl),
                "lead": lead["readouts"][ro]["targets"][name]["B"],
            }
        res["linearity_gap_B"] = h.nr.rms(
            (res["y26_pred"] - res["y01_pred"] - res["d01_26_pred"])[h.SKIP :]
        ) / h.nr.rms(res["d01_26_pred"][h.SKIP :])
        for k in ("y01_pred", "d01_26_pred", "y26_pred"):
            del res[k]
        m1 = dense_static(fa[rep][:, :SPLIT], y01a[:SPLIT], GRID)
        res["ridge_curve_A2_recomputed"] = {
            f"{g:g}": h.ratio(y01a[SPLIT:], dense_predict(m1, g, fa[rep])[SPLIT:])
            for g in GRID
        }
        res["ridge_curve_A2_lead"] = lead["readouts"][ro]["ridge_curve_A2"]
        out[ro] = res
    # fir_out (1 channel)
    rr = lead["readouts"]["fir_out"]["ridge"]
    qo = fir_probe(fa["out"], y01a, rr).predict({"u": fb["out"]})
    out["fir_out_y01_B"] = {
        "recomputed": h.ratio(y01b, qo),
        "lead": lead["readouts"]["fir_out"]["targets"]["y01"]["B"],
        "ridge": rr,
    }
    # integer lag of the raw output against y on B (the "1-sample delay")
    o = fb["out"][0].astype(np.float64)
    n = len(o)
    lags = {
        g: h.ratio(y01b[h.SKIP + 5 : n - 5], o[h.SKIP + 5 - g : n - 5 - g])
        for g in range(-3, 4)
    }
    out["raw_network_y01_vs_integer_shift"] = {str(g): v for g, v in lags.items()}
    return out


# ---------------------------------------------------------------- stage nolly


def stage_nolly() -> dict:
    out = {}
    x = h.load("x", (h.FIT, h.READ))
    vs = np.asarray(net.vs100r_section(1, h.FIT)[0])[APRIME]
    gain = h.nr.rms(vs) / h.nr.rms(np.asarray(x[h.FIT][APRIME]))
    out["level_gain"] = {
        "recomputed": gain,
        "lead": LEAD["protocol"]["nolly_level_gain"],
    }
    xa, xb = gain * np.asarray(x[h.FIT][APRIME]), gain * np.asarray(x[h.READ])
    out["input_peaks"] = {
        "Aprime_peak": float(np.max(np.abs(xa))),
        "B_peak": float(np.max(np.abs(xb))),
        "Aprime_rms": h.nr.rms(xa),
        "B_rms": h.nr.rms(xb),
        "B_fraction_above_Aprime_peak": float(np.mean(np.abs(xb) > np.max(np.abs(xa)))),
    }
    y16 = h.load(16, (h.FIT, h.READ))
    y16a, y16b = np.asarray(y16[h.FIT][APRIME]), np.asarray(y16[h.READ])
    model = net.load_model(42)
    # centre knobs: fir_pen y16 and d01_06
    fa, fb = features(model, xa, (5.0, 5.0)), features(model, xb, (5.0, 5.0))
    lead_c = unit("nolly_s42_nolly_k5-5_matched")
    r = lead_c["readouts"]["fir_pen"]["ridge"]
    q = fir_probe(fa["pen"], y16a, r).predict({"u": fb["pen"]})
    d06a = np.asarray(h.difference(1, 6, h.FIT))[APRIME]
    d06b = np.asarray(h.difference(1, 6, h.READ))
    qd = fir_probe(fa["pen"], d06a, r).predict({"u": fb["pen"]})
    out["center_fir_pen"] = {
        "ridge": r,
        "y16_B": {
            "recomputed": h.ratio(y16b, q),
            "lead": lead_c["readouts"]["fir_pen"]["targets"]["y16"]["B"],
        },
        "d01_06_B": {
            "recomputed": h.ratio(d06b, qd, 0.0),
            "lead": lead_c["readouts"]["fir_pen"]["targets"]["d01_06"]["B"],
        },
    }
    # centre static_blocks y16 (dense LS)
    rb = lead_c["readouts"]["static_blocks"]["ridge"]
    mb = dense_static(fa["blocks"], y16a, (rb,))
    out["center_static_blocks_y16_B"] = {
        "recomputed": h.ratio(y16b, dense_predict(mb, rb, fb["blocks"])),
        "lead": lead_c["readouts"]["static_blocks"]["targets"]["y16"]["B"],
    }
    del fa, fb
    # H5 on A' (frozen probe), FIR fed the measured y_ref (A, ridge 1e-4)
    h5 = h.FrozenProbe("hammerstein_5").fit({"x": np.asarray(x[h.FIT][APRIME])}, y16a)
    out["H5_Aprime_y16_B"] = {
        "recomputed": h.ratio(y16b, h5.predict({"x": np.asarray(x[h.READ])})),
        "lead": unit("nolly_frozen_y16")["hammerstein_5"]["Aprime"]["B"],
    }
    yref = h.difference_share(
        h.FeatureProbe(lambda i: [np.asarray(i["y_ref"])], 256, 1e-4), 1, 6
    )
    out["yref_fir_d01_06_fit_A"] = {
        "recomputed": yref,
        "lead": unit("nolly_yref_d01_06")["A"]["B"],
    }
    # gain-matched knobs (2.5, 5): static_pen 2.616 and its ridge sensitivity
    lead_g = unit("nolly_s42_nolly_k2.5-5_matched")
    ga, gb = features(model, xa, (2.5, 5.0)), features(model, xb, (2.5, 5.0))
    res = {}
    for ro, rep in (("static_pen", "pen"), ("static_blocks", "blocks")):
        ridge = lead_g["readouts"][ro]["ridge"]
        m_full = dense_static(ga[rep], y16a, DIAG_GRID)
        m_a1 = dense_static(ga[rep][:, :SPLIT], y16a[:SPLIT], GRID)
        a2 = {
            f"{g:g}": h.ratio(y16a[SPLIT:], dense_predict(m_a1, g, ga[rep])[SPLIT:])
            for g in GRID
        }
        pred = dense_predict(m_full, ridge, gb[rep])
        res[ro] = {
            "ridge_chosen_by_lead": ridge,
            "ridge_choice_recomputed_on_A2": min(a2, key=a2.get),
            "A2_curve_recomputed": a2,
            "A2_curve_lead": lead_g["readouts"][ro]["ridge_curve_A2"],
            "y16_B": {
                "recomputed": h.ratio(y16b, pred),
                "lead": lead_g["readouts"][ro]["targets"]["y16"]["B"],
            },
            "ORACLE_DIAGNOSTIC_B_by_ridge (B used to describe, not to select)": {
                f"{g:g}": h.ratio(y16b, dense_predict(m_full, g, gb[rep]))
                for g in DIAG_GRID
            },
            "pred_peak_B_over_Aprime": float(
                np.max(np.abs(pred))
                / np.max(np.abs(dense_predict(m_full, ridge, ga[rep])))
            ),
            "y16_peak_B_over_Aprime": float(
                np.max(np.abs(y16b)) / np.max(np.abs(y16a))
            ),
        }
        if rep == "pen":
            # where the error sits: share of residual energy in the worst 1 % of samples
            e = (y16b - pred)[h.SKIP :] ** 2
            top = np.sort(e)[::-1][: len(e) // 100]
            res[ro]["residual_energy_share_top_1pct_samples"] = float(
                top.sum() / e.sum()
            )
    out["gain_matched_k2.5"] = res
    return out


# ---------------------------------------------------------------- stage random


def stage_random() -> dict:
    out = {}
    xa = np.asarray(net.vs100r_section(1, h.FIT)[0])[APRIME]
    xb = np.asarray(net.vs100r_section(1, h.READ)[0])
    y01a, y01b = vs_y(1, h.FIT)[APRIME], vs_y(1, h.READ)
    lead = unit("vs100r_rand42_x01_k5-5_raw")
    for seed in (42, 43, 44):  # untrained draws (records exist for 42-44 only)
        model = net.load_model(seed, trained=False)
        fa, fb = features(model, xa, (5.0, 5.0)), features(model, xb, (5.0, 5.0))
        entry = {}
        # static_blocks: A2-selected ridge on the grid (as the lead), dense LS
        m1 = dense_static(fa["blocks"][:, :SPLIT], y01a[:SPLIT], GRID)
        a2 = {
            g: h.ratio(y01a[SPLIT:], dense_predict(m1, g, fa["blocks"])[SPLIT:])
            for g in GRID
        }
        g_best = min(a2, key=a2.get)
        m = dense_static(fa["blocks"], y01a, (g_best,))
        entry["static_blocks_y01_B"] = h.ratio(
            y01b, dense_predict(m, g_best, fb["blocks"])
        )
        entry["static_blocks_ridge"] = g_best
        # fir_pen at the A2-selected ridge
        probe, thetas = fir_multi_ridge(fa["pen"][:, :SPLIT], y01a[:SPLIT], GRID)
        a2p = {
            g: h.ratio(y01a[SPLIT:], fir_predict(probe, th, fa["pen"])[SPLIT:])
            for g, th in thetas.items()
        }
        gp = min(a2p, key=a2p.get)
        entry["fir_pen_y01_B"] = h.ratio(
            y01b, fir_probe(fa["pen"], y01a, gp).predict({"u": fb["pen"]})
        )
        entry["fir_pen_ridge"] = gp
        entry["blocks_rms_per_channel_range"] = [
            float(
                np.min(np.sqrt(np.mean(fa["blocks"].astype(np.float64) ** 2, axis=1)))
            ),
            float(
                np.max(np.sqrt(np.mean(fa["blocks"].astype(np.float64) ** 2, axis=1)))
            ),
        ]
        entry["blocks_effective_rank_99pct"] = int(
            np.searchsorted(
                np.cumsum(
                    np.sort(
                        np.linalg.eigvalsh(
                            np.cov(fa["blocks"][:, ::10].astype(np.float64))
                        )
                    )[::-1]
                )
                / np.trace(np.cov(fa["blocks"][:, ::10].astype(np.float64))),
                0.99,
            )
            + 1
        )
        if seed == 42:
            entry["lead"] = {
                "static_blocks": lead["readouts"]["static_blocks"]["targets"]["y01"][
                    "B"
                ],
                "fir_pen": lead["readouts"]["fir_pen"]["targets"]["y01"]["B"],
            }
        out[f"rand{seed}"] = entry
        log(f"random {seed} {entry}")
    # trained s42 effective rank for comparison
    model = net.load_model(42)
    fa = features(model, xa, (5.0, 5.0))
    c = np.cov(fa["blocks"][:, ::10].astype(np.float64))
    ev = np.sort(np.linalg.eigvalsh(c))[::-1]
    out["trained_s42_blocks_effective_rank_99pct"] = int(
        np.searchsorted(np.cumsum(ev) / ev.sum(), 0.99) + 1
    )
    return out


# ---------------------------------------------------------------- stage similarity


def stage_similarity() -> dict:
    """How different the 'test setting' takes are from trained takes on B (target
    only, descriptive), and the B minus in-sample A' gap at matched gain."""
    out = {}
    for a, b in ((26, 16), (9, 30), (1, 39), (1, 37), (1, 26), (1, 9)):
        xa, ya = (np.asarray(v) for v in net.vs100r_section(a, h.READ))
        xb, yb = (np.asarray(v) for v in net.vs100r_section(b, h.READ))
        out[f"{a}_vs_{b}"] = {
            "x_rel_rms_lag0": h.nr.rms(xb - xa) / h.nr.rms(xa),
            "y_rel_rms_lag0_after_skip": h.nr.rms((yb - ya)[h.SKIP :])
            / h.nr.rms(ya[h.SKIP :]),
        }
    gaps = {}
    for t, k in (
        (26, "5-5"),
        (16, "5-5"),
        (9, "5-5"),
        (30, "5-5"),
        (1, "5-5"),
        (37, "5-5"),
        (39, "5-5"),
        (26, "2.5-5"),
        (16, "2.5-7.5"),
        (9, "10-5"),
        (30, "10-5"),
    ):
        vals = []
        for s in (42, 43, 44):
            e = unit(f"vs100r_s{s}_x{t:02d}_k{k}_raw")["readouts"]["fir_pen"][
                "targets"
            ][f"y{t:02d}"]
            vals.append(e["B"] - e["A_in_sample"])
        gaps[f"y{t:02d}_k{k}"] = {"per_seed": vals, "mean": float(np.mean(vals))}
    out["fir_pen_B_minus_Aprime_in_sample"] = gaps
    return out


# ---------------------------------------------------------------- driver


STAGES = {
    "json": stage_json,
    "leak": stage_leak,
    "vs": stage_vs,
    "nolly": stage_nolly,
    "random": stage_random,
    "similarity": stage_similarity,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stages", nargs="+", choices=tuple(STAGES))
    args = parser.parse_args()
    torch.set_num_threads(2)
    WORK.mkdir(parents=True, exist_ok=True)
    result = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in args.stages:
        t0 = time.time()
        result[name] = STAGES[name]()
        result[name]["seconds"] = time.time() - t0
        OUT.write_text(json.dumps(h.nr.finite(result), indent=1, default=float) + "\n")
        log(f"stage {name} {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
