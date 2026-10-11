#!/usr/bin/env python3
"""Skeptic verification of lead 4 (BLA ceiling from the multisines vs DI capacity).

Independent recomputations (own code, harness cache, lead functions not called except
in the explicit `lead_linearity` check of the lead's own RowFit):
- multisine decomposition (BLA share with variance correction, LOO in time domain,
  noise between periods 2 and 3, odd/even levels with the frozen reading's lines);
- level-band FIR 256 (frozen `fir` spec: relative ridge 1e-2, constant) fitted on A
  rows whose 50 ms window (grid from 0) lies in a band, Gram and X^T y accumulated
  directly from an explicit design (no FFT correlation), read on B windows (grid
  from SKIP); level-scheduled FIR non-causal / causal (previous window) and a causal
  running-level variant (trailing 50 ms per sample, x only);
- ridge / basis attribution of the proposed rule's E: FIR 256 at H5's ridge 1e-4
  (own Gram) and a linear-only Laguerre basis at VL3's spec (library fit, A only);
- block bootstrap (1 s blocks) of E on V halves and on B (read only, no selection);
- statistics from the lead JSON: Spearman with p-values, jackknife, bootstrap, cab-on
  takes removed, excluded takes added, V-weighted global prediction.
B never enters a fit here; the B bootstrap only measures the read's sampling spread.

    uv run python diagnosis/probe_capacity/verify/lead4_verify.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from scipy.linalg import solve
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness as h  # noqa: E402

from fssr_nam.product.probes import (  # noqa: E402
    Probe,
    VolterraLaguerre,
    fit_probe,
    probe_contributions,
)

LEAD = json.loads((HERE.parent / "lead4_bla_ceiling.json").read_text())
SCRATCH = h.CACHE / "verify" / "lead4"
OUT = HERE / "lead4_verify.json"
P, W, SKIP, TAPS = 48_000, 2_400, h.SKIP, 256
LEVELS = (-30, -24, -18)
EDGES = (-33.0, -27.0, -21.0, -15.0)  # 5 schedule bands: <-33, 3 matched, >=-15
H2 = dict(h.GAIN_TAKES)  # {0: 30, 2.5: 16, 5: 20, 7.5: 6, 10: 7}
DI_TAKES = (4, 33, 16, 20, 7, 6, 30)  # band FIR takes (examples + H2 ladder)
LINE = np.arange(P // 2 + 1)
EXC = (LINE >= 21) & (LINE <= 19_999) & (LINE % 2 == 1) & (LINE % 8 != 3)
ODD = (LINE >= 21) & (LINE <= 19_999) & (LINE % 8 == 3)
EVEN = (LINE >= 22) & (LINE <= 19_998) & (LINE % 2 == 0)
RNG = np.random.default_rng(0)


def arr(take, name: str) -> np.ndarray:
    return np.asarray(h.load(take, (name,))[name], dtype=float)


# ---------------------------------------------------------------- multisine


def multisine(take: int, lvl: int) -> dict:
    """Own BLA decomposition on periods 2-3 of the 4 realizations."""
    xs = [arr("x", f"multisine_{lvl}dBFS_rms_r{r}") for r in (1, 2, 3, 4)]
    ys = [arr(take, f"multisine_{lvl}dBFS_rms_r{r}") for r in (1, 2, 3, 4)]
    xp = [x[P:].reshape(2, P) for x in xs]
    yp = [y[P:].reshape(2, P) for y in ys]
    total = np.mean([np.mean(v**2) for v in yp])  # time domain, periods 2-3
    U = [np.fft.rfft(v, axis=1).mean(0) for v in xp]
    Y = [np.fft.rfft(v, axis=1).mean(0) for v in yp]
    G = np.array([y[EXC] / u[EXC] for u, y in zip(U, Y)])
    m = len(G)
    g = G.mean(0)
    var = np.sum(np.abs(G - g) ** 2, 0) / (m - 1)
    bla_naive, bla_corr = [], []
    for u in U:
        z = np.zeros(P // 2 + 1, complex)
        z[EXC] = g * u[EXC]
        bla_naive.append(np.mean(np.fft.irfft(z, P) ** 2))
        bla_corr.append(2 * np.sum((np.abs(g) ** 2 - var / m) * np.abs(u[EXC]) ** 2) / P**2)
    loo = []
    for k in range(m):
        gk = np.delete(G, k, 0).mean(0)
        for p in range(2):
            z = np.zeros(P // 2 + 1, complex)
            z[EXC] = gk * np.fft.rfft(xp[k][p])[EXC]
            loo.append(np.mean((yp[k][p] - np.fft.irfft(z, P)) ** 2))
    noise = np.mean([np.mean((v[0] - v[1]) ** 2) / 2 for v in yp])
    odd, even = [], []
    for y in Y:
        pw = np.abs(y) ** 2
        odd.append(10 * np.log10(pw[ODD].mean() / pw[EXC].mean()))
        even.append(10 * np.log10(pw[EVEN].mean() / pw[EXC].mean()))
    return {
        "D": float(1 - np.mean(bla_corr) / total),
        "D_naive_no_variance_correction": float(1 - np.mean(bla_naive) / total),
        "D_loo_time_domain": float(np.mean(loo) / total),
        "noise_share": float(noise / total),
        "odd_db": float(np.mean(odd)),
        "even_db": float(np.mean(even)),
    }


def part_multisine() -> dict:
    out = {"recomputed": {}, "noise_share_all": {}}
    worst = {"D": 0.0, "D_loo": 0.0, "odd_db_vs_lead": 0.0, "odd_db_vs_frozen": 0.0,
             "even_db_vs_frozen": 0.0}
    for t in DI_TAKES:
        for lvl in LEVELS:
            mine = multisine(t, lvl)
            lead = LEAD["takes"][str(t)]["multisine"][str(lvl)]
            frozen = h.READING["multisine"].get(str(t), {}).get(str(lvl))
            entry = {
                "mine": mine,
                "lead": {k: lead[k] for k in ("D", "D_loo", "noise_share", "odd_db", "even_db")},
            }
            if frozen:
                entry["frozen_192k"] = {k: frozen[k] for k in ("odd_db", "even_db")}
                worst["odd_db_vs_frozen"] = max(worst["odd_db_vs_frozen"],
                                                abs(mine["odd_db"] - frozen["odd_db"]))
                worst["even_db_vs_frozen"] = max(worst["even_db_vs_frozen"],
                                                 abs(mine["even_db"] - frozen["even_db"]))
            worst["D"] = max(worst["D"], abs(mine["D"] - lead["D"]))
            worst["D_loo"] = max(worst["D_loo"], abs(mine["D_loo_time_domain"] - lead["D_loo"]))
            worst["odd_db_vs_lead"] = max(worst["odd_db_vs_lead"], abs(mine["odd_db"] - lead["odd_db"]))
            out["recomputed"][f"{t}|{lvl}"] = entry
    out["max_abs_gap"] = worst
    usable_max = all_max = 0.0
    for t in sorted(h.ROWS):
        for lvl in LEVELS:
            ys = [arr(t, f"multisine_{lvl}dBFS_rms_r{r}")[P:].reshape(2, P) for r in (1, 2, 3, 4)]
            total = np.mean([np.mean(v**2) for v in ys])
            s = float(np.mean([np.mean((v[0] - v[1]) ** 2) / 2 for v in ys]) / total)
            out["noise_share_all"][f"{t}|{lvl}"] = s
            all_max = max(all_max, s)
            if t in h.USABLE:
                usable_max = max(usable_max, s)
    out["noise_share_max_usable"] = usable_max
    out["noise_share_max_all42"] = all_max
    out["lead_noise_share_max"] = LEAD["summary"]["noise_share_max"]
    return out


# ---------------------------------------------------------------- DI band FIRs


def window_level(x: np.ndarray, start: int) -> np.ndarray:
    n = (len(x) - start) // W
    return 10 * np.log10(np.maximum(np.mean(x[start : start + n * W].reshape(n, W) ** 2, 1), 1e-30))


def band_index(level: np.ndarray) -> np.ndarray:
    return sum((level >= e).astype(int) for e in EDGES)


def accumulate(takes: tuple[int, ...]) -> dict:
    """Per band of the A windows (grid from 0): X^T X and X^T [y_t ..., y_16 - y_20],
    explicit design rows [1, x[n], ..., x[n-255]] (zero history before 0)."""
    path = SCRATCH / "a_grams.npz"
    if path.exists():
        z = np.load(path)
        if tuple(z["takes"]) == takes:
            return {"gram": z["gram"], "xty": z["xty"], "rows": z["rows"]}
    x = arr("x", h.FIT)
    ys = [h.load(t, (h.FIT,))[h.FIT] for t in takes]
    bands = band_index(window_level(x, 0))
    padded = np.concatenate((np.zeros(TAPS - 1), x))
    view = sliding_window_view(padded, TAPS)
    nb = len(EDGES) + 1
    gram = np.zeros((nb, TAPS + 1, TAPS + 1))
    xty = np.zeros((nb, TAPS + 1, len(takes) + 1))
    rows = np.zeros(nb, int)
    step = 20 * W  # 20 windows per chunk
    t0 = time.time()
    for start in range(0, len(bands) * W, step):
        stop = min(start + step, len(bands) * W)
        design = np.empty((stop - start, TAPS + 1))
        design[:, 0] = 1.0
        design[:, 1:] = view[start:stop][:, ::-1]  # column j+1 = x[n - j]
        yc = np.stack([np.asarray(y[start:stop]) for y in ys], 1)
        i16, i20 = takes.index(16), takes.index(20)
        yc = np.hstack((yc, yc[:, [i16]] - yc[:, [i20]]))
        per_row = np.repeat(bands[start // W : stop // W], W)
        for b in np.unique(per_row):
            pick = per_row == b
            d = design[pick]
            gram[b] += d.T @ d
            xty[b] += d.T @ yc[pick]
            rows[b] += pick.sum()
    print(f"A grams {time.time() - t0:.0f} s", flush=True)
    np.savez(path, gram=gram, xty=xty, rows=rows, takes=np.array(takes))
    return {"gram": gram, "xty": xty, "rows": rows}


def solve_ridge(gram: np.ndarray, b: np.ndarray, ridge: float) -> np.ndarray:
    reg = gram.copy()
    reg[1:, 1:] += ridge * np.mean(np.diag(gram)[1:]) * np.eye(TAPS)
    return solve(reg, b, assume_a="pos")


def render(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return theta[0] + fftconvolve(x, theta[1:])[: len(x)]


def pooled_share(y: np.ndarray, q: np.ndarray, windows: np.ndarray) -> float:
    n = len(windows)
    r = (y - q)[SKIP : SKIP + n * W].reshape(n, W)[windows]
    yy = y[SKIP : SKIP + n * W].reshape(n, W)[windows]
    return float(np.sum(r**2) / np.sum(yy**2))


def cap2(y: np.ndarray, q: np.ndarray) -> float:
    return h.ratio(y, q) ** 2


def per_sample(which: np.ndarray, length: int) -> np.ndarray:
    out = np.full(length, which[-1])
    out[:SKIP] = which[0]
    out[SKIP : SKIP + len(which) * W] = np.repeat(which, W)
    return out


def trailing_band(x: np.ndarray, span: int) -> np.ndarray:
    """Causal per-sample band: mean square of x over the `span` samples ending at n."""
    c = np.concatenate(([0.0], np.cumsum(x**2)))
    n = np.arange(len(x))
    ms = (c[n + 1] - c[np.maximum(n + 1 - span, 0)]) / span
    return band_index(10 * np.log10(np.maximum(ms, 1e-30)))


def schedules(thetas: list, x: np.ndarray, y: np.ndarray) -> dict:
    renders = np.stack([render(t, x) for t in thetas])
    which = band_index(window_level(x, SKIP))
    idx = np.arange(len(x))
    out = {}
    for name, sel in (
        ("noncausal_window", per_sample(which, len(x))),
        ("causal_previous_window", per_sample(np.concatenate((which[:1], which[:-1])), len(x))),
        ("causal_trailing_50ms_per_sample", trailing_band(x, W)),
        ("causal_trailing_20ms_per_sample", trailing_band(x, 960)),
    ):
        out[name] = cap2(y, renders[sel, idx])
    return out


def part_di(acc: dict) -> dict:
    x_b, x_v = arr("x", h.READ), arr("x", h.VALID)
    lvl_b = band_index(window_level(x_b, SKIP))
    gram, xty = acc["gram"], acc["xty"]
    out = {"A_rows_per_band": acc["rows"].tolist(), "B_windows_per_band":
           np.bincount(lvl_b, minlength=5).tolist(), "band_fir": {}, "schedule": {},
           "full_A_fir": {}}
    worst = 0.0
    for j, t in enumerate(DI_TAKES):
        y_b, y_v = arr(t, h.READ), arr(t, h.VALID)
        thetas = [solve_ridge(gram[b], xty[b][:, j], 1e-2) for b in range(5)]
        for i, lvl in enumerate(LEVELS, start=1):
            mine = pooled_share(y_b, render(thetas[i], x_b), lvl_b == i)
            lead = LEAD["takes"][str(t)]["di_test"][str(lvl)]["all"]["band_fir"]
            worst = max(worst, abs(mine - lead))
            out["band_fir"][f"{t}|{lvl}"] = {"mine": mine, "lead": lead,
                                             "multisine_D_lead": LEAD["takes"][str(t)]["multisine"][str(lvl)]["D"]}
        if t in H2.values():
            g = LEAD["takes"][str(t)]["di_test"]["global"]
            out["schedule"][str(t)] = {
                "B": schedules(thetas, x_b, y_b),
                "V": schedules(thetas, x_v, y_v),
                "lead_B_noncausal": g["scheduled_fir"] ** 2,
                "lead_B_causal": g["scheduled_fir_causal"] ** 2,
                "lead_V_noncausal": LEAD["takes"][str(t)]["di_test"]["validation"]["scheduled_fir"],
                "lead_V_causal": LEAD["takes"][str(t)]["di_test"]["validation"]["scheduled_fir_causal"],
            }
            full_g, full_b = gram.sum(0), xty.sum(0)[:, j]
            q2 = render(solve_ridge(full_g, full_b, 1e-2), x_b)
            q4 = render(solve_ridge(full_g, full_b, 1e-4), x_b)
            frozen = np.asarray(h.frozen_prediction("fir", t))
            out["full_A_fir"][str(t)] = {
                "ridge_1e-2_B": cap2(y_b, q2),
                "frozen_fir_cached_B": cap2(y_b, frozen),
                "rel_rms_gap_mine_vs_frozen": float(h.nr.rms(q2 - frozen) / h.nr.rms(frozen)),
                "ridge_1e-4_B": cap2(y_b, q4),
                "_q4_b": q4,
                "_q4_v": render(solve_ridge(full_g, full_b, 1e-4), x_v),
            }
    out["band_fir_max_abs_gap"] = worst
    # linearity in y of the band FIR (own fit): fit(y16) - fit(y20) - fit(y16 - y20)
    i16, i20 = DI_TAKES.index(16), DI_TAKES.index(20)
    th = [solve_ridge(gram[1], xty[1][:, k], 1e-2) for k in (i16, i20, len(DI_TAKES))]
    q = [render(v, x_b) for v in th]
    out["linearity_own_band_fir_-30"] = float(
        h.nr.rms((q[0] - q[1] - q[2])[SKIP:]) / h.nr.rms(q[2][SKIP:])
    )
    return out


def lead_linearity(acc: dict) -> dict:
    """The lead's RowFit/solve_fir on the -30 band of A: linearity in y and agreement
    with the own explicit-design fit (validates the lead's FFT rhs)."""
    import lead4_bla_ceiling as l4

    x_a, x_b = arr("x", h.FIT), arr("x", h.READ)
    lvl_a = band_index(window_level(x_a, 0))
    mask = np.zeros(len(x_a), bool)
    mask[: len(lvl_a) * W] = np.repeat(lvl_a == 1, W)
    fit = l4.RowFit(x_a, mask)
    y16, y20 = arr(16, h.FIT), arr(20, h.FIT)
    th = [l4.solve_fir(fit.gram, fit.rhs(v)) for v in (y16, y20, y16 - y20)]
    q = [l4.render(v, x_b) for v in th]
    mine = solve_ridge(acc["gram"][1], acc["xty"][1][:, DI_TAKES.index(20)], 1e-2)
    return {
        "lead_rowfit_linearity_-30": float(h.nr.rms((q[0] - q[1] - q[2])[SKIP:]) / h.nr.rms(q[2][SKIP:])),
        "lead_vs_own_theta_rel": float(np.linalg.norm(th[1] - mine) / np.linalg.norm(mine)),
        "lead_vs_own_gram_rel": float(np.linalg.norm(fit.gram - acc["gram"][1]) / np.linalg.norm(acc["gram"][1])),
    }


# ---------------------------------------------------------------- rule attribution


def window_sums(y: np.ndarray, qs: dict, windows: np.ndarray | None) -> dict:
    """Per 50 ms window (grid from SKIP), squared residual of each model, zeroed
    outside `windows` (all windows when None)."""
    n = (len(y) - SKIP) // W
    keep = np.ones(n, bool) if windows is None else windows
    return {
        name: ((y - q)[SKIP : SKIP + n * W].reshape(n, W) ** 2).sum(1) * keep
        for name, q in qs.items()
    }


def boot_sd(num: np.ndarray, den: np.ndarray, draws: int = 2000) -> float:
    """SD of E = 1 - sum(num)/sum(den) when the section's 1 s blocks (20 windows,
    remainder dropped) are drawn with replacement, as many as the section has."""
    nblk = len(num) // 20
    a, b = num[: nblk * 20].reshape(nblk, 20).sum(1), den[: nblk * 20].reshape(nblk, 20).sum(1)
    k = RNG.integers(0, nblk, (draws, max(round(len(num) / 20), 1)))
    return float(np.std(1 - a[k].sum(1) / b[k].sum(1)))


def part_rule(di: dict) -> dict:
    x_b, x_v = arr("x", h.READ), arr("x", h.VALID)
    lvl_b = band_index(window_level(x_b, SKIP))
    lvl_v = band_index(window_level(x_v, SKIP))
    out = {}
    for gain, t in H2.items():
        y_b, y_v = arr(t, h.READ), arr(t, h.VALID)
        full = di["full_A_fir"][str(t)]
        qb = {p: np.asarray(h.frozen_prediction(p, t)) for p in ("fir", "hammerstein_5", "volterra_laguerre_3")}
        qv = {p: np.asarray(h.frozen_prediction(p, t, read=h.VALID)) for p in qb}
        qb["fir_1e-4"], qv["fir_1e-4"] = full.pop("_q4_b"), full.pop("_q4_v")
        lead = LEAD["h2_rule"][f"{gain:g}"]["against_frozen_fir"]
        entry = out[f"{gain:g}"] = {"take": t}
        cells = {"global": (None, None), **{str(c): (lvl_b == i, lvl_v == i)
                                             for i, c in enumerate(LEVELS, start=1)}}
        for cell, (wb, wv) in cells.items():
            sb, sv = window_sums(y_b, qb, wb), window_sums(y_v, qv, wv)
            half = len(sv["fir"]) // 2  # 299 windows: the lead's split of V
            c = entry[cell] = {}
            for p in ("hammerstein_5", "volterra_laguerre_3"):
                e_b = 1 - sb[p].sum() / sb["fir"].sum()
                e_v = [1 - sv[p][s].sum() / sv["fir"][s].sum()
                       for s in (slice(0, half), slice(half, None))]
                blk = (sv[p][: len(sv[p]) // 20 * 20].reshape(-1, 20).sum(1),
                       sv["fir"][: len(sv[p]) // 20 * 20].reshape(-1, 20).sum(1))
                c[p] = {
                    "E_B_mine": float(e_b),
                    "E_B_lead": lead[p][cell]["E_B"],
                    "E_V_halves_mine": [float(v) for v in e_v],
                    "E_V_halves_lead": lead[p][cell]["E_V_halves"],
                    "margin_lead": lead[p][cell]["margin"],
                    "pass_lead": lead[p][cell]["pass"],
                    "boot_sd_E_one_V_half": boot_sd(sv[p][:half], sv["fir"][:half]),
                    "boot_sd_E_B_read_only": boot_sd(sb[p], sb["fir"]),
                    "V_1s_blocks_with_E_le_0": int(np.sum(blk[0] >= blk[1])),
                    "V_1s_blocks": len(blk[0]),
                    "E_B_vs_fir_ridge_1e-4": float(1 - sb[p].sum() / sb["fir_1e-4"].sum()),
                    "E_V_vs_fir_ridge_1e-4": float(1 - sv[p].sum() / sv["fir_1e-4"].sum()),
                }
            c["E_B_fir_1e-4_vs_frozen_fir"] = float(1 - sb["fir_1e-4"].sum() / sb["fir"].sum())
            c["E_V_fir_1e-4_vs_frozen_fir"] = float(1 - sv["fir_1e-4"].sum() / sv["fir"].sum())
    return out


def laguerre_linear(take: int = 20) -> dict:
    """Linear-only Laguerre basis at VL3's spec (size 8, pole 0.9, ridge 1e-4) and FIR
    256 at ridge 1e-4 by the library, fitted on A, read on B: share of VL3's and H5's
    E against the frozen FIR that a linear model of their own basis/ridge already gets."""
    x_a, x_b = arr("x", h.FIT), arr("x", h.READ)
    y_a, y_b = arr(take, h.FIT), arr(take, h.READ)
    out = {}
    for name, spec in (
        ("laguerre_order1_pole0.9_ridge1e-4", VolterraLaguerre(size=8, pole=0.9, order=1, ridge=1e-4, dc=True)),
        ("library_fir256_ridge1e-4", Probe(degree=1, taps=256, ridge=1e-4, dc=True)),
    ):
        res = fit_probe(spec, x_a, y_a)
        q = sum(probe_contributions(res, x_b).values())
        out[name] = cap2(y_b, q)
    g = LEAD["takes"][str(take)]["di_test"]["global"]
    out["frozen_fir_B"] = g["fir"] ** 2
    out["hammerstein_5_B"] = g["hammerstein_5"] ** 2
    out["volterra_laguerre_3_B"] = g["volterra_laguerre_3"] ** 2
    return out


# ---------------------------------------------------------------- JSON statistics


def spearman_block(a: list, b: list) -> dict:
    a, b = np.asarray(a), np.asarray(b)
    rho, p = stats.spearmanr(a, b)
    r, pp = stats.pearsonr(a, b)
    jack = [stats.spearmanr(np.delete(a, i), np.delete(b, i))[0] for i in range(len(a))]
    boot = []
    for _ in range(4000):
        k = RNG.integers(0, len(a), len(a))
        if np.ptp(a[k]) > 0 and np.ptp(b[k]) > 0:
            boot.append(stats.spearmanr(a[k], b[k])[0])
    return {
        "n": len(a), "spearman": float(rho), "p_spearman": float(p),
        "pearson": float(r), "p_pearson": float(pp),
        "jackknife_min_max": [float(min(jack)), float(max(jack))],
        "bootstrap_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }


def part_stats() -> dict:
    T = {int(k): v for k, v in LEAD["takes"].items()}
    loud = sorted(t for t, v in T.items() if not v["excluded_gate4"] and v["G"] > 0)
    out = {
        "G_ge_2.5_takes": loud,
        "matches_lead_group": loud == LEAD["summary"]["groups"]["G_ge_2.5"]["takes"],
        "G_values_of_loud": sorted({T[t]["G"] for t in loud}),
        "cab_on_in_loud": [t for t in loud if T[t]["cab"] != "off"],
    }
    sets = {
        "usable_G_ge_2.5": loud,
        "without_cab_on": [t for t in loud if T[t]["cab"] == "off"],
        "with_excluded_G_gt_0": sorted(t for t, v in T.items() if v["G"] > 0),
    }
    for name, takes in sets.items():
        e = out[name] = {}
        for lvl in map(str, LEVELS):
            d = [T[t]["multisine"][lvl]["D"] for t in takes]
            f = [T[t]["di_test"][lvl]["all"]["band_fir"] for t in takes]
            e[lvl] = spearman_block(d, f) | {
                "mean_gap_fir_minus_D": float(np.mean(np.subtract(f, d))),
                "max_gap": float(np.max(np.subtract(f, d))),
                "D_min_max": [float(min(d)), float(max(d))],
                "band_fir_min_max": [float(min(f)), float(max(f))],
            }
        g = [T[t]["prediction"]["global"] for t in takes]
        e["global_Dfirloo_weighted_B"] = spearman_block(
            [x["D_fir_loo_weighted_by_B_output"] for x in g], [x["fir_capacity2"] for x in g])
        e["global_Dfirloo_weighted_V"] = spearman_block(
            [x["D_fir_loo_weighted_by_V_output"] for x in g], [x["fir_capacity2"] for x in g])
        e["global_D_weighted_V"] = spearman_block(
            [x["D_weighted_by_V_output"] for x in g], [x["fir_capacity2"] for x in g])
    names = [str(v) for v in LEVELS]
    flat = [max(T[t]["multisine"][n]["D"] for n in names)
            - min(T[t]["multisine"][n]["D"] for n in names) for t in loud]
    out["D_flatness_max_range_over_levels"] = float(max(flat))
    # level re-match per take: best case = highest D at any measured level against
    # the lowest DI band FIR at any band (both within -30..-18 dBFS)
    rematch = {t: min(T[t]["di_test"][n]["all"]["band_fir"] for n in names)
               - max(T[t]["multisine"][n]["D"] for n in names) for t in loud}
    out["rematch_min_gap_over_takes"] = float(min(rematch.values()))
    out["rematch_takes_with_gap_le_0"] = [t for t, v in rematch.items() if v <= 0]
    out["rematch_gap_by_take"] = {str(t): float(v) for t, v in rematch.items()}
    out["examples_-30"] = {str(t): {"D": T[t]["multisine"]["-30"]["D"],
                                    "band_fir": T[t]["di_test"]["-30"]["all"]["band_fir"],
                                    "setting": T[t]["setting"]} for t in (4, 33)}
    usable = sorted(t for t, v in T.items() if not v["excluded_gate4"])
    out["usable34_pearson"] = {
        lvl: float(stats.pearsonr([T[t]["multisine"][lvl]["D"] for t in usable],
                                  [T[t]["di_test"][lvl]["all"]["band_fir"] for t in usable])[0])
        for lvl in map(str, LEVELS)}
    # within-segment level spread of the 1 s local fits (x only, A)
    x_a = arr("x", h.FIT)
    lv = window_level(x_a, 0).reshape(-1, 20)
    seg = 10 * np.log10(np.mean(10 ** (lv / 10), 1))
    out["local_1s_segments"] = {
        str(c): {
            "segments": int(np.sum((seg >= c - 3) & (seg < c + 3))),
            "median_sd_window_level_db": float(np.median(lv[(seg >= c - 3) & (seg < c + 3)].std(1))),
            "share_windows_outside_band": float(np.mean(np.abs(lv[(seg >= c - 3) & (seg < c + 3)] - c) >= 3)),
        } for c in LEVELS}
    return out


def memory_check(takes=(30, 20), taps_list=(256, 1024), stride: int = 5) -> dict:
    """Is 256 taps (5.3 ms) a limit on the DI? Band FIR on every `stride`-th A row of
    the -30 dBFS band, 256 vs 1024 taps (relative ridge 1e-2, constant), read on the
    -30 band windows of B and V. Same row subset for both lengths."""
    x_a, x_b, x_v = arr("x", h.FIT), arr("x", h.READ), arr("x", h.VALID)
    rows = np.flatnonzero(np.repeat(band_index(window_level(x_a, 0)) == 1, W))[::stride]
    ys = np.stack([arr(t, h.FIT)[rows] for t in takes], 1)
    win_b = band_index(window_level(x_b, SKIP)) == 1
    win_v = band_index(window_level(x_v, SKIP)) == 1
    out = {"rows": len(rows), "stride": stride}
    for taps in taps_list:
        view = sliding_window_view(np.concatenate((np.zeros(taps - 1), x_a)), taps)
        gram, xty = np.zeros((taps + 1, taps + 1)), np.zeros((taps + 1, len(takes)))
        for start in range(0, len(rows), 20_000):
            r = rows[start : start + 20_000]
            d = np.hstack((np.ones((len(r), 1)), view[r][:, ::-1]))
            gram += d.T @ d
            xty += d.T @ ys[start : start + 20_000]
        reg = gram.copy()
        reg[1:, 1:] += 1e-2 * np.mean(np.diag(gram)[1:]) * np.eye(taps)
        for j, t in enumerate(takes):
            th = solve(reg, xty[:, j], assume_a="pos")
            out[f"take{t}_taps{taps}"] = {
                "B_-30": pooled_share(arr(t, h.READ), th[0] + fftconvolve(x_b, th[1:])[: len(x_b)], win_b),
                "V_-30": pooled_share(arr(t, h.VALID), th[0] + fftconvolve(x_v, th[1:])[: len(x_v)], win_v),
            }
    return out


def strip(v):
    if isinstance(v, dict):
        return {k: strip(x) for k, x in v.items() if not k.startswith("_")}
    if isinstance(v, list | tuple):
        return [strip(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = {"stats_from_lead_json": part_stats()}
    print(f"stats {time.time() - t0:.0f} s", flush=True)
    result["multisine"] = part_multisine()
    print(f"multisine {time.time() - t0:.0f} s", flush=True)
    acc = accumulate(DI_TAKES)
    di = part_di(acc)
    print(f"di {time.time() - t0:.0f} s", flush=True)
    result["rule"] = part_rule(di)
    result["di"] = di
    result["lead_linearity"] = lead_linearity(acc)
    result["linear_basis_take20"] = laguerre_linear(20)
    result["memory_check_-30"] = memory_check()
    print(f"memory {time.time() - t0:.0f} s", flush=True)
    result["runtime_s"] = time.time() - t0
    OUT.write_text(json.dumps(h.nr.finite(strip(result)), indent=1) + "\n")
    print(f"done {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
