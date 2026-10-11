#!/usr/bin/env python3
"""Lead 3: RF power-amplifier behavioural bases as capacity probes (exploration, CPU).

Question: after a prefilter, does the SHAPE of a basis designed for hard saturation
lower the residual of the Nolly renders? Discriminating observation fixed before
measuring: output capacity below 0.6 at gain 5 (take 20), fitted on A, ridge chosen
off B, read once on B. Baselines to beat: the frozen probes (baseline.json; H5 0.803
at G5) and the Wiener-tanh basis of the scratch study, re-run here in protocol.

Bases, real-valued transcriptions (z = prefiltered input / its RMS on A, frozen):
- GMP, Morgan et al. 2006: aligned z|z|^(k-1) and |z|^k (k = 1..5; the even-symmetric
  |z|^k are a real-signal addition, they contain H5), lagging z(n)|z(n-m)|^k and leading
  z(n-m)|z(n)|^k (k = 1, 2; m in LAGS). The RF envelope |x| of complex baseband is
  replaced by the instantaneous |z|; a smoothed envelope (dynamic gain) is lead 4.
- DVR, Zhu 2015 (equation as given by Kantana et al. 2021, eq. 1), e^{j theta} ->
  sign(z): type 1 ||z|-b| sign z, type 4 ||z|-b| z, cross types 2 ||z(n-i)|-b|
  sign z(n-i) |z(n)|, 3 ||z(n-i)|-b| z(n), 5 ||z(n)|-b| z(n-i); plus the even ||z|-b|
  (real-signal addition, asymmetric clipping). Knots b = KNOTS (times the RMS on A).
  "dvr1" = z, |z|, type 1 and even terms: type 1 jumps by 2b at z = 0, so this family
  is NOT the continuous piecewise-linear one (Codex review).
- CPWL: z, |z|, (|z|-b)+ sign z, ||z|-b|: every continuous piecewise-linear function
  with knots at +-b (a hard clipper whose knee sits on a knot); "cpwl_sign" adds
  sign z (a zero-crossing jump: hard limiter). "all" = union + hinges + tanh + sign.
Each feature goes through its own causal FIR (TAPS) plus a constant, ridge relative
to the mean Gram diagonal (fssr_nam.product.probes), exactly as harness.FeatureProbe:
features [z^1..z^5] on the raw input reproduce Hammerstein 5 (instrument check).

Prefilters: raw x; fixed 2nd-order Butterworth high-pass at 300 Hz (frozen: probe
linear in the target); BLA of the take = FIR 256 (frozen FIR spec, ridge 1e-2) fitted
on A to the take's own output (NOT linear in the target: labelled). For the gain-pair
differences the BLA is that of take 20 (centre setting; refs 1 and 39 are repeats of
it at ESR 2e-8), fixed before fitting d, so linear in d. Every prefiltered basis also
has the raw x through its own FIR.

Fitting: all bases of one prefilter are column subsets of one master Gram (any subset
of features and taps <= TAPS), computed by block FFT correlation (same full-support
sums as probes.normal_equations; checked against it). One ridge per (prefilter,
basis, master) chosen on a split of A (fit A1 = first 3/4, mean error ratio over the
master's targets on A2 = last 1/4), then refit on all of A, read on B once. B never
enters a fit except in rows labelled ORACLE (basis fitted in-sample on B, prefilter
still from A).

CLI (repository root, CPU, 3 threads):
    uv run python diagnosis/probe_capacity/lead3_rf_pa_bases.py nolly
    uv run python diagnosis/probe_capacity/lead3_rf_pa_bases.py jvm
    uv run python diagnosis/probe_capacity/lead3_rf_pa_bases.py check
    uv run python diagnosis/probe_capacity/lead3_rf_pa_bases.py table
nolly, jvm and check update diagnosis/probe_capacity/lead3_rf_pa_bases.json; table
prints markdown tables from it next to the frozen baselines (baseline.json).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.fft import irfft, rfft
from scipy.linalg import cho_factor, cho_solve
from scipy.signal import butter, fftconvolve, sosfilt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as h  # noqa: E402

from fssr_nam.product.probes import (  # noqa: E402
    Probe,
    fit,
    normal_equations,
    regularized,
)

nr = h.nr
OUT = HERE / "lead3_rf_pa_bases.json"
WORKERS = 3
TAPS = 256
BLOCK, NFFT = 3840, 4096  # BLOCK + TAPS <= NFFT: no circular wrap at lags < TAPS
SPLIT = 0.75  # A1 = first 3/4 of A's blocks, A2 = the rest (ridge selection only)
RIDGES = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
ORACLE_RIDGE = 1e-6
KNOTS = (1 / 64, 1 / 16, 1 / 4, 1 / 2, 1.0, 2.0)
CROSS_KNOTS = (1 / 16, 1 / 2)
LAGS = (1, 8)
TANH = (0.5, 1, 2, 4, 8, 16)
PARAM_BUDGET = 1280  # H5: 5 branches x 256 taps (+ constant)


def rms(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(v))))


def delay(v: np.ndarray, m: int) -> np.ndarray:
    out = np.zeros_like(v)
    out[m:] = v[:-m]
    return out


def bank(z: np.ndarray, x: np.ndarray, raw: bool) -> dict[str, np.ndarray]:
    """Every candidate feature (frozen: depends on the input only; zero history
    before the section start, as an input continued by zeros)."""
    a, s = np.abs(z), np.sign(z)
    f = {} if raw else {"x": x}
    for k in range(1, 6):
        f[f"o{k}"] = z * a ** (k - 1)  # z|z|^(k-1): z, z|z|, z^3, z|z|^3, z^5
        f[f"e{k}"] = a**k  # |z|^k: |z|, z^2, |z|^3, z^4, |z|^5
    for m in LAGS:
        am, zm = delay(a, m), delay(z, m)
        for k in (1, 2):
            f[f"gmp_lag{m}_k{k}"] = z * am**k
            f[f"gmp_lead{m}_k{k}"] = zm * a**k
    for b in KNOTS:
        g = np.abs(a - b)
        f[f"hinge_{b:g}"] = np.maximum(a - b, 0) * s
        f[f"dvr1_{b:g}"] = g * s
        f[f"dvr4_{b:g}"] = g * z
        f[f"even_{b:g}"] = g
    for b in CROSS_KNOTS:
        g = np.abs(a - b)
        for m in LAGS:
            gm = np.abs(delay(a, m) - b)  # = b before the start (zero history)
            f[f"dvr2_{b:g}_{m}"] = gm * delay(s, m) * a
            f[f"dvr3_{b:g}_{m}"] = gm * z
            f[f"dvr5_{b:g}_{m}"] = g * delay(z, m)
    for c in TANH:
        f[f"tanh{c:g}"] = np.tanh(c * z)
    f["sign"] = s
    return f


def bases(names: list[str]) -> dict[str, list[str]]:
    """Named column subsets of the master feature list."""

    def pick(*prefixes: str) -> list[str]:
        return [n for n in names if n.startswith(prefixes)]

    lin = pick("x")  # the raw x branch (absent on the raw prefilter, where z = x/rms)
    aligned = lin + [f"o{k}" for k in range(1, 6)] + [f"e{k}" for k in range(1, 6)]
    dvr1 = [*lin, "o1", "e1", *pick("dvr1_", "even_")]
    cpwl = [*lin, "o1", "e1", *pick("hinge_", "even_")]
    dvr_static = dvr1 + pick("dvr4_")
    union = aligned + pick("gmp_", "dvr1_", "dvr4_", "even_", "dvr2_", "dvr3_", "dvr5_")
    return {
        "h5": [*lin, "o1", "e2", "o3", "e4", "o5"],
        "tanh": [*lin, "o1", *pick("tanh", "sign")],
        "gmp_aligned": aligned,
        "gmp": aligned + pick("gmp_"),
        "dvr1": dvr1,
        "cpwl": cpwl,
        "cpwl_sign": [*cpwl, "sign"],
        "dvr_static": dvr_static,
        "dvr": dvr_static + pick("dvr2_", "dvr3_", "dvr5_"),
        "union": union,
        "all": union + pick("hinge_", "tanh", "sign"),
    }


RF_BASES = ("gmp_aligned", "gmp", "dvr1", "cpwl", "cpwl_sign", "dvr_static", "dvr")
RF_BASES += ("union",)
BLA_BASES = ["h5", "tanh", "cpwl_sign", "dvr", "union"]  # the BLA prefilter: subset


def blocks(v: np.ndarray, extend: bool) -> np.ndarray:
    """rfft (NFFT) of consecutive BLOCK-sample blocks of v, each extended by the TAPS
    following samples when `extend` (zeros past the end): (blocks, NFFT // 2 + 1)."""
    count = -(-len(v) // BLOCK)
    padded = np.zeros(count * BLOCK + TAPS)
    padded[: len(v)] = v
    win = np.lib.stride_tricks.sliding_window_view(padded, BLOCK + TAPS)[::BLOCK]
    win = win[:count] if extend else win[:count, :BLOCK]
    return rfft(win, NFFT, axis=1, workers=WORKERS)


class Master:
    """Frozen features of one prefilter on one fit section, their scales, and the
    lagged correlations c[tau, k, j] = sum_n u_k[n] u_j[n + tau] (tau < TAPS) on the
    parts A1, A2 and all, from which any subset Gram is assembled. Each part is
    treated as a signal of its own (zero outside it), so that its Gram is the exact
    full-support Gram of that part (positive semi-definite); "all" is the library's
    Gram of the whole section, not the sum of the parts."""

    def __init__(
        self,
        prefilter: Callable[[np.ndarray], np.ndarray],
        x_fit: np.ndarray,
        raw: bool,
        sigma: float | None = None,
    ) -> None:
        self.prefilter, self.raw = prefilter, raw
        u = prefilter(x_fit)
        self.sigma = rms(u) if sigma is None else sigma
        feats = bank(u / self.sigma, x_fit, raw)
        self.names = list(feats)
        self.scales = {k: max(rms(v), 1e-300) for k, v in feats.items()}
        self.n = n = len(x_fit)
        count = -(-n // BLOCK)
        cut = int(SPLIT * count)
        self.cut = cut
        self.spans = {"A1": (0, cut * BLOCK), "A2": (cut * BLOCK, n), "all": (0, n)}
        width = NFFT // 2 + 1
        size = len(feats)
        self.xc = np.empty((width, size, count), complex)
        y = np.empty((width, count, size), complex)
        self.sums = {p: np.empty((size, TAPS)) for p in self.spans}
        for k, name in enumerate(self.names):
            v = feats.pop(name) / self.scales[name]
            self.xc[:, k, :] = np.conj(blocks(v, False)).T
            y[:, :, k] = blocks(v, True).T
            total = np.concatenate(([0.0], np.cumsum(v)))
            for p, (s, e) in self.spans.items():
                self.sums[p][k] = total[e - np.arange(TAPS)] - total[s]
        # the last block of A1 extended into A2 (continuous: "all") or with zeros
        # (A1 alone: its extension is the plain block, i.e. conj(xc))
        last = np.conj(self.xc[:, :, cut - 1 : cut]).transpose(0, 2, 1)
        self.c = self._parts(y, last)

    def _parts(self, y: np.ndarray, last: np.ndarray) -> dict[str, np.ndarray]:
        """Correlations of the features with y (blocks extended; (F, blocks, m)) per
        part; `last` is A1's last block of y with a zero extension, (F, 1, m)."""
        cut = self.cut

        def corr(xc: np.ndarray, other: np.ndarray) -> np.ndarray:
            spec = np.empty((xc.shape[0], xc.shape[1], other.shape[2]), complex)
            for f in range(0, xc.shape[0], 64):
                spec[f : f + 64] = xc[f : f + 64] @ other[f : f + 64]
            return irfft(spec, NFFT, axis=0, workers=WORKERS)[:TAPS]

        core = corr(self.xc[:, :, : cut - 1], y[:, : cut - 1])
        a2 = corr(self.xc[:, :, cut:], y[:, cut:])
        edge = self.xc[:, :, cut - 1 : cut]
        return {
            "A1": core + corr(edge, last),
            "A2": a2,
            "all": core + corr(edge, y[:, cut - 1 : cut]) + a2,
        }

    def rhs(self, targets: list[np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
        """Correlations of the features with each target, per part: rc[tau, k, t],
        dc[t] (sum of the target) and yy[t] (sum of squares)."""
        targets = [np.asarray(t) for t in targets]
        yt = np.stack([blocks(t, True).T for t in targets], axis=2)
        last = np.stack([blocks(t, False)[self.cut - 1] for t in targets], axis=1)[
            :, None, :
        ]
        out = {p: {"rc": c} for p, c in self._parts(yt, last).items()}
        for p, (s, e) in self.spans.items():
            seg = [np.asarray(t[s:e]) for t in targets]
            out[p]["dc"] = np.array([v.sum() for v in seg])
            out[p]["yy"] = np.array([v @ v for v in seg])
        return out

    def columns(self, basis: list[str], taps: int = TAPS) -> list[tuple[int, int]]:
        return [(self.names.index(b), taps) for b in basis]

    def gram(self, part: str, cols: list[tuple[int, int]]) -> np.ndarray:
        c, sums = self.c[part], self.sums[part]
        s, e = self.spans[part]
        offs = np.cumsum([1] + [m for _, m in cols])
        g = np.empty((offs[-1], offs[-1]))
        g[0, 0] = e - s
        lags = np.arange(TAPS)
        for a, (k, mk) in enumerate(cols):
            ra = slice(offs[a], offs[a] + mk)
            g[0, ra] = g[ra, 0] = sums[k, :mk]
            for b in range(a, len(cols)):
                j, mj = cols[b]
                rb = slice(offs[b], offs[b] + mj)
                lag = lags[:mk, None] - lags[None, :mj]
                block = np.where(
                    lag >= 0, c[np.maximum(lag, 0), k, j], c[np.maximum(-lag, 0), j, k]
                )
                g[ra, rb] = block
                g[rb, ra] = block.T
        return g

    @staticmethod
    def vector(r: dict, cols: list[tuple[int, int]]) -> np.ndarray:
        return np.concatenate(
            [r["dc"][None, :]] + [r["rc"][:m, k, :] for k, m in cols], axis=0
        )

    def features(self, x: np.ndarray) -> dict[str, np.ndarray]:
        z = self.prefilter(x) / self.sigma
        return bank(z, x, self.raw)

    def render(
        self, feats: dict, cols: list[tuple[int, int]], theta: np.ndarray
    ) -> np.ndarray:
        """Predictions (n, targets) on a section from its features."""
        n = len(feats[self.names[0]])
        out = np.tile(theta[0], (n, 1))
        start = 1
        for k, m in cols:
            name = self.names[k]
            v = feats[name] / self.scales[name]
            for t in range(theta.shape[1]):
                out[:, t] += fftconvolve(v, theta[start : start + m, t])[:n]
            start += m
        return out


def solve(gram: np.ndarray, ridge: float, rhs: np.ndarray) -> np.ndarray:
    factor = cho_factor(regularized(Probe(ridge=ridge), gram), overwrite_a=True)
    return cho_solve(factor, rhs)


def fit_basis(master: Master, cols: list, r: dict) -> tuple[np.ndarray, list, list]:
    """One ridge for every target of the master (the lowest mean A2 ratio after a
    fit on A1), then refit on all of A: a single linear operator per prefilter and
    basis, so fit(y_eval) - fit(y_ref) = fit(d) holds for the reported numbers.
    Returns theta (size, targets), the ridge per target (all equal) and the A2
    residual ratios per ridge and target."""
    g1 = master.gram("A1", cols)
    thetas = [solve(g1, ridge, master.vector(r["A1"], cols)) for ridge in RIDGES]
    del g1
    g2, r2 = master.gram("A2", cols), master.vector(r["A2"], cols)
    errors = []
    for theta in thetas:
        fit_term = np.sum(theta * (g2 @ theta), 0) - 2 * np.sum(theta * r2, 0)
        err = r["A2"]["yy"] + fit_term
        errors.append(np.sqrt(np.maximum(err, 0) / r["A2"]["yy"]))
    del g2
    errors = np.array(errors)  # (ridges, targets)
    ridge = RIDGES[int(errors.mean(1).argmin())]
    theta = solve(master.gram("all", cols), ridge, master.vector(r["all"], cols))
    return theta, [ridge] * theta.shape[1], errors.T.tolist()


class SubsetProbe:
    """harness.CapacityProbe over a Master (fit section fixed by the master), one
    basis, one ridge: lets the harness's difference_share / linearity_check run."""

    def __init__(self, master: Master, basis: list[str], ridge: float) -> None:
        self.master, self.cols, self.ridge = master, master.columns(basis), ridge
        self.gram = master.gram("all", self.cols)

    def fit(self, inputs: dict, target: np.ndarray) -> SubsetProbe:
        assert len(inputs["x"]) == self.master.n
        r = self.master.rhs([target])["all"]
        self.theta = solve(self.gram, self.ridge, self.master.vector(r, self.cols))
        return self

    def predict(self, inputs: dict) -> np.ndarray:
        feats = self.master.features(np.asarray(inputs["x"]))
        return self.master.render(feats, self.cols, self.theta)[:, 0]


# ---------------------------------------------------------------- prefilters


def hp300(fs: float) -> Callable[[np.ndarray], np.ndarray]:
    sos = butter(2, 300, "highpass", fs=fs, output="sos")
    return lambda x: sosfilt(sos, x)


def identity(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)


def bla(x_fit: np.ndarray, y_fit: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """FIR 256 (frozen FIR spec, ridge 1e-2) fitted on (x_fit, y_fit), constant
    dropped: the scratch study's BLA prefilter."""
    result = fit(h.FROZEN["fir"], np.asarray(x_fit), np.asarray(y_fit))
    taps = result.theta[1:] / result.scales[0]
    return lambda x: fftconvolve(np.asarray(x, dtype=float), taps)[: len(x)]


# ---------------------------------------------------------------- Nolly


def nolly_targets() -> list[dict]:
    out = [{"kind": "output", "take": t, "gain": g} for g, t in h.GAIN_TAKES.items()]
    for p in h.pairs(("one_knob",)):
        if h.knob(p) == "G":
            out.append(
                {
                    "kind": "difference",
                    "ref": p["ref"],
                    "eval": p["eval"],
                    "setting": h.ROWS[p["eval"]]["setting"],
                }
            )
    return out


def target_signal(t: dict, section: str) -> np.ndarray:
    if t["kind"] == "output":
        return np.asarray(h.load(t["take"], (section,))[section])
    return h.difference(t["ref"], t["eval"], section)


def score(t: dict, q: np.ndarray, x_read: np.ndarray) -> dict:
    y = target_signal(t, h.READ)
    if t["kind"] == "output":
        return {
            "capacity_B": h.ratio(y, q),
            "by_level_B": h.level_capacity_from(x_read, y, q),
        }
    return {"share_B": h.ratio(y, q, floor=0.0)}


def run_master(
    label: str,
    master: Master,
    targets: list[dict],
    names: list[str],
    flags: dict,
    log: Callable,
) -> list[dict]:
    """Every basis on every target of one master: selection on A, one read of B."""
    r = master.rhs([target_signal(t, h.FIT) for t in targets])
    x_read = np.asarray(h.load("x", (h.READ,))[h.READ])
    feats = master.features(x_read)
    rows = []
    for basis in names:
        start = time.time()
        members = bases(master.names)[basis]
        cols = master.columns(members)
        theta, ridges, errors = fit_basis(master, cols, r)
        q = master.render(feats, cols, theta)
        for i, t in enumerate(targets):
            rows.append(
                {
                    "prefilter": label,
                    "basis": basis,
                    **t,
                    **flags.get(t["kind"], {}),
                    "features": len(cols),
                    "params": 1 + sum(m for _, m in cols),
                    "ridge": ridges[i],
                    "A2_ratio_by_ridge": dict(
                        zip(map(str, RIDGES), errors[i], strict=True)
                    ),
                    **score(t, q[:, i], x_read),
                }
            )
        log(f"{label} {basis} {len(cols)} features {time.time() - start:.0f} s")
    return rows


def matched(
    label: str, master: Master, targets: list[dict], rows: list[dict], log: Callable
) -> list[dict]:
    """The RF basis with the lowest A2 error on take 20 (G5) at its retained ridge,
    its taps cut so that its parameters fit H5's budget (1 + 1280)."""

    def a2(row: dict) -> float:  # at the ridge retained for the published row
        return row["A2_ratio_by_ridge"][str(row["ridge"])]

    g5 = [
        row
        for row in rows
        if row["prefilter"] == label
        and row["basis"] in RF_BASES
        and row.get("take") == 20
    ]
    best = min(g5, key=a2)["basis"]
    members = bases(master.names)[best]
    taps = PARAM_BUDGET // len(members)
    cols = master.columns(members, taps)
    r = master.rhs([target_signal(t, h.FIT) for t in targets])
    theta, ridges, errors = fit_basis(master, cols, r)
    x_read = np.asarray(h.load("x", (h.READ,))[h.READ])
    q = master.render(master.features(x_read), cols, theta)
    log(f"{label} matched {best} taps {taps}")
    return [
        {
            "prefilter": label,
            "basis": f"{best}@{taps}taps",
            **t,
            "features": len(cols),
            "params": 1 + sum(m for _, m in cols),
            "ridge": ridges[i],
            "A2_ratio_by_ridge": dict(zip(map(str, RIDGES), errors[i], strict=True)),
            **score(t, q[:, i], x_read),
        }
        for i, t in enumerate(targets)
    ]


def oracle(
    label: str, master_a: Master, take: int, names: list[str], log: Callable
) -> list[dict]:
    """ORACLE: same prefilter and knots as on A, basis fitted in-sample on B."""
    x_b = np.asarray(h.load("x", (h.READ,))[h.READ])
    y_b = np.asarray(h.load(take, (h.READ,))[h.READ])
    mb = Master(master_a.prefilter, x_b, master_a.raw, master_a.sigma)
    r = mb.rhs([y_b])["all"]
    feats = mb.features(x_b)
    out = []
    for basis in names:
        cols = mb.columns(bases(mb.names)[basis])
        theta = solve(mb.gram("all", cols), ORACLE_RIDGE, mb.vector(r, cols))
        q = mb.render(feats, cols, theta)[:, 0]
        out.append(
            {
                "label": "ORACLE (basis fitted in-sample on B)",
                "prefilter": label,
                "basis": basis,
                "take": take,
                "params": 1 + sum(m for _, m in cols),
                "ridge": ORACLE_RIDGE,
                "capacity_B_in_sample": h.ratio(y_b, q),
            }
        )
        log(f"oracle {label} {basis} {out[-1]['capacity_B_in_sample']:.4f}")
    return out


def instrument_check(log: Callable) -> dict:
    """Block-FFT Gram vs probes.normal_equations, and H5 parity with the frozen probe
    and harness.FeatureProbe on take 20 (raw prefilter) and on take 2."""
    x_a = np.asarray(h.load("x", (h.FIT,))[h.FIT])
    master = Master(identity, x_a, raw=True)
    cols = master.columns(["o1", "e2", "o3", "e4", "o5"])
    takes = [20, 2]
    r = master.rhs([np.asarray(h.load(t, (h.FIT,))[h.FIT]) for t in takes])
    theta = solve(master.gram("all", cols), 1e-4, master.vector(r["all"], cols))
    x_b = np.asarray(h.load("x", (h.READ,))[h.READ])
    q = master.render(master.features(x_b), cols, theta)
    out = {}
    for i, t in enumerate(takes):
        y_b = np.asarray(h.load(t, (h.READ,))[h.READ])
        frozen = np.asarray(h.frozen_prediction("hammerstein_5", t))
        out[f"take_{t}"] = {
            "h5_capacity_master": h.ratio(y_b, q[:, i]),
            "h5_capacity_frozen": h.ratio(y_b, frozen),
            "max_abs_prediction_gap_rel": float(
                np.max(np.abs(q[:, i] - frozen)) / rms(frozen)
            ),
        }
    # lagged and cross features on the HP300 prefilter vs the library Gram
    hp = Master(hp300(h.FS), x_a, raw=False)
    names = ["x", "gmp_lag8_k1", "dvr2_0.5_8", "dvr1_0.0625"]
    cols = hp.columns(names, 32)
    feats = hp.features(x_a)
    u = [feats[n] / hp.scales[n] for n in names]
    y_a = np.asarray(h.load(20, (h.FIT,))[h.FIT])
    gram_ref, rhs_ref = normal_equations(Probe(degree=4, taps=32), u, y_a)
    gram = hp.gram("all", cols)
    rhs = hp.vector(hp.rhs([y_a])["all"], cols)[:, 0]
    out["gram_vs_normal_equations"] = {
        "gram_max_rel_gap": float(np.max(np.abs(gram - gram_ref)) / np.max(gram_ref)),
        "rhs_max_rel_gap": float(
            np.max(np.abs(rhs - rhs_ref)) / np.max(np.abs(rhs_ref))
        ),
    }

    # the same basis through harness.FeatureProbe (A -> B), against the master
    def chosen(inputs: dict) -> list[np.ndarray]:
        f = hp.features(np.asarray(inputs["x"]))
        return [f[n] for n in names]

    probe = h.FeatureProbe(chosen, 32, 1e-4)
    q_feature = h.fit_read(probe, y_a)
    theta = solve(gram, 1e-4, rhs[:, None])
    q_master = hp.render(hp.features(x_b), cols, theta)[:, 0]
    out["feature_probe_parity"] = {
        "capacity_feature_probe": h.ratio(np.asarray(h.load(20)[h.READ]), q_feature),
        "capacity_master": h.ratio(np.asarray(h.load(20)[h.READ]), q_master),
        "max_abs_prediction_gap_rel": float(
            np.max(np.abs(q_feature - q_master)) / rms(q_feature)
        ),
    }
    out["pass"] = bool(
        all(
            abs(v["h5_capacity_master"] - v["h5_capacity_frozen"]) < 1e-9
            for k, v in out.items()
            if k.startswith("take_")
        )
        and out["gram_vs_normal_equations"]["gram_max_rel_gap"] < 1e-10
        and out["feature_probe_parity"]["max_abs_prediction_gap_rel"] < 1e-8
    )
    log(f"instrument check {json.dumps(out)}")
    return out


def nolly(log: Callable) -> dict:
    result = {"instrument_check": instrument_check(log)}
    x_a = np.asarray(h.load("x", (h.FIT,))[h.FIT])
    targets = nolly_targets()
    rows, oracles, matches, linearity = [], [], [], {}
    names = list(bases([]))
    frozen = {
        k: {"linear_in_target": True, "note": "one ridge for all targets"}
        for k in ("output", "difference")
    }
    best = ["h5", "tanh", "cpwl_sign", "union", "all"]
    for label, pre, raw in (("raw", identity, True), ("hp300", hp300(h.FS), False)):
        master = Master(pre, x_a, raw)
        rows += run_master(label, master, targets, names, frozen, log)
        matches += matched(label, master, targets, rows, log)
        oracles += oracle(label, master, 20, best, log)
        ridge = next(
            r["ridge"]
            for r in rows
            if r["prefilter"] == label and r["basis"] == "union" and r.get("eval") == 7
        )
        probe = SubsetProbe(master, bases(master.names)["union"], ridge)
        linearity[f"{label}_union_1_7"] = h.linearity_check(probe, 1, 7)
        linearity[f"{label}_union_1_7_share_via_harness"] = h.difference_share(
            probe, 1, 7
        )
        log(f"linearity {label} {linearity}")
        del master, probe
    # BLA of each ladder take, fitted on A to that take (target-dependent prefilter)
    for target in [t for t in targets if t["kind"] == "output"]:
        take = target["take"]
        y_a = np.asarray(h.load(take, (h.FIT,))[h.FIT])
        master = Master(bla(x_a, y_a), x_a, raw=False)
        flags = {
            "output": {
                "linear_in_target": False,
                "note": "prefilter = BLA of this take fitted on all of A to its own "
                "output: the A2 ridge selection is not independent of the prefilter",
            },
            "difference": {
                "linear_in_target": True,
                "note": "prefilter = BLA of take 20 (centre setting, a repeat of "
                "refs 1 and 39) fitted on all of A, fixed before fitting d; one "
                "ridge for the four differences",
            },
        }
        rows += run_master("bla_take", master, [target], BLA_BASES, flags, log)
        if take == 20:
            diffs = [t for t in targets if t["kind"] == "difference"]
            rows += run_master("bla_take", master, diffs, BLA_BASES, flags, log)
            oracles += oracle("bla_take", master, 20, ["h5", "tanh", "union"], log)
        del master
    result.update(
        {
            "rows": rows,
            "matched_parameters": matches,
            "oracle": oracles,
            "linearity": linearity,
        }
    )
    return result


# ---------------------------------------------------------------- JVM410H OD1


def jvm(log: Callable) -> dict:
    """Same machinery on the Marshall JVM410H OD1 (ToneTwist, INTERNAL_DEV, 44.1 kHz)
    gain ladder, PreampOut: A = 0-150 s, B = 215-300 s of one input file (source
    independence NOT established). Ridge chosen on the A split, B read once."""
    sys.path.insert(0, str(h.ROOT / "diagnosis/probe_metrics"))
    import real_pairs

    settings = ("B050_M050_T050_G000", "B050_M050_T050_G050", "B050_M050_T050_G100")
    rate, x, _ = real_pairs.jvm_audio("PreampOut", settings[0])
    skip = round(0.05 * rate)
    ys = {s: real_pairs.jvm_audio("PreampOut", s)[2] for s in settings}
    names = {  # a subset, to keep the runtime down (machine shared with other runs)
        "raw": ["h5", "tanh"],
        "hp300": ["h5", "tanh", "cpwl_sign", "dvr"],
    }
    rows = []
    for label, pre, raw in (("raw", identity, True), ("hp300", hp300(rate), False)):
        master = Master(pre, x["A"], raw)
        r = master.rhs([ys[s]["A"] for s in settings])
        feats = master.features(x["B"])
        for basis in names[label]:
            cols = master.columns(bases(master.names)[basis])
            theta, ridges, errors = fit_basis(master, cols, r)
            q = master.render(feats, cols, theta)
            for i, s in enumerate(settings):
                y = ys[s]["B"]
                rows.append(
                    {
                        "prefilter": label,
                        "basis": basis,
                        "setting": s,
                        "params": 1 + sum(m for _, m in cols),
                        "ridge": ridges[i],
                        "A2_ratio_by_ridge": dict(
                            zip(map(str, RIDGES), errors[i], strict=True)
                        ),
                        "capacity_B": rms((y - q[:, i])[skip:])
                        / max(rms(y[skip:]), h.FLOOR),
                    }
                )
            log(f"jvm {label} {basis}")
        del master
    return {
        "device": "Marshall JVM410H OD1 PreampOut (ToneTwist, 44.1 kHz)",
        "protocol": "A = 0-150 s, B = 215-300 s of the same input file; skip 50 ms; "
        "floor 1e-4; source independence not established",
        "rows": rows,
    }


def table() -> str:
    """Markdown tables from the JSON, frozen baselines (baseline.json) alongside."""
    result = json.loads(OUT.read_text())
    base = json.loads((HERE / "baseline.json").read_text())["probes"]
    gains = list(h.GAIN_TAKES)
    lines = [
        "Output capacity on B (fit A, ridge from the A split)",
        "| prefilter | basis | params | "
        + " | ".join(f"G{g:g}" for g in gains)
        + " | ridge G5 |",
        "|---|---|---|" + "---|" * (len(gains) + 1),
    ]
    for name in ("hammerstein_5", "volterra_laguerre_3"):
        cap = base[name]["capacity_by_gain"]
        lines.append(
            f"| frozen | {name} | | "
            + " | ".join(f"{cap[f'{g:g}']:.3f}" for g in gains)
            + " | |"
        )
    rows = result["nolly"]["rows"] + result["nolly"]["matched_parameters"]
    keys = dict.fromkeys((r["prefilter"], r["basis"]) for r in rows)
    for pre, basis in keys:
        mine = {
            r["take"]: r
            for r in rows
            if r["prefilter"] == pre and r["basis"] == basis and r["kind"] == "output"
        }
        cells = [
            f"{mine[t]['capacity_B']:.3f}" if t in mine else ""
            for t in h.GAIN_TAKES.values()
        ]
        any_row = next(iter(mine.values()))
        ridge = mine[20]["ridge"] if 20 in mine else ""
        lines.append(
            f"| {pre} | {basis} | {any_row['params']} | "
            + " | ".join(cells)
            + f" | {ridge} |"
        )
    pairs = [(r["ref"], r["eval"]) for r in rows if r["kind"] == "difference"]
    pairs = list(dict.fromkeys(pairs))
    lines += [
        "",
        "Gain-pair difference share on B, RMS(d - q)/RMS(d)",
        "| prefilter | basis | " + " | ".join(f"{a}->{b}" for a, b in pairs) + " |",
        "|---|---|" + "---|" * len(pairs),
    ]
    for name in ("hammerstein_5", "volterra_laguerre_3"):
        share = {
            (s["ref"], s["eval"]): s["share_from_x"]
            for s in base[name]["difference_share_one_knob"]
        }
        lines.append(
            f"| frozen | {name} | "
            + " | ".join(f"{share[p]:.3f}" for p in pairs)
            + " |"
        )
    for pre, basis in keys:
        mine = {
            (r["ref"], r["eval"]): r["share_B"]
            for r in rows
            if r["prefilter"] == pre
            and r["basis"] == basis
            and r["kind"] == "difference"
        }
        if mine:
            lines.append(
                f"| {pre} | {basis} | "
                + " | ".join(f"{mine[p]:.3f}" for p in pairs)
                + " |"
            )
    a_to_b = {
        (r["prefilter"], r["basis"]): r["capacity_B"]
        for r in rows
        if r.get("take") == 20
    }
    lines += [
        "",
        "ORACLE: basis fitted in-sample on B, take 20 (G5), next to the A -> B value",
        "| prefilter | basis | params | ORACLE in-sample B | A -> B |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {o['prefilter']} | {o['basis']} | {o['params']} | "
        f"{o['capacity_B_in_sample']:.3f} | "
        f"{a_to_b[(o['prefilter'], o['basis'])]:.3f} |"
        for o in result["nolly"]["oracle"]
    ]
    h5 = base["hammerstein_5"]["capacity_by_level"]["5"]
    lines += [
        "",
        "Level bins at G5 (take 20): capacity per input quintile on B",
        "| prefilter | basis | "
        + " | ".join(f"{b['input_db'][0]:.1f} dB" for b in h5)
        + " |",
        "|---|---|" + "---|" * len(h5),
        "| frozen | hammerstein_5 | "
        + " | ".join(f"{b['capacity']:.3f}" for b in h5)
        + " |",
    ]
    for r in result["nolly"]["rows"]:
        if r.get("take") == 20:
            lines.append(
                f"| {r['prefilter']} | {r['basis']} | "
                + " | ".join(f"{b['capacity']:.3f}" for b in r["by_level_B"])
                + " |"
            )
    return "\n".join(lines + table_extra(result))


def table_extra(result: dict) -> list[str]:
    """JVM rows and checks, when present."""
    lines = []
    if "jvm" in result:
        lines += [
            "",
            "JVM410H OD1 PreampOut, capacity on B (one-file A/B split)",
            "| prefilter | basis | setting | params | capacity B | ridge |",
            "|---|---|---|---|---|---|",
        ]
        for r in result["jvm"]["rows"]:
            lines.append(
                f"| {r['prefilter']} | {r['basis']} | {r['setting']} | "
                f"{r['params']} | {r['capacity_B']:.3f} | {r['ridge']} |"
            )
    if "check" in result:
        lines += ["", "Checks", "```", json.dumps(result["check"], indent=1), "```"]
    return lines


def check(log: Callable) -> dict:
    """(1) The part Grams A1 and A2 (and all) against probes.normal_equations on the
    truncated parts of a short section (40 blocks): positive semi-definite exact
    Grams, as used by the ridge selection. (2) The scratch study's HP300 Wiener-tanh
    (features [x, tanh(a z), sign z], no linear z branch, z normalised per section)
    against the in-protocol tanh basis, take 20, ridge 1e-4, through
    harness.FeatureProbe (A -> B)."""
    x = np.asarray(h.load("x", (h.FIT,))[h.FIT])[: BLOCK * 40 + 1234]
    y = np.asarray(h.load(20, (h.FIT,))[h.FIT])[: len(x)]
    master = Master(hp300(h.FS), x, raw=False)
    names = ["x", "gmp_lag8_k1", "dvr2_0.5_8", "even_2"]
    cols = master.columns(names, 32)
    feats = master.features(x)
    u = [feats[n] / master.scales[n] for n in names]
    r = master.rhs([y])
    parts = {}
    for part, (s, e) in master.spans.items():
        gram_ref, rhs_ref = normal_equations(
            Probe(degree=4, taps=32), [v[s:e] for v in u], y[s:e]
        )
        gram = master.gram(part, cols)
        rhs = master.vector(r[part], cols)[:, 0]
        parts[part] = {
            "gram_max_rel_gap": float(
                np.max(np.abs(gram - gram_ref)) / np.max(gram_ref)
            ),
            "rhs_max_rel_gap": float(
                np.max(np.abs(rhs - rhs_ref)) / np.max(np.abs(rhs_ref))
            ),
            "min_eigenvalue_over_max_entry": float(
                np.linalg.eigvalsh(gram).min() / np.max(gram_ref)
            ),
        }
    log(f"part grams {parts}")
    x_a = np.asarray(h.load("x", (h.FIT,))[h.FIT])
    hp = hp300(h.FS)
    sigma = rms(hp(x_a))

    def tanh_bank(inputs: dict, own_sigma: bool, linear: bool) -> list[np.ndarray]:
        xs = np.asarray(inputs["x"])
        v = hp(xs)
        z = v / (rms(v) if own_sigma else sigma)
        out = [xs] + ([z] if linear else [])
        return out + [np.tanh(c * z) for c in TANH] + [np.sign(z)]

    y_b = np.asarray(h.load(20, (h.READ,))[h.READ])
    y_a = np.asarray(h.load(20, (h.FIT,))[h.FIT])
    scratch = {}
    for label, own, linear in (  # the first: out of protocol, z scaled on x_B
        ("scratch_features_per_section_rms_OUT_OF_PROTOCOL", True, False),
        ("scratch_features_rms_of_A", False, False),
        ("in_protocol_tanh_basis_with_z", False, True),
    ):
        probe = h.FeatureProbe(
            lambda i, own=own, linear=linear: tanh_bank(i, own, linear), TAPS, 1e-4
        )
        scratch[label] = h.ratio(y_b, h.fit_read(probe, y_a))
        log(f"scratch reconciliation {label} {scratch[label]:.4f}")
    return {"part_grams_vs_normal_equations": parts, "scratch_hp300_tanh": scratch}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("nolly", "jvm", "check", "table"))
    args = parser.parse_args()
    if args.command == "table":
        print(table())
        return
    start = time.time()

    def log(message: str) -> None:
        print(f"[{time.time() - start:7.0f} s] {message}", flush=True)

    result = json.loads(OUT.read_text()) if OUT.exists() else {}
    result["protocol"] = {
        "fit": "di_train (A); ridge chosen on A2 (last 1/4 of A's blocks) after a "
        "fit on A1, then refit on all of A; read di_test (B) once; 48 kHz",
        "metric": "harness.ratio: RMS(y - q)/max(RMS(y), 1e-4) after SKIP 2400 "
        "(outputs); floor 0 for differences (unrepresented / d_audio)",
        "taps": TAPS,
        "ridges": RIDGES,
        "knots_times_rms_A": KNOTS,
        "cross_knots": CROSS_KNOTS,
        "lags": LAGS,
        "bases": bases(list(bank(np.ones(16), np.ones(16), False))),
    }
    result[args.command] = {"nolly": nolly, "jvm": jvm, "check": check}[args.command](
        log
    )
    result[args.command]["runtime_s"] = time.time() - start
    OUT.write_text(json.dumps(nr.finite(result), indent=1) + "\n")
    log(f"wrote {OUT.relative_to(h.ROOT)}")


if __name__ == "__main__":
    main()
