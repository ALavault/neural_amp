#!/usr/bin/env python3
"""Skeptic verification of lead 3 (RF-PA bases), exploration, CPU only.

Independent recomputation of the tanh+sign basis (raw and after a 2nd-order
Butterworth high-pass), with feature code written here (not lead3's Master/bank):
- features frozen on A: z = prefilter(x) / RMS(prefilter(x_A)); raw basis
  [z, tanh(c z) for c in 0.5..16, sign z]; high-passed basis adds the raw x branch;
  each feature scaled by its RMS on A, 256-tap causal FIR each, constant, ridge
  relative to the mean Gram diagonal (library normal_equations / regularized);
- stage `scan`: cutoffs None (raw), 100, 200, 300, 500, 800 Hz, ridges 1e-6..1e-2,
  fit on A, read V = di_validation and B = di_test for the five gain takes and the
  four gain-pair differences. Selection uses V only; B is also recorded at every
  cutoff as a VERIFICATION DIAGNOSTIC (never a selection) to size the optimism of
  the out-of-protocol 300 Hz choice;
- stage `checks`: harness.FeatureProbe parity, harness.linearity_check, the A1/A2
  ridge selection redone here, level of the high-passed input on A, V and B;
- stage `jvm`: JVM A/B slices and a repeated-content test between them.
Partial results: tmp/probe_capacity/verify/lead3_<stage>.json; `summary` merges them
into verify/lead3_verify.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.linalg import cho_factor, cho_solve
from scipy.signal import butter, sosfilt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness as h  # noqa: E402

from fssr_nam.product.probes import Probe, normal_equations, regularized  # noqa: E402

TAPS = 256
SLOPES = (0.5, 1, 2, 4, 8, 16)
RIDGES = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
CUTOFFS = (None, 100, 200, 300, 500, 800)
GAINS = {f"{g:g}": t for g, t in h.GAIN_TAKES.items()}  # 0:30 2.5:16 5:20 7.5:6 10:7
DIFFS = ((1, 6), (1, 7), (20, 16), (39, 30))  # the four gain one-knob pairs
LEAD_RIDGE = {"raw": 1e-2, "hp300": 1e-4}  # lead JSON rows, basis "tanh"
CACHE = h.CACHE / "verify"
OUT = HERE / "lead3_verify.json"


def rms(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(v))))


def label(cutoff: int | None) -> str:
    return "raw" if cutoff is None else f"hp{cutoff}"


class Basis:
    """tanh+sign features frozen on A (sigma and scales from x_A only)."""

    def __init__(self, cutoff: int | None, x_a: np.ndarray) -> None:
        self.sos = (
            None
            if cutoff is None
            else butter(2, cutoff, "highpass", fs=h.FS, output="sos")
        )
        self.sigma = rms(self.pre(x_a))
        self.scales = [max(rms(f), 1e-300) for f in self.unscaled(x_a)]

    def pre(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x if self.sos is None else sosfilt(self.sos, x)

    def unscaled(self, x: np.ndarray) -> list[np.ndarray]:
        x = np.asarray(x, dtype=float)
        z = self.pre(x) / self.sigma
        out = [] if self.sos is None else [x]
        return out + [z] + [np.tanh(c * z) for c in SLOPES] + [np.sign(z)]

    def features(self, x: np.ndarray) -> list[np.ndarray]:
        return [f / s for f, s in zip(self.unscaled(x), self.scales, strict=True)]


class Solver:
    """Normal equations of the features on one fit signal; several targets."""

    def __init__(self, u: list[np.ndarray]) -> None:
        self.k, self.n = len(u), len(u[0])
        self.size = next_fast_len(self.n + TAPS)
        self.gram, _ = normal_equations(Probe(degree=self.k, taps=TAPS), u, None)
        self.spec = [rfft(f, self.size) for f in u]

    def rhs(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        spec_y = rfft(y, self.size)
        parts = [irfft(np.conj(s) * spec_y, self.size)[:TAPS] for s in self.spec]
        return np.concatenate([[y.sum()], *parts])

    def solve(self, rhs: np.ndarray, ridge: float) -> np.ndarray:
        g = regularized(Probe(degree=self.k, taps=TAPS, ridge=ridge), self.gram)
        return cho_solve(cho_factor(g), rhs)


class Reader:
    """Predictions on one section from its (frozen) features, any theta."""

    def __init__(self, feats: list[np.ndarray]) -> None:
        self.n = len(feats[0])
        self.size = next_fast_len(self.n + TAPS)
        self.spec = [rfft(f, self.size) for f in feats]

    def predict(self, theta: np.ndarray) -> np.ndarray:
        acc = np.zeros(self.size // 2 + 1, complex)
        for k, s in enumerate(self.spec):
            acc += s * rfft(theta[1 + k * TAPS : 1 + (k + 1) * TAPS], self.size)
        return theta[0] + irfft(acc, self.size)[: self.n]


def sec(take: int | str, section: str) -> np.ndarray:
    return np.asarray(h.load(take, (section,))[section], dtype=float)


def save(name: str, data: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"lead3_{name}.json").write_text(json.dumps(data, indent=1) + "\n")


def load_stage(name: str) -> dict:
    path = CACHE / f"lead3_{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


# ---------------------------------------------------------------- stage scan


def scan(log) -> dict:
    x = {s: sec("x", s) for s in (h.FIT, h.VALID, h.READ)}
    takes = sorted(set(GAINS.values()) | {t for p in DIFFS for t in p})
    y = {t: {s: sec(t, s) for s in (h.FIT, h.VALID, h.READ)} for t in takes}
    out = load_stage("scan")
    for cutoff in CUTOFFS:
        name = label(cutoff)
        if name in out:
            log(f"{name} cached")
            continue
        start = time.time()
        basis = Basis(cutoff, x[h.FIT])
        solver = Solver(basis.features(x[h.FIT]))
        rhs = {t: solver.rhs(y[t][h.FIT]) for t in takes}
        # the difference d = y7 - y1 fitted directly (not via rhs subtraction)
        rhs_d17 = solver.rhs(y[7][h.FIT] - y[1][h.FIT])
        del solver.spec
        readers = {s: Reader(basis.features(x[s])) for s in (h.VALID, h.READ)}
        entry = {"sigma_A": basis.sigma, "by_ridge": {}}
        for ridge in RIDGES:
            theta = {t: solver.solve(rhs[t], ridge) for t in takes}
            row = {}
            for s, reader in readers.items():
                q = {t: reader.predict(theta[t]) for t in takes}
                row[s] = {
                    "output": {g: h.ratio(y[t][s], q[t]) for g, t in GAINS.items()},
                    "difference": {
                        f"{r}->{e}": h.ratio(y[e][s] - y[r][s], q[e] - q[r], 0.0)
                        for r, e in DIFFS
                    },
                }
                if (
                    s == h.READ
                    and ridge == LEAD_RIDGE.get("raw" if cutoff is None else "hp300")
                    and cutoff in (None, 300)
                ):
                    row[s]["by_level"] = {
                        g: h.level_capacity_from(x[h.READ], y[t][h.READ], q[t])
                        for g, t in GAINS.items()
                    }
                    q_d = reader.predict(solver.solve(rhs_d17, ridge))
                    gap = q[7] - q[1] - q_d
                    row[s]["linearity_1_7_own_solver"] = rms(gap[h.SKIP :]) / rms(
                        q_d[h.SKIP :]
                    )
            entry["by_ridge"][f"{ridge:g}"] = row
        out[name] = entry
        save("scan", out)
        g5 = {
            r: round(v[h.READ]["output"]["5"], 4) for r, v in entry["by_ridge"].items()
        }
        log(f"{name} B G5 by ridge {g5} ({time.time() - start:.0f} s)")
        del solver, readers
    return out


# ---------------------------------------------------------------- stage checks


def checks(log) -> dict:
    out = load_stage("checks")
    x_a, x_v, x_b = (sec("x", s) for s in (h.FIT, h.VALID, h.READ))
    basis = Basis(300, x_a)
    out["hp300_input_level_db_rel_A"] = {
        "V": 20 * np.log10(rms(basis.pre(x_v)) / basis.sigma),
        "B": 20 * np.log10(rms(basis.pre(x_b)) / basis.sigma),
        "raw_B": 20 * np.log10(rms(x_b) / rms(x_a)),
        "raw_V": 20 * np.log10(rms(x_v) / rms(x_a)),
    }
    log(f"levels {out['hp300_input_level_db_rel_A']}")
    if "feature_probe" not in out:
        # harness.FeatureProbe with these features (frozen sigma and scales: the
        # FeatureProbe recomputes the scales on A, identical to Basis.scales)
        probe = h.FeatureProbe(lambda i: basis.unscaled(i["x"]), TAPS, 1e-4)
        start = time.time()
        out["feature_probe"] = {
            "hp300_tanh_G5_ridge1e-4": h.output_capacity(probe, 20),
            "seconds": time.time() - start,
        }
        save("checks", out)
        log(f"feature probe {out['feature_probe']}")
        out["linearity_check_harness_hp300_tanh_1_7"] = h.linearity_check(probe, 1, 7)
        save("checks", out)
        log(f"linearity {out['linearity_check_harness_hp300_tanh_1_7']}")
    if "a1a2" not in out:
        out["a1a2"] = a1a2(basis, x_a, log)
        save("checks", out)
    return out


def a1a2(basis: Basis, x_a: np.ndarray, log) -> dict:
    """The lead's ridge rule redone here: fit on A1 (first 1125 blocks of 3840), A2
    the rest, each part a signal of its own (zero history), mean A2 ratio over the
    lead's nine targets (five outputs, four differences), no SKIP (as fit_basis)."""
    cut = int(0.75 * -(-len(x_a) // 3840)) * 3840
    feats = basis.features(x_a)
    solver = Solver([f[:cut] for f in feats])
    reader = Reader([f[cut:] for f in feats])
    targets = {f"G{g}": sec(t, h.FIT) for g, t in GAINS.items()}
    for r, e in DIFFS:
        targets[f"{r}->{e}"] = sec(e, h.FIT) - sec(r, h.FIT)
    rhs = {k: solver.rhs(v[:cut]) for k, v in targets.items()}
    del solver.spec
    table = {}
    for ridge in RIDGES:
        table[f"{ridge:g}"] = {
            k: rms(v[cut:] - reader.predict(solver.solve(rhs[k], ridge))) / rms(v[cut:])
            for k, v in targets.items()
        }
    mean = {r: float(np.mean(list(v.values()))) for r, v in table.items()}
    chosen = min(mean, key=mean.get)
    log(f"A1/A2 hp300 tanh mean by ridge {mean} -> {chosen}")
    return {"cut": cut, "A2_ratio": table, "mean": mean, "chosen_ridge": chosen}


# ---------------------------------------------------------------- stage jvm


def jvm(log) -> dict:
    """JVM A/B slices (real_pairs.jvm_audio) and a repeated-content test: maximum
    normalised cross-correlation of 1 s chunks of x_B against all of x_A."""
    sys.path.insert(0, str(h.ROOT / "diagnosis/probe_metrics"))
    import real_pairs
    import soundfile as sf

    path = real_pairs.JVM / "DRY/trainval/trainval.input.wav"
    info = sf.info(path)
    rate, x, _ = real_pairs.jvm_audio("PreampOut", "B050_M050_T050_G050")
    a, b = x["A"], x["B"]
    out = {
        "rate": rate,
        "file_seconds": info.frames / rate,
        "A_seconds": [0, len(a) / rate],
        "B_seconds": [215, 215 + len(b) / rate],
        "A_rms_db": 20 * np.log10(rms(a)),
        "B_rms_db": 20 * np.log10(rms(b)),
    }
    size = next_fast_len(len(a) + rate)
    spec_a = rfft(a, size)
    energy = np.concatenate(([0.0], np.cumsum(a**2)))
    chunks = []
    for start in np.linspace(0, len(b) - rate, 8).astype(int):
        c = b[start : start + rate]
        corr = irfft(spec_a * np.conj(rfft(c, size)), size)[: len(a) - rate + 1]
        local = np.sqrt(energy[rate:] - energy[:-rate])[: len(corr)]
        ncc = np.abs(corr) / np.maximum(local * np.sqrt(c @ c), 1e-30)
        k = int(np.argmax(ncc))
        chunks.append(
            {
                "B_offset_s": 215 + start / rate,
                "max_ncc_in_A": float(ncc[k]),
                "at_A_s": k / rate,
            }
        )
    out["repeat_test"] = chunks
    log(f"jvm {json.dumps(out)}")
    save("jvm", out)
    return out


# ---------------------------------------------------------------- summary


def summary(log) -> dict:
    lead = json.loads((HERE.parent / "lead3_rf_pa_bases.json").read_text())
    scan_ = load_stage("scan")
    rows = {
        (r["prefilter"], r["basis"], r.get("gain"), r.get("ref"), r.get("eval")): r
        for r in lead["nolly"]["rows"]
    }
    out: dict = {"recompute": [], "lead_file": "diagnosis/probe_capacity/"}
    for pre, ridge in LEAD_RIDGE.items():
        mine = scan_[pre if pre == "raw" else "hp300"]["by_ridge"][f"{ridge:g}"][h.READ]
        for g, t in GAINS.items():
            theirs = rows[(pre, "tanh", float(g), None, None)]["capacity_B"]
            out["recompute"].append(
                {
                    "what": f"{pre} tanh+sign output capacity B, G{g} (take {t})",
                    "lead": theirs,
                    "mine": mine["output"][g],
                    "abs_gap": abs(theirs - mine["output"][g]),
                }
            )
        for r, e in DIFFS:
            key = (pre, "tanh", None, r, e)
            if key in rows:
                theirs = rows[key]["share_B"]
                v = mine["difference"][f"{r}->{e}"]
                out["recompute"].append(
                    {
                        "what": f"{pre} tanh+sign difference share B {r}->{e}",
                        "lead": theirs,
                        "mine": v,
                        "abs_gap": abs(theirs - v),
                    }
                )
    # cutoff selection on V (fixed ridge 1e-4, and joint ridge x cutoff on V)
    sel = {}
    hp = [c for c in CUTOFFS if c is not None]
    for g in GAINS:
        fixed = {label(c): scan_[label(c)]["by_ridge"]["0.0001"] for c in CUTOFFS}
        v_fixed = {k: v[h.VALID]["output"][g] for k, v in fixed.items()}
        b_fixed = {k: v[h.READ]["output"][g] for k, v in fixed.items()}
        best_hp = min((label(c) for c in hp), key=v_fixed.get)
        joint = {
            (label(c), r): scan_[label(c)]["by_ridge"][r][h.VALID]["output"][g]
            for c in hp
            for r in scan_[label(c)]["by_ridge"]
        }
        jc, jr = min(joint, key=joint.get)
        sel[g] = {
            "V_by_cutoff_ridge1e-4": v_fixed,
            "B_by_cutoff_ridge1e-4_DIAGNOSTIC_NOT_SELECTION": b_fixed,
            "V_chosen_hp_cutoff": best_hp,
            "B_at_V_chosen": b_fixed[best_hp],
            "B_at_hp300": b_fixed["hp300"],
            "B_argmin_cutoff_DIAGNOSTIC": min((label(c) for c in hp), key=b_fixed.get),
            "joint_V_choice": {"cutoff": jc, "ridge": jr},
            "B_at_joint_V_choice": scan_[jc]["by_ridge"][jr][h.READ]["output"][g],
        }
    common = {
        label(c): float(
            np.mean(
                [
                    scan_[label(c)]["by_ridge"]["0.0001"][h.VALID]["output"][g]
                    for g in GAINS
                ]
            )
        )
        for c in hp
    }
    cc = min(common, key=common.get)
    sel["common_cutoff_mean_V_over_gains"] = {
        "mean_V": common,
        "chosen": cc,
        "B_by_gain_at_chosen": {
            g: scan_[cc]["by_ridge"]["0.0001"][h.READ]["output"][g] for g in GAINS
        },
    }
    out["cutoff_selection_on_V"] = sel
    out["derived_from_lead_json"] = derived(lead)
    out["checks"] = load_stage("checks")
    out["jvm"] = load_stage("jvm")
    out["scan"] = scan_
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    log(f"wrote {OUT.relative_to(h.ROOT)}")
    return out


def derived(lead: dict) -> dict:
    """Spreads of the level bins (max - min over input quintiles), RF structure over
    tanh+sign, all vs union, matched-parameter rows vs frozen H5: from the lead's
    JSON and baseline.json (no recomputation)."""
    base = json.loads((HERE.parent / "baseline.json").read_text())["probes"]
    h5 = base["hammerstein_5"]
    out: dict = {"level_spread": {}, "tanh_minus_union": {}, "all_minus_union": {}}
    for g in GAINS:
        caps = [b["capacity"] for b in h5["capacity_by_level"][g]]
        out["level_spread"].setdefault("frozen_h5", {})[g] = max(caps) - min(caps)
    rows = lead["nolly"]["rows"]
    for r in rows:
        if r["kind"] != "output" or r["prefilter"] == "bla_take":
            continue
        g = f"{r['gain']:g}"
        caps = [b["capacity"] for b in r["by_level_B"]]
        key = f"{r['prefilter']}_{r['basis']}"
        out["level_spread"].setdefault(key, {})[g] = max(caps) - min(caps)
    value = {
        (r["prefilter"], r["basis"], r.get("gain"), r.get("ref"), r.get("eval")): r.get(
            "capacity_B", r.get("share_B")
        )
        for r in rows
    }
    for (pre, basis, gain, ref, ev), v in value.items():
        name = f"G{gain:g}" if gain is not None else f"{ref}->{ev}"
        if basis == "union" and pre in ("raw", "hp300"):
            out["tanh_minus_union"][f"{pre}_{name}"] = (
                value[(pre, "tanh", gain, ref, ev)] - v
            )
            out["all_minus_union"][f"{pre}_{name}"] = (
                value[(pre, "all", gain, ref, ev)] - v
            )
    oracle = {
        (o["prefilter"], o["basis"]): o["capacity_B_in_sample"]
        for o in lead["nolly"]["oracle"]
    }
    out["oracle_hp300_tanh_minus_union"] = (
        oracle[("hp300", "tanh")] - oracle[("hp300", "union")]
    )
    out["matched_minus_frozen_h5"] = {
        f"{r['prefilter']}_G{r['gain']:g}": r["capacity_B"]
        - h5["capacity_by_gain"][f"{r['gain']:g}"]
        for r in lead["nolly"]["matched_parameters"]
        if r["kind"] == "output"
    }
    return out


def main() -> None:
    stage = sys.argv[1]
    start = time.time()

    def log(message: str) -> None:
        print(f"[{time.time() - start:6.0f} s] {message}", flush=True)

    {"scan": scan, "checks": checks, "jvm": jvm, "summary": summary}[stage](log)


if __name__ == "__main__":
    main()
