#!/usr/bin/env python3
"""Shared measurement harness of the probe-capacity studies (exploration, CPU only).

Question shared by the studies: which probe represents the Nolly renders' extreme
nonlinearity? The frozen Nolly reading (diagnosis/probe_metrics/NOLLY_READING.md,
nolly_reading.json) refused H2 at every gain: Hammerstein 5 and Volterra-Laguerre 3
leave 0.48 (gain 0) to 0.80 (gains 2.5-7.5) of the output unexplained. Starting point:
the scratch study capacity_split.py (session scratchpad), moved here. Baselines to
beat: baseline.json (frozen probes, written by `harness.py baseline`).

Protocol, identical to the frozen reading and reused from it (nolly_reading.py):
- every section cut from the 192 kHz file at its layout offset, no latency advance,
  then decimated by 4 on its own (`nr.dec`), float64, at 48 kHz;
- a probe is fitted on section A = di_train (5.76 M samples) and read on
  B = di_test (1.44 M samples); the first SKIP = 2400 samples of a read section are
  unscored (zero-state start); the floor 1e-4 of `Protocol` bounds the denominator
  of the output capacity.
- B never enters a fit. Select hyper-parameters on V = di_validation (read=VALID; an
  addition, the frozen reading never used V), then report B once. Oracles (anything
  fitted on B) must be labelled as such.
- Takes: settings.csv order 1..42; ROWS[take] gives G, B, M, T, P, cab, block. Gate 4
  (marker lags) excluded EXCLUDED = 9, 12, 13, 18, 23, 24, 38, 41 from the frozen
  reading: usable for capacity studies, but say so when used.

Cache: tmp/probe_capacity next to the repository (outside it), built once by
`harness.py build`: sections48k/<tag>/<section>.npy, tag "x" (source) or "take_NN";
loaded memory-mapped (read-only) by `load`. Frozen-probe predictions are cached in
frozen_pred/<probe>/ by `frozen_prediction` (keyed by name, take and section only:
delete frozen_pred/ if the sections are ever rebuilt).

Probe interface (`CapacityProbe`): fit(inputs_A, target_A) -> self and
predict(inputs_B) -> array, with inputs = {"x": source section, "y_ref": reference
take's output on the same section (only when a reference is given)}. A probe whose
features are fixed before seeing the target (FrozenProbe, FeatureProbe) is linear in
the target, so fit(y_eval) - fit(y_ref) = fit(y_eval - y_ref): the frozen bound and
decomposition hold. `linearity_check` measures it for any probe.

Use from the repository root (uv run python), CPU, at most 3 threads:
    sys.path.insert(0, "diagnosis/probe_capacity"); import harness as h
    h.output_capacity(h.FrozenProbe("hammerstein_5"), 20)    # 0.8026, about 8 s
    h.difference_share(h.FeatureProbe(lambda i: [i["y_ref"]]), 1, 4)
    uv run python diagnosis/probe_capacity/harness.py build     # cache, 3 processes
    uv run python diagnosis/probe_capacity/harness.py baseline  # -> baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Protocol as Interface

import numpy as np
from scipy.linalg import solve
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "diagnosis/probe_metrics"))
import nolly_reading as nr  # noqa: E402
from calibrate import frozen_probes  # noqa: E402

from fssr_nam.product.fidelity import Protocol  # noqa: E402
from fssr_nam.product.probes import (  # noqa: E402
    Probe,
    fit_probe,
    normal_equations,
    probe_contributions,
    regularized,
)

FS, SKIP = nr.FS, nr.SKIP  # 48 kHz; 2400 unscored samples at the start of B
FLOOR = Protocol.__dataclass_fields__["floor"].default  # 1e-4
FIT, VALID, READ = "di_train", "di_validation", "di_test"  # A, V, B
SECTIONS = (
    FIT,
    VALID,
    READ,
    "pink_noise_-30dBFS_rms",
    *nr.MULTISINES,  # 12: multisine_{-30,-24,-18}dBFS_rms_r{1..4}
    *nr.STEPS,  # 3: steps_{-30,-24,-18}dBFS_rms
)
RENDERS = ROOT / "datasets/raw/internal/capture_v2/nolly"
SOURCE = ROOT / nr.LAYOUT["file"]
CACHE = ROOT.parent / "tmp" / "probe_capacity"
WINDOW = round(0.05 * FS)  # level-stratified capacity: 50 ms input windows

READING = json.loads((ROOT / "diagnosis/probe_metrics/nolly_reading.json").read_text())
ROWS = nr.takes()  # order -> settings.csv row (block, G, B, M, T, P, cab, file ...)
_GATE = {int(k): g["pass"] for k, g in READING["gates"]["takes"].items()}
USABLE = tuple(sorted(o for o, ok in _GATE.items() if ok))  # 34 takes
EXCLUDED = tuple(sorted(o for o, ok in _GATE.items() if not ok))  # gate 4 (8 takes)
# H2 ladder {gain: take}: {0: 30, 2.5: 16, 5: 20, 7.5: 6, 10: 7}
GAIN_TAKES = dict(sorted((nr.gain_takes(ROWS) | {5.0: 20}).items()))
FROZEN = frozen_probes() | frozen_probes("v2")  # fir, hammerstein_3/5, v-l 2/3


def tag(take: int | str) -> str:
    return "x" if take == "x" else f"take_{int(take):02d}"


# ---------------------------------------------------------------- cache of sections


def _build_one(job: tuple[str, Path]) -> str:
    name, path = job
    folder = CACHE / "sections48k" / name
    folder.mkdir(parents=True, exist_ok=True)
    for section, signal in nr.read_sections(path, SECTIONS).items():
        partial = folder / f".{section}.{uuid.uuid4().hex}.partial.npy"
        np.save(partial, nr.dec(signal))
        partial.replace(folder / f"{section}.npy")
    return name


def build(jobs: int = 3) -> None:
    """Decimated sections of the source and of every render, skipped when present."""
    todo = [("x", SOURCE)] + [
        (tag(o), RENDERS / row["file"]) for o, row in sorted(ROWS.items())
    ]
    todo = [
        (name, path)
        for name, path in todo
        if not all(
            (CACHE / "sections48k" / name / f"{s}.npy").exists() for s in SECTIONS
        )
    ]
    with ProcessPoolExecutor(jobs) as pool:
        for name in pool.map(_build_one, todo):
            print(name, flush=True)


def load(take: int | str, names=(FIT, READ)) -> dict[str, np.ndarray]:
    """Cached 48 kHz sections of the source ("x") or of a take (settings.csv order),
    read-only memory maps, float64."""
    folder = CACHE / "sections48k" / tag(take)
    return {s: np.load(folder / f"{s}.npy", mmap_mode="r") for s in names}


def section_inputs(section: str, ref: int | None = None) -> dict[str, np.ndarray]:
    """Probe inputs on one section: x, and y_ref when a reference take is given."""
    out = {"x": load("x", (section,))[section]}
    if ref is not None:
        out["y_ref"] = load(ref, (section,))[section]
    return out


# ---------------------------------------------------------------- probes


class CapacityProbe(Interface):
    """fit on section A (inputs_A, target_A), predict on another section (inputs).
    The target on the read section is never passed to the probe."""

    def fit(
        self, inputs: dict[str, np.ndarray], target: np.ndarray
    ) -> CapacityProbe: ...

    def predict(self, inputs: dict[str, np.ndarray]) -> np.ndarray: ...


class FrozenProbe:
    """One of the frozen probes (FROZEN: fir = FIR 256, hammerstein_3, hammerstein_5,
    volterra_laguerre_2, volterra_laguerre_3, hyper-parameters of dev_select.json and
    dev_select_v2.json), fitted and rendered by the library exactly as analyze_pair
    does. Uses inputs["x"] only."""

    def __init__(self, name: str) -> None:
        self.name, self.spec = name, FROZEN[name]

    def fit(self, inputs: dict[str, np.ndarray], target: np.ndarray) -> FrozenProbe:
        self.result = fit_probe(self.spec, np.asarray(inputs["x"]), np.asarray(target))
        return self

    def predict(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        return sum(probe_contributions(self.result, np.asarray(inputs["x"])).values())


class FeatureProbe:
    """q = c + sum_k h_k * (f_k / s_k): each feature f_k through its own causal FIR of
    `taps`, a constant, a ridge relative to the mean Gram diagonal (the library's fit,
    as in the scratch study). `features(inputs) -> list of arrays` must not depend on
    the target; s_k is the RMS of f_k on the fit section. Then theta is linear in the
    target. features = [x**k for k in 1..5], taps 256, ridge 1e-4 is Hammerstein 5."""

    def __init__(
        self,
        features: Callable[[dict[str, np.ndarray]], list[np.ndarray]],
        taps: int = 256,
        ridge: float = 1e-4,
    ) -> None:
        self.features, self.taps, self.ridge = features, taps, ridge

    def fit(self, inputs: dict[str, np.ndarray], target: np.ndarray) -> FeatureProbe:
        u = self.features(inputs)
        self.scales = [max(nr.rms(f), 1e-300) for f in u]
        probe = Probe(degree=len(u), taps=self.taps, ridge=self.ridge)
        gram, rhs = normal_equations(
            probe, [f / s for f, s in zip(u, self.scales, strict=True)], target
        )
        self.theta = solve(regularized(probe, gram), rhs, assume_a="pos")
        return self

    def predict(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        u, m = self.features(inputs), self.taps
        out = np.full(len(u[0]), self.theta[0])
        for k, (f, s) in enumerate(zip(u, self.scales, strict=True)):
            out += fftconvolve(f / s, self.theta[1 + k * m : 1 + (k + 1) * m])[: len(f)]
        return out


# ---------------------------------------------------------------- metrics


def ratio(y: np.ndarray, q: np.ndarray, floor: float = FLOOR) -> float:
    """RMS((y - q)[SKIP:]) / max(RMS(y[SKIP:]), floor). With the default floor it is
    capacity_eval of the frozen reading (r / RMS of the output's own samples); the
    difference share uses floor 0, as unrepresented / d_audio does."""
    return nr.rms((y - q)[SKIP:]) / max(nr.rms(y[SKIP:]), floor)


def fit_read(
    probe: CapacityProbe,
    target_a: np.ndarray,
    ref: int | None = None,
    fit: str = FIT,
    read: str = READ,
) -> np.ndarray:
    """Fit the probe on (section_inputs(fit, ref), target_a); predict on `read`."""
    probe.fit(section_inputs(fit, ref), np.asarray(target_a))
    return np.asarray(probe.predict(section_inputs(read, ref)))


def output_capacity(
    probe: CapacityProbe,
    take: int,
    ref: int | None = None,
    fit: str = FIT,
    read: str = READ,
) -> float:
    """Residual / RMS of the take's own output on `read` after fitting on `fit`; with
    ref, the probe also receives that take's output as inputs["y_ref"]."""
    q = fit_read(probe, load(take, (fit,))[fit], ref, fit, read)
    return ratio(load(take, (read,))[read], q)


def level_capacity_from(
    x: np.ndarray, y: np.ndarray, q: np.ndarray, bins: int = 5
) -> list[dict]:
    """Residual / RMS of y per bin of short-term input level: non-overlapping 50 ms
    windows of x after SKIP (last partial window dropped), equal-count bins on the
    quantiles of their RMS in dBFS. The edges depend on x only, hence are common to
    every take on a given section."""
    count = (len(x) - SKIP) // WINDOW
    frame = slice(SKIP, SKIP + count * WINDOW)
    xs, ys, rs = (np.asarray(v[frame]).reshape(count, WINDOW) for v in (x, y, y - q))
    level = 10 * np.log10(np.maximum(np.mean(xs**2, axis=1), 1e-30))
    edges = np.quantile(level, np.linspace(0, 1, bins + 1))
    which = np.searchsorted(edges[1:-1], level, side="right")
    out = []
    for b in range(bins):
        pick = which == b
        y_rms = nr.rms(ys[pick])
        out.append(
            {
                "input_db": [float(edges[b]), float(edges[b + 1])],
                "windows": int(pick.sum()),
                "output_db": nr.db(y_rms),
                "capacity": nr.rms(rs[pick]) / max(y_rms, FLOOR),
            }
        )
    return out


def level_capacity(
    probe: CapacityProbe,
    take: int,
    ref: int | None = None,
    bins: int = 5,
    fit: str = FIT,
    read: str = READ,
) -> list[dict]:
    """output_capacity stratified by short-term input level (level_capacity_from)."""
    q = fit_read(probe, load(take, (fit,))[fit], ref, fit, read)
    return level_capacity_from(
        section_inputs(read)["x"], load(take, (read,))[read], q, bins=bins
    )


def difference(ref: int, eval_take: int, section: str) -> np.ndarray:
    """d = y_eval - y_ref on one section."""
    return load(eval_take, (section,))[section] - load(ref, (section,))[section]


def difference_share(
    probe: CapacityProbe,
    ref: int,
    eval_take: int,
    fit: str = FIT,
    read: str = READ,
) -> float:
    """RMS(d_B - q_B) / RMS(d_B), d = y_eval - y_ref, probe fitted on d_A with inputs
    x and y_ref (the reference take's output). For a probe linear in the target this
    is the frozen reading's unrepresented / d_audio."""
    q = fit_read(probe, difference(ref, eval_take, fit), ref, fit, read)
    return ratio(difference(ref, eval_take, read), q, floor=0.0)


def linearity_check(
    probe: CapacityProbe,
    ref: int,
    eval_take: int,
    fit: str = FIT,
    read: str = READ,
) -> float:
    """RMS(fit(y_eval) - fit(y_ref) - fit(d)) / RMS(fit(d)) on read after SKIP, the
    three fits on the same inputs (x, y_ref); about 1e-12 for a probe linear in y."""
    a = {t: load(t, (fit,))[fit] for t in (ref, eval_take)}
    q_eval = fit_read(probe, a[eval_take], ref, fit, read)
    q_ref = fit_read(probe, a[ref], ref, fit, read)
    q_d = fit_read(probe, a[eval_take] - a[ref], ref, fit, read)
    return nr.rms((q_eval - q_ref - q_d)[SKIP:]) / nr.rms(q_d[SKIP:])


def frozen_prediction(
    name: str, take: int, ref: int | None = None, read: str = READ
) -> np.ndarray:
    """Prediction on `read` of the frozen probe `name` fitted on A to the take's
    output (ref None) or to the difference take - ref; cached in frozen_pred/."""
    label = tag(take) if ref is None else f"diff_{int(ref):02d}_{int(take):02d}"
    path = CACHE / "frozen_pred" / name / f"{label}_{read}.npy"
    if not path.exists():
        target = load(take, (FIT,))[FIT] if ref is None else difference(ref, take, FIT)
        q = fit_read(FrozenProbe(name), target, None, FIT, read)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.partial.npy")
        np.save(partial, q)
        partial.replace(path)
    return np.load(path, mmap_mode="r")


# ---------------------------------------------------------------- pairs


def pairs(kinds: tuple[str, ...] | None = None, excluded: bool = False) -> list[dict]:
    """(ref, eval, kind) pairs exactly as nolly_reading.json lists them (kinds repeat,
    one_knob, combination_test, combination_train, cab), "excluded": False. With
    excluded=True, also the pairs of nolly_reading.pairs() that involve a take of
    EXCLUDED (gate 4 failed), "excluded": True: usable for capacity studies only if
    the use is reported."""
    out = [
        {"ref": p["ref"], "eval": p["eval"], "kind": p["kind"], "excluded": False}
        for p in READING["pairs"]
    ]
    if excluded:
        out += [
            {"ref": r, "eval": e, "kind": k, "excluded": True}
            for r, e, k in nr.pairs(ROWS)
            if r in EXCLUDED or e in EXCLUDED
        ]
    return [p for p in out if kinds is None or p["kind"] in kinds]


def knob(pair: dict) -> str:
    """The knob a one-knob pair moves (G, B, M, T or P)."""
    r, e = ROWS[pair["ref"]], ROWS[pair["eval"]]
    (moved,) = [k for k in "GBMTP" if float(r[k]) != float(e[k])]
    return moved


# ---------------------------------------------------------------- baseline


def _baseline_job(job: tuple[str, int, int | None]) -> dict:
    name, take, ref = job
    q = frozen_prediction(name, take, ref)
    y = load(take)[READ] if ref is None else difference(ref, take, READ)
    floor = FLOOR if ref is None else 0.0
    out = {"probe": name, "take": take, "ref": ref, "value": ratio(y, q, floor)}
    if ref is None and take in GAIN_TAKES.values():
        out["by_level"] = level_capacity_from(section_inputs(READ)["x"], y, q)
    return out


def _frozen_values(name: str) -> tuple[dict, dict]:
    """capacity_eval (and capacity_ref of the anchors) and unrepresented / d_audio
    per pair, as nolly_reading.json reports them on di_test."""
    capacity, share = {}, {}
    for p in READING["pairs"]:
        seg = p["probes"][name][READ]
        capacity[p["eval"]] = seg["capacity_eval"]
        capacity.setdefault(p["ref"], seg["capacity_ref"])
        share[(p["ref"], p["eval"])] = seg["unrepresented"] / seg["d_audio_norm"]
    return capacity, share


def _linearity(name: str, ref: int, take: int) -> float:
    q_d = frozen_prediction(name, take, ref)
    gap = frozen_prediction(name, take) - frozen_prediction(name, ref) - q_d
    return nr.rms(gap[SKIP:]) / nr.rms(q_d[SKIP:])


def baseline(jobs: int = 3) -> dict:
    """Frozen probes: output capacity of every take (excluded ones flagged), level-
    stratified at the 5 H2 gains, difference share from x for every one-knob pair,
    linearity in y, and the instrument check against nolly_reading.json."""
    one_knob = pairs(("one_knob",), excluded=True)
    work = [(n, t, None) for n in FROZEN for t in sorted(ROWS)]
    work += [(n, p["eval"], p["ref"]) for n in FROZEN for p in one_knob]
    with ProcessPoolExecutor(jobs) as pool:
        done = list(pool.map(_baseline_job, work))
    result = {
        "protocol": "fit di_train, read di_test, 48 kHz, SKIP 2400, floor 1e-4",
        "excluded_takes_gate4": list(EXCLUDED),
        "gain_takes": {f"{g:g}": t for g, t in GAIN_TAKES.items()},
        "probes": {},
    }
    check = {"capacity_max_abs_gap": 0.0, "share_max_abs_gap": 0.0, "linearity": 0.0}
    for name in FROZEN:
        frozen_capacity, frozen_share = _frozen_values(name)
        mine = [d for d in done if d["probe"] == name]
        takes = {d["take"]: d for d in mine if d["ref"] is None}
        diffs = {(d["ref"], d["take"]): d for d in mine if d["ref"] is not None}
        entry = result["probes"][name] = {
            "version": FROZEN[name].version(),
            "capacity_by_gain": {
                f"{g:g}": takes[t]["value"] for g, t in GAIN_TAKES.items()
            },
            "capacity_by_level": {
                f"{g:g}": takes[t]["by_level"] for g, t in GAIN_TAKES.items()
            },
            "capacity_by_take": {
                str(t): {"value": d["value"], "excluded": t in EXCLUDED}
                for t, d in sorted(takes.items())
            },
            "difference_share_one_knob": [],
        }
        for p in one_knob:
            key = (p["ref"], p["eval"])
            entry["difference_share_one_knob"].append(
                {
                    **p,
                    "knob": knob(p),
                    "setting": ROWS[p["eval"]]["setting"],
                    "share_from_x": diffs[key]["value"],
                    "frozen_share": frozen_share.get(key),
                    "linearity_gap": _linearity(name, *key),
                }
            )
        gaps = [abs(takes[t]["value"] - v) for t, v in frozen_capacity.items()]
        shares = [
            abs(s["share_from_x"] - s["frozen_share"])
            for s in entry["difference_share_one_knob"]
            if s["frozen_share"] is not None
        ]
        check["capacity_max_abs_gap"] = max(check["capacity_max_abs_gap"], *gaps)
        check["share_max_abs_gap"] = max(check["share_max_abs_gap"], *shares)
        check["linearity"] = max(
            check["linearity"],
            *(s["linearity_gap"] for s in entry["difference_share_one_knob"]),
        )
    h2 = READING["readings"]["H2_capacity"]
    check["H2"] = {
        name: {
            "frozen": h2[name]["by_gain"],
            "harness": result["probes"][name]["capacity_by_gain"],
            "max_abs_gap": max(
                abs(h2[name]["by_gain"][g] - v)
                for g, v in result["probes"][name]["capacity_by_gain"].items()
            ),
        }
        for name in nr.CAPACITY_PROBES
    }
    check["pass_H2"] = all(v["max_abs_gap"] <= 0.005 for v in check["H2"].values())
    check["pass"] = (
        check["pass_H2"]
        and check["capacity_max_abs_gap"] <= 0.005
        and check["share_max_abs_gap"] <= 0.005
        and check["linearity"] <= 1e-9
    )
    result["instrument_check"] = check
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("build", "baseline"))
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if args.command == "build":
        build(args.jobs)
        return
    result = baseline(args.jobs)
    out = HERE / "baseline.json"
    out.write_text(json.dumps(nr.finite(result), indent=1) + "\n")
    print(json.dumps(result["instrument_check"], indent=1))


if __name__ == "__main__":
    main()
