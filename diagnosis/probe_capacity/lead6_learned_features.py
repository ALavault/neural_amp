#!/usr/bin/env python3
"""Lead 6: linear readout on frozen learned features (exploratory, CPU, 3 threads).

Status: exploratory. D-0010 excluded metrics that rely on a learned network; the user
explicitly authorised following this lead. Nothing here is a frozen probe.

Question: do the internal features of a trained amp network (VS100R arm C,
vs100r48fast_film_s42-44, last checkpoints; no Nolly-trained network exists yet), read
by a linear readout, represent a device output and the difference between two takes
better than the frozen probes (FIR 256, Hammerstein 5, Volterra-Laguerre 3)?

Features (lead6_learned_features_net.py): penultimate activations "pen" (16 ch), the
8 SSM blocks' skip outputs "blocks" (128 ch), the network output "out" (1 ch), each
section from the zero state. Knobs: "center" (gain 5, contour 5) for every take, a
knob-blind basis like a frozen probe's; "own" (the take's gain and contour), knob-
aware; "pair" (features at the reference's and at the evaluated take's knobs, side by
side) for differences. Control: the same architecture at a random initialisation
(untrained, seed 42 draw), knob-blind.

Readouts (READOUTS): static (one weight per channel) or a causal FIR of 256 taps per
channel, constant unpenalised, ridge relative to the mean Gram diagonal, exactly
harness.FeatureProbe (checked in `checks`). The features never read the target, so
every readout is linear in the target (checked).

Protocol (deviations from the frozen reading, stated):
- fit section A' = the last 30 s (4th quarter) of di_train (samples 4 320 000 to
  5 760 000 at 48 kHz), not the whole of A, to keep the CPU forward passes affordable
  (one-piece run of 1.44 M samples: about 18 s). Quarter chosen from x alone, after a
  first try on the 1st quarter: its input peak is the largest of the four (-7.7 dBFS
  VS100R, -1.0 dBFS Nolly; 1st quarter -10.3 and -5.3), and fitted on the 1st quarter
  Hammerstein 5 diverged on B (ratio 4.55 on VS100R take 1, polynomial extrapolation
  above the fit range; kept in tmp, units_first_quarter). The frozen probes are fitted
  on both A and A' for a matched comparison, and `checks.full_A` refits the readouts
  of take 1 (seed 42) on all of A;
- the ridge of each readout is chosen on A' alone: fit on its first 80 %, score on its
  last 20 % (A2), grid RIDGES, on the feature set's primary target; then refit on A'
  and used unchanged for every target of the feature set (so linearity holds);
- read on B = di_test, once per learned readout in the final protocol (B is never in
  any fit, nor in arm C's training data: dry/trainval holds di_train + di_validation
  of the train settings only). Earlier B exposures, all on take 1 (seed 42, center),
  are listed in the JSON (protocol.b_exposures_before_final_protocol): pipeline
  checks with readouts fitted on the whole of A, a smoke test under the 1st-quarter
  A', and the frozen probes under that quarter, whose B value triggered the window
  move. V is not used: it is inside arm C's trainval;
- ratios as the harness: RMS((y - q)[SKIP:]) / max(RMS(y[SKIP:]), floor), floor 1e-4
  for outputs, 0 for differences (= unrepresented / d_audio).

VS100R: input = each take's own loopback, target = its FX send, OD2, Power Dimension
off; train-setting takes had their di_train target in arm C's training data (the
features were fitted to it), the 8 combination-test takes never did. Differences
y_eval - y_ref use the reference's loopback as input, for slip-free pairs whose
loopbacks are at lag 0 on A' and B (`alignment`, which also locates the worst 1 s of
input mismatch). Nolly (harness cache): out of domain for these networks; the input
is level-matched to the VS100R loopback (one gain from x only) or left raw.
Output-referenced comparison (stage yref): FeatureProbe([measured y_ref]), 256 taps,
ridge 1e-4, on every difference; it is fed the measured reference output (labelled).

Subset (documented): VS100R output takes OUTPUT_TAKES, differences DIFF_EVALS against
take 1, "pair" features for PAIR_EVALS with seed 42 only (cut during the run: 13 to
16 min a unit under machine load), random control on RANDOM_TAKES and Nolly; Nolly
raw level and gain-matched knobs with seed 42 only. Wall time under shared load
(load average 16-33 on 32 cores, this job niced at 3 threads): frozen 25 min, learned
2 h 40, checks 9 min, yref 1 min.

Usage (repository root):
    uv run python diagnosis/probe_capacity/lead6_learned_features.py run
    uv run python diagnosis/probe_capacity/lead6_learned_features.py run --stages yref
    uv run python diagnosis/probe_capacity/lead6_learned_features.py assemble
`run` computes the missing units of the chosen stages (frozen, vs100r, nolly, checks,
yref; all by default), cached in tmp/probe_capacity/lead6/units outside the
repository, then assembles diagnosis/probe_capacity/lead6_learned_features.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as h  # noqa: E402
import lead6_learned_features_net as net  # noqa: E402

APRIME = slice(4_320_000, 5_760_000)  # A' = last 30 s (4th quarter) of di_train
SPLIT = 1_152_000  # A' = A1 (its first 80 %) + A2 (its last 20 %): ridge choice
RIDGES = (1e-8, 1e-6, 1e-4, 1e-2, 1.0)
READOUTS = {  # name: (representation, taps)
    "static_out": ("out", 1),
    "fir_out": ("out", 256),
    "static_pen": ("pen", 1),
    "fir_pen": ("pen", 256),
    "static_blocks": ("blocks", 1),
}
SEEDS = (42, 43, 44)
CENTER = (5.0, 5.0)
OUTPUT_TAKES = (1, 39, 26, 9, 37, 16, 30)  # train settings, then test (16, 30)
REF = 1
DIFF_EVALS = (39, 4, 26, 9, 22, 37)  # slip-free, loopback lag 0 against take 1
# Differences also read with "pair" features (32 / 256 channels: 13 min a unit under
# load), seed 42 only, gain pairs only: cut from (26, 9, 37) x 3 seeds during the run.
PAIR_EVALS = (26, 9)
PAIR_SEEDS = (42,)
RANDOM_TAKES = (1, 26, 9)
FROZEN = ("fir", "hammerstein_5", "volterra_laguerre_3")
FROZEN_PARAMS = {"fir": 257, "hammerstein_5": 1281, "volterra_laguerre_3": 165}
UNITS = net.CACHE / "units"
OUT = HERE / "lead6_learned_features.json"


# ---------------------------------------------------------------- targets


def _target(fit, read, kind: str, x_read=None) -> dict:
    return {"fit": fit, "read": read, "kind": kind, "x_read": x_read}


def vs_output(take: int) -> dict:
    _, y_a = net.vs100r_section(take, h.FIT)
    x_b, y_b = net.vs100r_section(take, h.READ)
    return _target(y_a[APRIME], y_b, "output", x_b)


def vs_difference(eval_take: int) -> dict:
    (_, ya), (_, yb) = (net.vs100r_section(REF, s) for s in (h.FIT, h.READ))
    (_, ea), (_, eb) = (net.vs100r_section(eval_take, s) for s in (h.FIT, h.READ))
    return _target(ea[APRIME] - ya[APRIME], eb - yb, "difference")


def nolly_output(take: int) -> dict:
    y = h.load(take)
    return _target(y[h.FIT][APRIME], y[h.READ], "output", h.load("x")[h.READ])


def nolly_difference(ref: int, eval_take: int) -> dict:
    fit = h.difference(ref, eval_take, h.FIT)[APRIME]
    return _target(fit, h.difference(ref, eval_take, h.READ), "difference")


def floor(target: dict) -> float:
    return h.FLOOR if target["kind"] == "output" else 0.0


def knobs_of(take: int) -> tuple[float, float]:
    return tuple(net.vs100r_takes()[take]["gain_contour"])


def nolly_pairs() -> list[dict]:
    return h.pairs(("one_knob",))


def nolly_targets() -> dict:
    """name -> callable building the Nolly target (gain takes, one-knob pairs)."""
    out = {}
    for t in h.GAIN_TAKES.values():
        out[f"y{t:02d}"] = lambda t=t: nolly_output(t)
    for p in nolly_pairs():
        r, e = p["ref"], p["eval"]
        out[f"d{r:02d}_{e:02d}"] = lambda r=r, e=e: nolly_difference(r, e)
    return out


# ---------------------------------------------------------------- unit specs


def _spec(net_id, x, knobs, targets, primary, linearity=None, level="raw") -> dict:
    tag = "+".join(f"{g:g}-{c:g}" for g, c in knobs)
    xname = x if isinstance(x, str) else f"x{x:02d}"
    domain = "nolly" if x == "nolly" else "vs100r"
    return {
        "key": f"{domain}_{net_id}_{xname}_k{tag}_{level}",
        "domain": domain,
        "net": net_id,
        "x": x,
        "knobs": [list(k) for k in knobs],
        "level": level,
        "targets": targets,  # name -> callable building the target
        "primary": primary,
        "linearity": linearity,
    }


def _ref_targets() -> dict:
    out = {f"y{REF:02d}": lambda: vs_output(REF), "y26": lambda: vs_output(26)}
    for e in DIFF_EVALS:
        out[f"d{REF:02d}_{e:02d}"] = lambda e=e: vs_difference(e)
    return out


def vs100r_specs() -> list[dict]:
    specs = []
    lin = ("y01", "y26", "d01_26")
    for seed in SEEDS:
        net_id = f"s{seed}"
        specs.append(_spec(net_id, REF, [CENTER], _ref_targets(), "y01", lin))
        for t in OUTPUT_TAKES[1:]:
            targets = {f"y{t:02d}": lambda t=t: vs_output(t)}
            specs.append(_spec(net_id, t, [CENTER], targets, f"y{t:02d}"))
            if knobs_of(t) != CENTER:
                specs.append(_spec(net_id, t, [knobs_of(t)], targets, f"y{t:02d}"))
        for e in PAIR_EVALS if seed in PAIR_SEEDS else ():
            targets = {
                f"d01_{e:02d}": lambda e=e: vs_difference(e),
                "y01": lambda: vs_output(REF),
                f"y{e:02d}": lambda e=e: vs_output(e),
            }
            pair = ("y01", f"y{e:02d}", f"d01_{e:02d}")
            specs.append(
                _spec(net_id, REF, [CENTER, knobs_of(e)], targets, pair[2], pair)
            )
    specs.append(_spec("rand42", REF, [CENTER], _ref_targets(), "y01", lin))
    for t in RANDOM_TAKES[1:]:
        targets = {f"y{t:02d}": lambda t=t: vs_output(t)}
        specs.append(_spec("rand42", t, [CENTER], targets, f"y{t:02d}"))
    return specs


def nolly_specs() -> list[dict]:
    lazy = nolly_targets()
    lin = ("y20", "y16", "d20_16")
    specs = [
        _spec(f"s{s}", "nolly", [CENTER], lazy, "y20", lin, "matched") for s in SEEDS
    ]
    specs.append(_spec("s42", "nolly", [CENTER], lazy, "y20", lin, "raw"))
    specs.append(_spec("rand42", "nolly", [CENTER], lazy, "y20", lin, "matched"))
    for gain, take in h.GAIN_TAKES.items():
        if gain != CENTER[0]:
            targets = {f"y{take:02d}": lazy[f"y{take:02d}"]}
            knobs = [(gain, CENTER[1])]
            specs.append(
                _spec("s42", "nolly", knobs, targets, f"y{take:02d}", None, "matched")
            )
    return specs


# ---------------------------------------------------------------- learned units


def nolly_gain() -> float:
    """Level match: RMS of the VS100R take-1 loopback on A' over the Nolly source's
    (x only, no target)."""
    vs = net.vs100r_section(REF, h.FIT)[0][APRIME]
    return h.nr.rms(np.asarray(vs)) / h.nr.rms(np.asarray(h.load("x")[h.FIT][APRIME]))


def section_inputs(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    if spec["domain"] == "vs100r":
        xa = net.vs100r_section(spec["x"], h.FIT)[0][APRIME]
        xb = net.vs100r_section(spec["x"], h.READ)[0]
        return np.asarray(xa), np.asarray(xb)
    x = h.load("x")
    gain = nolly_gain() if spec["level"] == "matched" else 1.0
    return gain * np.asarray(x[h.FIT][APRIME]), gain * np.asarray(x[h.READ])


def unit_features(model, spec: dict) -> tuple[dict, dict]:
    """Features on A' and B, channels of every knob set side by side."""
    xa, xb = section_inputs(spec)
    fit, read = {}, {}
    for knobs in spec["knobs"]:
        k = net.controls(*knobs)
        for store, x in ((fit, xa), (read, xb)):
            for name, value in net.features(model, x, k).items():
                store.setdefault(name, []).append(value)
    return (
        {n: np.concatenate(v) for n, v in fit.items()},
        {n: np.concatenate(v) for n, v in read.items()},
    )


def choose_ridge(u_fit: np.ndarray, taps: int, target: dict) -> tuple[float, dict]:
    """Ridge minimising the A2 ratio after a fit on A1 (both inside A')."""
    r = net.readout(u_fit[:, :SPLIT], taps)
    rhs = r.rhs(target["fit"][:SPLIT])
    curve = {}
    for ridge in RIDGES:
        q = r.predict(r.solve(rhs, ridge), u_fit)
        curve[ridge] = h.ratio(target["fit"][SPLIT:], q[SPLIT:], floor(target))
    return min(curve, key=curve.get), {f"{k:g}": v for k, v in curve.items()}


def read_targets(u_fit, u_read, taps, targets, ridge) -> tuple[dict, dict]:
    r = net.readout(u_fit, taps)
    out, preds = {}, {}
    for name, t in targets.items():
        theta = r.solve(r.rhs(t["fit"]), ridge)
        q = r.predict(theta, u_read)
        entry = {
            "B": h.ratio(t["read"], q, floor(t)),
            "A_in_sample": h.ratio(t["fit"], r.predict(theta, u_fit), floor(t)),
        }
        if t["x_read"] is not None:
            entry["by_level_B"] = h.level_capacity_from(t["x_read"], t["read"], q)
        out[name], preds[name] = entry, q
    return out, preds


def run_unit(model, spec: dict) -> dict:
    t0 = time.time()
    targets = {name: build() for name, build in spec["targets"].items()}
    u_fit, u_read = unit_features(model, spec)
    result = {k: v for k, v in spec.items() if k != "targets"}
    result["targets"] = sorted(targets)
    result["readouts"] = {}
    for name, (rep, taps) in READOUTS.items():
        ridge, curve = choose_ridge(u_fit[rep], taps, targets[spec["primary"]])
        read, preds = read_targets(u_fit[rep], u_read[rep], taps, targets, ridge)
        entry = {"ridge": ridge, "ridge_curve_A2": curve, "targets": read}
        entry["params"] = 1 + len(u_fit[rep]) * taps
        if spec["linearity"] is not None:
            a, b, d = (preds[k] for k in spec["linearity"])
            gap = h.nr.rms((b - a - d)[h.SKIP :]) / h.nr.rms(d[h.SKIP :])
            entry["linearity_gap"] = gap
        result["readouts"][name] = entry
    result["network_raw"] = network_raw(u_read["out"], targets, spec)
    result["seconds"] = time.time() - t0
    return result


def network_raw(out: np.ndarray, targets: dict, spec: dict) -> dict:
    """The network's own output, no readout (a trained model, not a probe): ratio to
    each output target when one knob set is used; for "pair" features, the model's
    predicted difference out(eval knobs) - out(reference knobs) against d."""
    raw = {}
    for name, t in targets.items():
        if t["kind"] == "output" and len(out) == 1:
            raw[name] = h.ratio(t["read"], out[0].astype(np.float64))
        elif t["kind"] == "difference" and len(out) == 2:
            q = out[1].astype(np.float64) - out[0].astype(np.float64)
            raw[name] = h.ratio(t["read"], q, 0.0)
    return raw


# ---------------------------------------------------------------- frozen units


def frozen_unit(inputs_a, inputs_b, target: dict, probes=FROZEN) -> dict:
    """Frozen probes fitted on A (whole di_train) and on A' (its last 30 s)."""
    out = {}
    for name in probes:
        out[name] = {}
        for label, xa, ya in (
            ("A", inputs_a, target["fit_full"]),
            ("Aprime", inputs_a[APRIME], target["fit"]),
        ):
            probe = h.FrozenProbe(name).fit({"x": np.asarray(xa)}, np.asarray(ya))
            q = probe.predict({"x": np.asarray(inputs_b)})
            entry = {"B": h.ratio(target["read"], q, floor(target))}
            if target["x_read"] is not None and name == "hammerstein_5":
                entry["by_level_B"] = h.level_capacity_from(
                    target["x_read"], target["read"], q
                )
            out[name][label] = entry
    return out


def vs100r_frozen_jobs() -> dict:
    jobs = {}
    for t in OUTPUT_TAKES:
        jobs[f"vs100r_frozen_y{t:02d}"] = (t, None)
    for e in DIFF_EVALS:
        jobs[f"vs100r_frozen_d01_{e:02d}"] = (REF, e)
    return jobs


def vs100r_frozen(take: int, eval_take: int | None) -> dict:
    xa, ya = net.vs100r_section(take, h.FIT)
    xb = net.vs100r_section(take, h.READ)[0]
    if eval_take is None:
        target = vs_output(take)
        target["fit_full"] = ya
    else:
        target = vs_difference(eval_take)
        target["fit_full"] = net.vs100r_section(eval_take, h.FIT)[1] - ya
    return frozen_unit(xa, xb, target)


def nolly_frozen(name: str) -> dict:
    """Frozen probes on Nolly fitted on A' only (whole-A values: baseline.json)."""
    x = h.load("x")
    target = nolly_targets()[name]()
    out = {}
    probes = FROZEN if name.startswith("y") else FROZEN[:2]
    for probe in probes:
        fitted = h.FrozenProbe(probe).fit(
            {"x": np.asarray(x[h.FIT][APRIME])}, target["fit"]
        )
        q = fitted.predict({"x": np.asarray(x[h.READ])})
        entry = {"B": h.ratio(target["read"], q, floor(target))}
        if target["x_read"] is not None and probe == "hammerstein_5":
            entry["by_level_B"] = h.level_capacity_from(
                target["x_read"], target["read"], q
            )
        out[probe] = {"Aprime": entry}
    return out


# ---------------------------------------------------------------- checks


def alignment() -> dict:
    """Integer lag (48 kHz, |lag| <= 24) of each evaluated take's loopback against
    take 1's, their relative RMS mismatch at lag 0, and the worst 1 s window of that
    mismatch (seconds from the section start), on A' (key di_train) and B."""
    out = {}
    for e in DIFF_EVALS:
        out[str(e)] = {}
        for section, cut in ((h.FIT, APRIME), (h.READ, slice(None))):
            x1 = np.asarray(net.vs100r_section(REF, section)[0][cut])
            xe = np.asarray(net.vs100r_section(e, section)[0][cut])
            lags = list(range(-24, 25))
            c = [np.dot(x1[30:-30], xe[30 + g : len(xe) - 30 + g]) for g in lags]
            second = [
                h.nr.rms(xe[i : i + h.FS] - x1[i : i + h.FS])
                / max(h.nr.rms(x1[i : i + h.FS]), 1e-12)
                for i in range(0, len(x1) - h.FS + 1, h.FS)
            ]
            worst = int(np.argmax(second))
            out[str(e)][section] = {
                "lag": lags[int(np.argmax(c))],
                "mismatch": h.nr.rms(xe - x1) / h.nr.rms(x1),
                "worst_second": {"start_s": worst, "mismatch": second[worst]},
            }
            if section == h.FIT:
                out[str(e)][section]["start_s_offset"] = APRIME.start / h.FS
    return out


def full_a_check(model) -> dict:
    """Take 1, seed 42, center knobs: readouts refitted on the whole of A (one-piece
    run of 5.76 M samples) with the ridges chosen on A', read on B, against A'."""
    unit = json.loads((UNITS / "vs100r_s42_x01_k5-5_raw.json").read_text())
    xa, ya = net.vs100r_section(REF, h.FIT)
    xb, _ = net.vs100r_section(REF, h.READ)
    k = net.controls(*CENTER)
    u_fit, u_read = net.features(model, xa, k), net.features(model, xb, k)
    targets = {"y01": vs_output(REF), "d01_26": vs_difference(26)}
    targets["y01"]["fit"] = ya
    targets["d01_26"]["fit"] = net.vs100r_section(26, h.FIT)[1] - ya
    out = {}
    for name, (rep, taps) in READOUTS.items():
        ridge = unit["readouts"][name]["ridge"]
        read, _ = read_targets(u_fit[rep], u_read[rep], taps, targets, ridge)
        out[name] = {
            t: {
                "B_fit_A": read[t]["B"],
                "B_fit_Aprime": unit["readouts"][name]["targets"][t]["B"],
            }
            for t in targets
        }
    return out


def featureprobe_check(model) -> dict:
    """The readouts against harness.FeatureProbe on take 1 (pen, ridge 1e-4): largest
    absolute gap of the B predictions over the prediction's RMS."""
    xa, ya = net.vs100r_section(REF, h.FIT)
    xb, _ = net.vs100r_section(REF, h.READ)
    k = net.controls(*CENTER)
    u_a = net.features(model, np.asarray(xa[APRIME]), k)["pen"]
    u_b = net.features(model, np.asarray(xb), k)["pen"]
    out = {}
    for taps in (1, 256):
        r = net.readout(u_a, taps)
        mine = r.predict(r.solve(r.rhs(ya[APRIME]), 1e-4), u_b)
        probe = h.FeatureProbe(lambda i: [f.astype(np.float64) for f in i["u"]], taps)
        ref = probe.fit({"u": u_a}, np.asarray(ya[APRIME])).predict({"u": u_b})
        out[f"taps_{taps}"] = float(np.max(np.abs(mine - ref)) / h.nr.rms(ref))
    return out


# ---------------------------------------------------------------- driver


def save(key: str, value: dict) -> None:
    UNITS.mkdir(parents=True, exist_ok=True)
    path = UNITS / f"{key}.json"
    partial = path.with_name(f".{key}.partial.json")
    partial.write_text(json.dumps(h.nr.finite(value), indent=1) + "\n")
    partial.replace(path)


def done(key: str) -> bool:
    return (UNITS / f"{key}.json").exists()


def log(text: str) -> None:
    print(time.strftime("%H:%M:%S"), text, flush=True)


def run_frozen() -> None:
    if not done("vs100r_alignment"):
        save("vs100r_alignment", alignment())
    for key, (take, eval_take) in vs100r_frozen_jobs().items():
        if not done(key):
            t0 = time.time()
            save(key, vs100r_frozen(take, eval_take))
            log(f"{key} {time.time() - t0:.0f} s")
    for name in nolly_targets():
        key = f"nolly_frozen_{name}"
        if not done(key):
            save(key, nolly_frozen(name))
            log(key)


def output_referenced(inputs_a, inputs_b, target: dict) -> dict:
    """Output-referenced comparison (harness smoke probe): FeatureProbe([y_ref]),
    256 taps, ridge 1e-4 fixed a priori, fitted on A and on A', read on B. It is fed
    the MEASURED reference output: the frozen bound's r_ref is vacuous for it."""
    out = {}
    for label, cut in (("A", slice(None)), ("Aprime", APRIME)):
        probe = h.FeatureProbe(lambda i: [np.asarray(i["y_ref"])], 256, 1e-4)
        fit = {"y_ref": inputs_a[cut]}
        probe.fit(fit, np.asarray(target["fit_full"][cut]))
        q = probe.predict({"y_ref": inputs_b})
        out[label] = {"B": h.ratio(target["read"], q, floor(target))}
    return out


def run_output_referenced() -> None:
    _, y1a = net.vs100r_section(REF, h.FIT)
    _, y1b = net.vs100r_section(REF, h.READ)
    for e in DIFF_EVALS:
        key = f"vs100r_yref_d01_{e:02d}"
        if not done(key):
            target = vs_difference(e)
            target["fit_full"] = net.vs100r_section(e, h.FIT)[1] - y1a
            save(key, output_referenced(y1a, y1b, target))
            log(key)
    for p in nolly_pairs():
        r, e = p["ref"], p["eval"]
        key = f"nolly_yref_d{r:02d}_{e:02d}"
        if not done(key):
            target = nolly_difference(r, e)
            target["fit_full"] = h.difference(r, e, h.FIT)
            y = h.load(r)
            save(key, output_referenced(y[h.FIT], y[h.READ], target))
            log(key)


def run_learned(specs: list[dict]) -> None:
    by_net: dict[str, list[dict]] = {}
    for spec in specs:
        if not done(spec["key"]):
            by_net.setdefault(spec["net"], []).append(spec)
    for net_id, todo in by_net.items():
        trained = not net_id.startswith("rand")
        seed = int(net_id.removeprefix("rand").removeprefix("s"))
        model = net.load_model(seed, trained)
        for spec in todo:
            result = run_unit(model, spec)
            save(spec["key"], result)
            log(f"{spec['key']} {result['seconds']:.0f} s")


def run_checks() -> None:
    if done("checks"):
        return
    model = net.load_model(42)
    checks = {"featureprobe_equivalence": featureprobe_check(model)}
    checks["full_A"] = full_a_check(model)
    save("checks", checks)
    log("checks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run", "assemble"))
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["frozen", "vs100r", "nolly", "checks", "yref"],
    )
    args = parser.parse_args()
    torch.set_num_threads(3)
    if args.command == "run":
        if "frozen" in args.stages:
            run_frozen()
        if "vs100r" in args.stages:
            run_learned(vs100r_specs())
        if "nolly" in args.stages:
            run_learned(nolly_specs())
        if "checks" in args.stages:
            run_checks()
        if "yref" in args.stages:
            run_output_referenced()
    import lead6_learned_features_report as report

    OUT.write_text(json.dumps(h.nr.finite(report.assemble(UNITS)), indent=1) + "\n")
    log(f"wrote {OUT.relative_to(h.ROOT)}")


if __name__ == "__main__":
    main()
