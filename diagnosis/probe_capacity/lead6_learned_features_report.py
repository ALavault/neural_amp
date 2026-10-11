"""Assembly of lead6_learned_features.json from the cached units of
lead6_learned_features.py: per-unit details, then tables (output capacity and
difference share, per seed with mean / min / max, beside the frozen probes on the same
condition), A / B gaps, linearity and checks."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import harness as h
import lead6_learned_features as l6
import lead6_learned_features_net as net

READOUTS = tuple(l6.READOUTS)


def _units(folder: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))}


def _mode(unit: dict) -> str:
    knobs = [tuple(k) for k in unit["knobs"]]
    if len(knobs) == 2:
        return "pair"
    return "center" if knobs[0] == l6.CENTER else "own"


def _stats(values: dict[str, float]) -> dict:
    v = [x for x in values.values() if x is not None]
    if not v:
        return {"per_net": values}
    return {"per_net": values, "mean": statistics.mean(v), "min": min(v), "max": max(v)}


def _selected(units: dict, domain, x, target, mode, nets, level) -> list[dict]:
    return [
        u
        for u in units.values()
        if u.get("domain") == domain
        and u.get("x") == x
        and u.get("net") in nets
        and u.get("level") == level
        and _mode(u) == mode
        and target in u["targets"]
    ]


def _learned(
    units: dict, domain: str, x, target: str, mode: str, nets, level: str = "raw"
) -> dict:
    """Readout ratios on B (and in-sample on A') of `target` over the units of
    `domain`, input x, knob mode, input level, for the networks `nets`."""
    chosen = _selected(units, domain, x, target, mode, nets, level)
    out = {}
    for name in (*READOUTS, "network_raw"):
        b, a = {}, {}
        for unit in chosen:
            if name == "network_raw":
                b[unit["net"]] = unit["network_raw"].get(target)
                continue
            entry = unit["readouts"][name]["targets"][target]
            b[unit["net"]], a[unit["net"]] = entry["B"], entry["A_in_sample"]
        if b:
            out[name] = {"B": _stats(b)}
            if a:
                out[name]["A_in_sample"] = _stats(a)
    return out


def _seeds() -> tuple[str, ...]:
    return tuple(f"s{s}" for s in l6.SEEDS)


def _frozen_vs(units: dict, target: str) -> dict:
    unit = units.get(f"vs100r_frozen_{target}", {})
    return {
        probe: {fit: v["B"] for fit, v in fits.items()} for probe, fits in unit.items()
    }


def _take_meta(order: int) -> dict:
    t = net.vs100r_takes()[order]
    keys = ("setting", "gain_contour", "split", "takes_same_pair_in_training")
    out = {k: t[k] for k in keys}
    # product_vs100r_eval labels the train settings' takes "test_seen" (their
    # di_test is unseen); "test" takes are settings never trained on.
    out["setting_in_arm_c_training"] = t["split"] != "test"
    out["slips"] = bool(t["pass"]["slipped_s"])
    return out


def vs100r_tables(units: dict) -> dict:
    seeds = _seeds()
    outputs = []
    for t in l6.OUTPUT_TAKES:
        target = f"y{t:02d}"
        row = {"take": t, **_take_meta(t), "frozen": _frozen_vs(units, target)}
        row["learned_center"] = _learned(units, "vs100r", t, target, "center", seeds)
        if tuple(_take_meta(t)["gain_contour"]) == l6.CENTER:
            row["learned_own"] = {"same_as_center": True}
        else:
            row["learned_own"] = _learned(units, "vs100r", t, target, "own", seeds)
        row["random_init_center"] = _learned(
            units, "vs100r", t, target, "center", ("rand42",)
        )
        outputs.append(row)
    differences = []
    aligned = units.get("vs100r_alignment", {})
    for e in l6.DIFF_EVALS:
        target = f"d{l6.REF:02d}_{e:02d}"
        row = {"ref": l6.REF, "eval": e, **_take_meta(e)}
        row["alignment"] = aligned.get(str(e))
        row["lag_zero"] = row["alignment"] is not None and all(
            s["lag"] == 0 for s in row["alignment"].values()
        )
        row["frozen"] = _frozen_vs(units, target)
        row["output_referenced_fir_yref_fed"] = _yref(units, f"vs100r_yref_{target}")
        x = l6.REF
        row["learned_center"] = _learned(units, "vs100r", x, target, "center", seeds)
        row["learned_pair"] = _learned(units, "vs100r", x, target, "pair", seeds)
        row["random_init_center"] = _learned(
            units, "vs100r", x, target, "center", ("rand42",)
        )
        differences.append(row)
    return {"output_capacity": outputs, "difference_share": differences}


def _yref(units: dict, key: str) -> dict | None:
    """FeatureProbe([measured y_ref]) on A and A' (labelled: fed the reference's
    measured output, the frozen bound's r_ref is vacuous for it)."""
    unit = units.get(key)
    return None if unit is None else {fit: v["B"] for fit, v in unit.items()}


def _baseline() -> dict:
    return json.loads((l6.HERE / "baseline.json").read_text())["probes"]


def _frozen_nolly(units: dict, target: str, baseline: dict, pair=None) -> dict:
    out = {}
    for probe in l6.FROZEN:
        entry = {}
        if pair is None:
            take = int(target[1:])
            entry["A"] = baseline[probe]["capacity_by_take"][str(take)]["value"]
        else:
            entry["A"] = next(
                s["share_from_x"]
                for s in baseline[probe]["difference_share_one_knob"]
                if (s["ref"], s["eval"]) == pair
            )
        aprime = units.get(f"nolly_frozen_{target}", {}).get(probe)
        if aprime is not None:
            entry["Aprime"] = aprime["Aprime"]["B"]
        out[probe] = entry
    return out


def _nolly_row(units: dict, target: str) -> dict:
    seeds = _seeds()
    row = {}
    for level in ("matched", "raw"):
        row[f"learned_center_{level}"] = _learned(
            units, "nolly", "nolly", target, "center", seeds, level
        )
    row["random_init_center_matched"] = _learned(
        units, "nolly", "nolly", target, "center", ("rand42",), "matched"
    )
    return row


def nolly_tables(units: dict) -> dict:
    baseline = _baseline()
    gains = []
    for gain, take in h.GAIN_TAKES.items():
        target = f"y{take:02d}"
        row = {
            "gain": gain,
            "take": take,
            "frozen": _frozen_nolly(units, target, baseline),
        }
        row |= _nolly_row(units, target)
        if gain == l6.CENTER[0]:
            row["learned_gain_matched"] = {"same_as_center_matched": True}
        else:
            row["learned_gain_matched"] = _learned(
                units, "nolly", "nolly", target, "own", _seeds(), "matched"
            )
        gains.append(row)
    pairs = []
    for p in h.pairs(("one_knob",)):
        target = f"d{p['ref']:02d}_{p['eval']:02d}"
        pair = (p["ref"], p["eval"])
        row = {"ref": p["ref"], "eval": p["eval"], "knob": h.knob(p)}
        row["frozen"] = _frozen_nolly(units, target, baseline, pair)
        row["output_referenced_fir_yref_fed"] = _yref(units, f"nolly_yref_{target}")
        row |= _nolly_row(units, target)
        pairs.append(row)
    return {"output_capacity_by_gain": gains, "difference_share_one_knob": pairs}


def a_b_gaps(tables: dict) -> dict:
    """Mean over networks and takes of B minus in-sample A' ratio, per readout and
    knob mode, for takes whose setting was in arm C's training data and for the
    combination-test takes (never trained on)."""
    out: dict[str, dict] = {}
    for row in tables["output_capacity"]:
        group = (
            "trained_setting" if row["setting_in_arm_c_training"] else "test_setting"
        )
        for mode in ("learned_center", "learned_own"):
            for name, entry in row[mode].items():
                if not isinstance(entry, dict) or "A_in_sample" not in entry:
                    continue
                b, a = entry["B"]["per_net"], entry["A_in_sample"]["per_net"]
                gaps = [b[k] - a[k] for k in b]
                key = f"{mode}/{name}"
                out.setdefault(key, {}).setdefault(group, []).extend(gaps)
    return {
        k: {g: {"mean": statistics.mean(v), "n": len(v)} for g, v in groups.items()}
        for k, groups in out.items()
    }


def linearity(units: dict) -> dict:
    out = {}
    for unit in units.values():
        for name, entry in unit.get("readouts", {}).items():
            gap = entry.get("linearity_gap")
            if gap is not None:
                out[name] = max(out.get(name, 0.0), gap)
    return out


def ridges(units: dict) -> dict:
    """How often each ridge of the grid was chosen, per readout."""
    out: dict[str, dict] = {}
    for unit in units.values():
        for name, entry in unit.get("readouts", {}).items():
            key = f"{entry['ridge']:g}"
            out.setdefault(name, {})[key] = out.get(name, {}).get(key, 0) + 1
    return out


def first_quarter(folder: Path) -> dict:
    """Frozen probes of VS100R take 1 fitted on the 1st quarter of di_train, read on
    B, before the fit window was moved (kept as history, not used)."""
    old = folder.parent / "units_first_quarter" / "vs100r_frozen_y01.json"
    if not old.exists():
        return {}
    data = json.loads(old.read_text())
    return {p: {k: v["B"] for k, v in f.items()} for p, f in data.items()}


def learned_units(units: dict) -> dict:
    return {k: v for k, v in units.items() if "readouts" in v}


def assemble(folder: Path) -> dict:
    units = _units(folder)
    tables = vs100r_tables(units)
    return {
        "lead": "6: linear readout on frozen learned features (VS100R arm C)",
        "status": (
            "exploratory: D-0010 excluded metrics relying on a learned network; the"
            " user explicitly authorised this lead. Not a frozen probe."
        ),
        "protocol": {
            "networks": [net.RUN.format(s) for s in l6.SEEDS],
            "checkpoint": "last",
            "fit": "A' = di_train samples 4 320 000 to 5 760 000 (last 30 s), 48 kHz",
            "ridge_choice": "A' only: fit first 1 152 000, score last 288 000",
            "ridge_grid": list(l6.RIDGES),
            "read": (
                "di_test (B); SKIP 2400; floor 1e-4 outputs, 0 differences. In the"
                " final protocol each learned readout is read once on B."
            ),
            "b_exposures_before_final_protocol": [
                "pipeline checks, VS100R take 1, seed 42, center knobs: the network"
                " output (ratio 0.238) and readouts fitted on the WHOLE of di_train"
                " (fir_pen 0.0415 at ridge 1e-4, static_pen 0.175, static_blocks 0.072)"
                " with FIR 0.724 and Hammerstein 5 0.721 (whole A);",
                "smoke test of the unit take 1 / seed 42 / center under a first-quarter"
                " A' (fir_pen 0.046 on y01, and its 7 other targets);",
                "frozen probes of take 1 fitted on the 1st quarter of di_train"
                " (history_first_quarter: Hammerstein 5 4.55, Volterra-Laguerre 3"
                " 2.13): this B value triggered the move of the fit window; the new"
                " quarter was then picked by an x-only rule (largest input peak).",
            ],
            "readouts": {
                k: {"representation": r, "taps": t} for k, (r, t) in l6.READOUTS.items()
            },
            "frozen_params": l6.FROZEN_PARAMS,
            "vs100r_output_takes": list(l6.OUTPUT_TAKES),
            "vs100r_difference_evals": list(l6.DIFF_EVALS),
            "vs100r_pair_evals": list(l6.PAIR_EVALS),
            "vs100r_pair_seeds": list(l6.PAIR_SEEDS),
            "random_init_takes": list(l6.RANDOM_TAKES),
            "nolly_gain_matched": "seed 42 only",
            "nolly_raw_level": "seed 42 only",
            "nolly_level_gain": l6.nolly_gain(),
        },
        "vs100r": {
            "alignment": units.get("vs100r_alignment"),
            "tables": tables,
            "history_first_quarter": first_quarter(folder),
        },
        "nolly": {"tables": nolly_tables(units)},
        "checks": {
            **units.get("checks", {}),
            "linearity_max_gap": linearity(units),
            "alignment_lag_zero_all_pairs": all(
                r["lag_zero"] for r in tables["difference_share"]
            ),
            "ridge_choices": ridges(units),
            "vs100r_b_minus_a_in_sample": a_b_gaps(tables),
        },
        "units": learned_units(units),
    }
