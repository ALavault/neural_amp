#!/usr/bin/env python3
"""Numbers of the P4 dossier (probes as fidelity metrics), from the campaign outputs.

Inputs: diagnosis/probe_metrics/ (report_numbers.json, table_test.csv, dev_select*.json,
dynamics_verdict.json, decomposition_*.json, gap_checks_*.json, real_pairs_*.csv,
real_extra_*.csv, sweep_integrity.json, cost*.json) and two frozen protocol constants
of src/fssr_nam/product/fidelity.py. Output: numbers.json next to this file, through
paper/numbers_lib.py. No probe is fitted and no model is run: medians, ratios, exact
Mann-Whitney tests and Spearman correlations of existing scores only. CPU, seconds.

    uv run python paper/p4_probe_metrics/numbers.py
"""

import os
import sys

# This file is named numbers.py: drop its directory from sys.path so that the standard
# library module `numbers` (imported by numpy and scipy) is not shadowed, and import
# paper/numbers_lib.py instead.
sys.path[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)

import csv  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

from numbers_lib import ROOT, Numbers, fr, load, sci  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
P = "diagnosis/probe_metrics/"
RN = P + "report_numbers.json"
TABLE = P + "table_test.csv"
FIDELITY = "src/fssr_nam/product/fidelity.py"
PROBES = {
    "fir": "fir",
    "h3": "hammerstein_3",
    "h5": "hammerstein_5",
    "vl2": "volterra_laguerre_2",
    "vl3": "volterra_laguerre_3",
}
DEFECTS = {
    "gain": "gain",
    "color": "coloration_db",
    "delay": "delay_samples",
    "allpass": "allpass",
    "redistrib": "redistribution_bias",
    "prefilter": "prefilter_hz",
    "release": "release_ms",
    "cubic": "cubic",
    "quad": "quadratic",
    "cross": "cross_term",
    "u7": "order_beyond_degree",
    "echo": "echo_beyond_memory",
    "noise": "noise_db",
}
FAMILIES = ("di", "multisine", "noise", "plucks", "sweep", "two_tone")

N = Numbers()


def csv_rows(path: str) -> list[dict]:
    return list(csv.DictReader((ROOT / path).open()))


def p_text(p: float) -> str:
    return fr(p, 2) if p >= 0.01 else sci(p, 1)


def sig(x: float, n: int = 2) -> str:
    """Fixed-point text with n significant digits (costs spanning several decades)."""
    return fr(x, max(0, n - 1 - math.floor(math.log10(abs(x)))))


def auc(pos: list[float], neg: list[float]) -> tuple[float, float]:
    """AUC and exact one-sided p, positives above negatives (as report_numbers.py)."""
    test = mannwhitneyu(pos, neg, alternative="greater", method="exact")
    return float(test.statistic) / (len(pos) * len(neg)), float(test.pvalue)


def protocol() -> None:
    dev = load(P + "dev_select.json")
    dev2 = load(P + "dev_select_v2.json")
    src, src2 = P + "dev_select.json", P + "dev_select_v2.json"
    N.add("dev_fir_ridge", dev["fir"]["ridge"], src, text=sci(dev["fir"]["ridge"], 0))
    N.add(
        "dev_h_ridge",
        dev["hammerstein_3"]["ridge"],
        src,
        text=sci(dev["hammerstein_3"]["ridge"], 0),
    )
    for short, name in (
        ("fir", "fir"),
        ("h3", "hammerstein_3"),
        ("h5", "hammerstein_5"),
    ):
        taps = dev[name]["taps"]
        N.add(f"dev_{short}_taps", taps, src)
        ridge = str(dev[name]["ridge"])
        by_ridge = dev[name]["by_ridge"]
        chosen = by_ridge[ridge]["median_ratio"]
        N.add(f"dev_{short}_capacity", chosen, src, 3)
        N.add(
            f"dev_{short}_condition",
            by_ridge[ridge]["median_condition"],
            src,
            text=sci(by_ridge[ridge]["median_condition"], 1),
        )
    N.add("dev_fir_capacity_128", dev["fir"]["by_taps"]["128"]["median_ratio"], src, 3)
    for short in ("vl2", "vl3"):
        d = dev2[PROBES[short]]
        N.add(f"dev_{short}_size", d["size"], src2)
        N.add(f"dev_{short}_pole", d["pole"], src2, 2)
        N.add(f"dev_{short}_ridge", d["ridge"], src2, text=sci(d["ridge"], 0))
        capacity = d["by_ridge"][str(d["ridge"])]
        N.add(f"dev_{short}_capacity", capacity, src2, 3)
    wh = dev2["wiener_hammerstein"]
    N.add("wh_dev_capacity", wh["median_capacity"], src2, 3)
    N.add("wh_dev_spread", wh["median_start_spread"], src2, 3)
    N.add("wh_dev_threshold", 0.5 * wh["median_capacity"], src2, 3)
    nesting = dev2["nesting"]
    outside = nesting["hammerstein_outside_volterra"]
    N.add("nest_h_out_vl_min", min(outside), src2, 2)
    N.add("nest_h_out_vl_max", max(outside), src2, 2)
    N.add(
        "nest_vl_out_h_median", nesting["volterra_outside_hammerstein_median"], src2, 2
    )
    N.add("nest_vl_out_h_max", nesting["volterra_outside_hammerstein_max"], src2, 2)
    code = (ROOT / FIDELITY).read_text()
    warning = float(re.search(r"capacity_warning: float = ([0-9.]+)", code).group(1))
    cancelling = float(re.search(r"^CANCELLING = ([0-9.]+)", code, re.M).group(1))
    N.add("proto_capacity_warning", warning, FIDELITY, 1)
    N.add("proto_cancelling", cancelling, FIDELITY, 1)


def bench_counts() -> None:
    rn = load(RN)
    rows = csv_rows(TABLE)
    N.add("bench_defects", len({r["defect"] for r in rows}), TABLE)
    N.add("bench_families", len({r["family"] for r in rows}), TABLE)
    bound = rn["bound"]
    N.add("math_bound_violations", sum(v["violations"] for v in bound.values()), RN)
    N.add("math_bound_rows_total", sum(v["rows"] for v in bound.values()), RN)
    N.add("math_bound_rows_fir", bound["fir"]["rows"], RN)
    N.add("math_bound_rows_vl3", bound["volterra_laguerre_3"]["rows"], RN)
    N.add("math_bound_rows_wh", bound["wiener_hammerstein"]["rows"], RN)
    readings = rn["readings_strongest_middle"]
    for short, name in (
        ("fir", "fir"),
        ("h3", "hammerstein_3"),
        ("h5", "hammerstein_5"),
        ("vl2", "volterra_laguerre_2"),
        ("vl3", "volterra_laguerre_3"),
        ("esr", "esr"),
        ("mrstft", "mrstft"),
        ("imd", "imd"),
        ("harm", "harmonic_total"),
        ("mag", "magnitude"),
        ("phase", "phase"),
        ("env", "envelope"),
    ):
        counts = readings[name]
        N.add(f"det_{short}_yes", counts.get("détecté", 0), RN)
        N.add(f"det_{short}_lim", counts.get("limite", 0), RN)
        N.add(f"det_{short}_no", counts.get("non détecté", 0), RN)
    weakest = rn["weakest_detected"]
    for defect in (
        "cubic",
        "quadratic",
        "order_beyond_degree",
        "allpass",
        "delay_samples",
    ):
        short = {v: k for k, v in DEFECTS.items()}[defect]
        for metric, mshort in (
            ("fir", "fir"),
            ("hammerstein_3", "h3"),
            ("hammerstein_5", "h5"),
            ("volterra_laguerre_2", "vl2"),
            ("volterra_laguerre_3", "vl3"),
            ("esr", "esr"),
            ("mrstft", "mrstft"),
            ("magnitude", "mag"),
        ):
            N.add(f"weak_{short}_{mshort}", weakest[defect][metric], RN, None)


def captured_shares() -> None:
    cap = load(RN)["captured_strongest_middle"]
    for dshort, defect in DEFECTS.items():
        for pshort, probe in PROBES.items():
            cell = cap[defect][probe]
            N.add(f"capt_{dshort}_{pshort}", cell["captured_median"], RN, 2)
            N.add(f"unrep_{dshort}_{pshort}", cell["unrepresented_median"], RN, 2)
    noise = [cap["noise_db"][p]["captured_median"] for p in PROBES.values()]
    N.add("capt_noise_min", min(noise), RN, 2)
    N.add("capt_noise_max", max(noise), RN, 2)
    echo = [
        cap["echo_beyond_memory"][p]["unrepresented_median"] for p in PROBES.values()
    ]
    N.add("unrep_echo_min", min(echo), RN, 2)
    N.add("unrep_echo_max", max(echo), RN, 2)
    u7 = [
        cap["order_beyond_degree"][p]["unrepresented_median"] for p in PROBES.values()
    ]
    N.add("unrep_u7_min", min(u7), RN, 2)
    N.add("unrep_u7_max", max(u7), RN, 2)
    wh = cap["gain"]["wiener_hammerstein"]["captured_median"]
    N.add("wh_capt_gain", wh, RN, 2)
    N.add("wh_capt_cubic", cap["cubic"]["wiener_hammerstein"]["captured_median"], RN, 2)


def table_readings() -> None:
    """Readings of table_test.csv at the strongest severity and the middle level."""
    rows = csv_rows(TABLE)
    strongest, levels = {}, defaultdict(set)
    for r in rows:
        strongest[r["defect"]] = max(
            strongest.get(r["defect"], -math.inf), float(r["severity"])
        )
        levels[(r["metric"], r["family"])].add(float(r["level_db"]))
    middle = {k: sorted(v)[len(v) // 2] for k, v in levels.items()}

    def cells(metric: str, defect: str, family: str | None = None) -> list[dict]:
        return [
            r
            for r in rows
            if r["metric"] == metric
            and r["defect"] == defect
            and (family is None or r["family"] == family)
            and float(r["severity"]) == strongest[defect]
            and float(r["level_db"]) == middle[(metric, r["family"])]
        ]

    def share_range(key: str, defect: str, family: str) -> None:
        values = [
            float(cells(p, defect, family)[0]["captured_share"])
            for p in PROBES.values()
        ]
        N.add(f"{key}_min", min(values), TABLE, 2)
        N.add(f"{key}_max", max(values), TABLE, 2)

    share_range("capt_echo_plucks", "echo_beyond_memory", "plucks")
    share_range("capt_echo_di", "echo_beyond_memory", "di")
    share_range("capt_echo_noise", "echo_beyond_memory", "noise")
    share_range("capt_echo_twotone", "echo_beyond_memory", "two_tone")
    share_range("capt_release_di", "release_ms", "di")
    share_range("capt_release_noise", "release_ms", "noise")
    over = [
        float(r["captured_share"])
        for r in rows
        if r["family"] == "di"
        and r["metric"] in ("hammerstein_3", "hammerstein_5")
        and r["defect"] in ("quadratic", "cross_term")
    ]
    N.add("capt_over_di_min", min(over), TABLE, 2)
    N.add("capt_over_di_max", max(over), TABLE, 2)

    def ratio_median(metric: str, defect: str) -> float:
        return statistics.median(
            float(r["detection_ratio"]) for r in cells(metric, defect)
        )

    N.add("base_mag_allpass_r", ratio_median("magnitude", "allpass"), TABLE, 1)
    N.add("base_mag_delay_r", ratio_median("magnitude", "delay_samples"), TABLE, 1)
    N.add("base_phase_cubic_r", ratio_median("phase", "cubic"), TABLE, 1)
    gain_phase = max(float(r["median_score"]) for r in cells("phase", "gain"))
    N.add("base_phase_gain_score_max", gain_phase, TABLE, text=sci(gain_phase, 1))
    env_gain = ratio_median("envelope", "gain")
    env_release = ratio_median("envelope", "release_ms")
    N.add("base_env_gain_r", env_gain, TABLE, text=sci(env_gain, 1))
    N.add("base_env_release_r", env_release, TABLE, text=sci(env_release, 1))
    for dshort, defect in (
        ("quad", "quadratic"),
        ("cubic", "cubic"),
        ("cross", "cross_term"),
    ):
        detected = sorted(
            r["metric"].removeprefix("harmonic_order_")
            for r in rows
            if r["metric"].startswith("harmonic_order_")
            and r["defect"] == defect
            and r["family"] == "sweep"
            and float(r["severity"]) == strongest[defect]
            and float(r["level_db"]) == middle[(r["metric"], "sweep")]
            and r["reading"] == "détecté"
        )
        N.add(f"harm_orders_{dshort}", " et ".join(detected), TABLE, None)


def repeatability() -> None:
    rep = load(RN)["repeatability"]
    # Metrics of the reading rule fixed before the test numbers (NOTE.md), probes of
    # the v2 freeze included; magnitude, phase and envelope were added post hoc
    # (baselines_extra.py) and are kept apart.
    core = (
        "d_audio",
        "esr",
        "fir",
        "hammerstein_3",
        "hammerstein_5",
        "volterra_laguerre_2",
        "volterra_laguerre_3",
        "mrstft",
    )
    medians = [rep[m]["median"] for m in core]
    p90 = [rep[m]["p90"] for m in core]
    N.add("rep_core_median_min", min(medians), RN, 4)
    N.add("rep_core_median_max", max(medians), RN, 4)
    N.add("rep_core_p90_min", min(p90), RN, 2)
    N.add("rep_core_p90_max", max(p90), RN, 2)
    extra = ("magnitude", "phase", "envelope")
    N.add("rep_base_median_min", min(rep[m]["median"] for m in extra), RN, 4)
    N.add("rep_base_median_max", max(rep[m]["median"] for m in extra), RN, 3)
    N.add("rep_base_p90_min", min(rep[m]["p90"] for m in extra), RN, 2)
    N.add("rep_base_p90_max", max(rep[m]["p90"] for m in extra), RN, 2)
    N.add("rep_esr_median", rep["esr"]["median"], RN, 4)
    N.add("rep_env_median", rep["envelope"]["median"], RN, 3)
    N.add("rep_imd_median", rep["imd"]["median"], RN, 3)
    N.add("rep_imd_p90", rep["imd"]["p90"], RN, 2)
    N.add("rep_wh_median", rep["wiener_hammerstein"]["median"], RN, 3)
    N.add("rep_wh_p90", rep["wiener_hammerstein"]["p90"], RN, 2)
    N.add("rep_phase_p90", rep["phase"]["p90"], RN, 2)
    N.add(
        "rep_harm_total_p90",
        rep["harmonic_total"]["p90"],
        RN,
        text=sci(rep["harmonic_total"]["p90"], 0),
    )
    orders = [rep[f"harmonic_order_{k}"]["p90"] for k in range(1, 6)]
    N.add("rep_harm_order_p90_max", max(orders), RN, 2)


def wiener_hammerstein() -> None:
    cells = {
        k: v
        for k, v in load(RN)["wiener_hammerstein"].items()
        if not k.startswith("repeat_floor")
    }
    N.add("wh_cells", len(cells), RN)
    N.add("wh_unreliable_cells", sum(v["unreliable"] for v in cells.values()), RN)
    spreads = [v["start_spread"] for v in cells.values()]
    N.add("wh_spread_median", statistics.median(spreads), RN, 3)
    cost = load(P + "cost.json")
    N.add("wh_cost_s", cost["wiener_hammerstein"], P + "cost.json", 0)


def decomposition() -> None:
    static = (
        "gain",
        "coloration_db",
        "delay_samples",
        "allpass",
        "quadratic",
        "cubic",
        "redistribution_bias",
        "prefilter_hz",
    )
    unrep = ("noise_db", "echo_beyond_memory")
    sums = defaultdict(list)
    for family in ("noise", "plucks"):
        data = load(P + f"decomposition_{family}.json")["defects"]
        for defect, cell in data.items():
            total = sum(cell["hammerstein_5"]["branch_share"].values())
            if defect in static:
                sums["static"].append(total)
            elif defect in unrep:
                sums["unrep"].append(total)
    src = P + "decomposition_noise.json; " + P + "decomposition_plucks.json"
    for defect, tag in (("cross_term", "cross"), ("quadratic", "quad")):
        order2 = [
            load(P + f"decomposition_{family}.json")["defects"][defect][
                "hammerstein_3"
            ]["branch_share"]["order_2"]
            for family in ("noise", "plucks")
        ]
        N.add(f"decomp_{tag}_h3_order2_min", min(order2), src, 2)
        N.add(f"decomp_{tag}_h3_order2_max", max(order2), src, 2)
    N.add("decomp_static_min", min(sums["static"]), src, 1)
    N.add("decomp_static_max", max(sums["static"]), src, 1)
    N.add("decomp_unrep_min", min(sums["unrep"]), src, 1)
    N.add("decomp_unrep_max", max(sums["unrep"]), src, 1)


def dynamics() -> None:
    src = P + "dynamics_verdict.json"
    d = load(src)
    levels = d["levels"]
    for level, tag in (("-12.0", "m12"), ("-6.0", "m6")):
        for severity in ("60.0", "100.0", "200.0"):
            N.add(
                f"e2_release_{tag}_{severity[:-2]}",
                levels[level]["release"][severity],
                src,
                2,
            )
    floors = [v["floor"] for v in levels.values()]
    N.add("e2_floor_min", min(floors), src, text=sci(min(floors), 1))
    N.add("e2_floor_max", max(floors), src, text=sci(max(floors), 1))
    worst = [v["worst_other"][1] for v in levels.values()]
    N.add("e2_worst_other", statistics.median(worst), src, 2)
    names = sorted({v["worst_other"][0] for v in levels.values()})
    readable = {"coloration_db": "coloration"}
    N.add(
        "e2_worst_other_name",
        ", ".join(readable.get(name, name) for name in names),
        src,
        None,
    )
    spec = [v["release"]["200.0"] / v["worst_other"][1] for v in levels.values()]
    N.add("e2_spec_ratio_min", min(spec), src, 0)
    N.add("e2_spec_ratio_max", max(spec), src, 0)
    control = [
        v["attack_of_release"][s] / v["release"][s]
        for v in levels.values()
        for s in v["release"]
    ]
    N.add("e2_attack_ratio_min", min(control), src, 2)
    N.add("e2_attack_ratio_max", max(control), src, 2)
    echo = [v["co_responders"]["echo_beyond_memory"] for v in levels.values()]
    pre = [v["co_responders"]["prefilter_hz"] for v in levels.values()]
    N.add("e2_echo_min", min(echo), src, 2)
    N.add("e2_echo_max", max(echo), src, 2)
    N.add("e2_prefilter_min", min(pre), src, 2)
    N.add("e2_prefilter_max", max(pre), src, 2)
    N.add("e2_levels", len(levels), src)
    N.add("e2_criteria_all_levels", min(d["counts"].values()), src)


def french_list(items: list) -> str:
    """'2', '2 et 4', '2, 4 et 5'."""
    words = [str(i) for i in items]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " et " + words[-1]


def harmonic_orders(rows: list[dict], pairs: str) -> None:
    # Harmonic signature order by order (1 to 5) on the three sweeps, same rules as
    # report_numbers.py: sections flagged by the integrity guard left out, floor =
    # median of the centre repeats, low-signal change left out.
    broken = {
        (r["order"], r["section"])
        for r in load(P + "sweep_integrity.json")
        if not r["intact"]
    }
    above, ratios = [], []
    for segment in ("sweep_-20dBFS", "sweep_-10dBFS", "sweep_-3dBFS"):
        for k in range(1, 6):
            sel = [
                r
                for r in rows
                if r["segment"] == segment
                and r["metric"] == f"harmonic_order_{k}"
                and all(
                    (int(t.split("#")[1]), segment) not in broken
                    for t in (r["ref"], r["eval"])
                )
            ]
            repeats = [float(r["score"]) for r in sel if r["kind"] == "repeat"]
            if not repeats:
                continue
            base = statistics.median(repeats)
            for r in sel:
                if r["kind"] == "positive" and r["low_signal"] != "True":
                    ratio = float(r["score"]) / base
                    ratios.append(ratio)
                    if ratio >= 10:
                        above.append((segment, r["eval"], k))
    hsrc = pairs + "; " + P + "sweep_integrity.json"
    N.add("vs_harm_order_pos_max", max(ratios), hsrc, 1)
    N.add("vs_harm_order_ge10_cells", len(above), hsrc)
    N.add("vs_harm_order_ge10_changes", len({(a, b) for a, b, _ in above}), hsrc)
    N.add(
        "vs_harm_order_ge10_orders",
        french_list(sorted({k for _, _, k in above})),
        hsrc,
        None,
    )
    N.add(
        "vs_harm_order_ge10_segment",
        ", ".join(sorted({a for a, _, _ in above})),
        hsrc,
        None,
    )


def vs100r() -> None:
    real = load(RN)["real_vs100r"]
    for key, tag in (
        ("di_test|hammerstein_3|all", "h3_all"),
        ("di_test|hammerstein_3|common", "h3_common"),
        ("di_test|fir|common", "fir_common"),
        ("di_test|esr|all", "esr_all"),
        ("di_test|esr|common", "esr_common"),
        ("di_test|mrstft|all", "mrstft_all"),
        ("di_test|mrstft|common", "mrstft_common"),
        ("pink_noise_-30dBFS_rms|esr|all", "pink_esr"),
        ("pink_noise_-30dBFS_rms|hammerstein_3|all", "pink_h3"),
        ("sweep_-10dBFS|harmonic_total|all", "sweep10_harm"),
        ("sweep_-20dBFS|harmonic_total|all", "sweep20_harm"),
        ("sweep_-3dBFS|harmonic_total|all", "sweep3_harm_guarded"),
    ):
        cell = real[key]
        N.add(f"vs_auc_{tag}", cell["auc"], RN, 2)
        N.add(f"vs_p_{tag}", cell["p_exact"], RN, text=p_text(cell["p_exact"]))
        N.add(f"vs_npos_{tag}", cell["n_pos"], RN)
        N.add(f"vs_nneg_{tag}", cell["n_neg"], RN)
    for metric, tag in (("hammerstein_3", "h3"), ("esr", "esr"), ("mrstft", "mrstft")):
        cell = real[f"di_test|{metric}|all"]
        N.add(f"vs_neg_median_{tag}", cell["negative_median_over_floor"], RN, 2)
        N.add(f"vs_neg_max_{tag}", cell["negative_max_over_floor"], RN, 2)
        N.add(f"vs_pos_min_{tag}", min(cell["positive_over_floor"]), RN, 1)
        N.add(f"vs_pos_max_{tag}", max(cell["positive_over_floor"]), RN, 1)
    positives = [
        v
        for key, cell in real.items()
        if key.split("|")[0] in ("di_test", "pink_noise_-30dBFS_rms")
        for v in cell["positive_over_floor"]
    ]
    N.add("vs_pos_max_all", max(positives), RN, 1)
    N.add("vs_pos_ge10", sum(v >= 10 for v in positives), RN)
    total = [
        v
        for key, cell in real.items()
        if key.endswith("|all") and "|harmonic_total|" in key
        for v in cell["positive_over_floor"]
    ]
    N.add("vs_harm_total_pos_max", max(total), RN, 1)

    pairs = P + "real_pairs_vs100r.csv"
    rows = csv_rows(pairs)

    def amplitude(r: dict) -> float:
        return float(r["score"]) ** (0.5 if r["metric"] == "esr" else 1)

    def floor(metric: str) -> float:
        return statistics.median(
            amplitude(r)
            for r in rows
            if r["segment"] == "di_test"
            and r["metric"] == metric
            and r["kind"] == "repeat"
        )

    for metric, tag in (("hammerstein_3", "h3"), ("esr", "esr"), ("mrstft", "mrstft")):
        low = [
            r
            for r in rows
            if r["segment"] == "di_test"
            and r["metric"] == metric
            and r["low_signal"] == "True"
        ]
        N.add(f"vs_gain0_{tag}", amplitude(low[0]) / floor(metric), pairs, 1)
        treble = [
            r
            for r in rows
            if r["segment"] == "di_test"
            and r["metric"] == metric
            and r["eval"].startswith("G050_C050_B050_M050_T100")
        ]
        N.add(f"vs_treble10_{tag}", amplitude(treble[0]) / floor(metric), pairs, 2)
    repeats = {
        (r["ref"].split("#")[1], r["eval"].split("#")[1]): float(r["score"])
        for r in rows
        if r["segment"] == "di_test"
        and r["metric"] == "hammerstein_3"
        and r["kind"] == "repeat"
    }
    late = repeats[("20", "39")]
    early = [v / late for k, v in repeats.items() if "1" in k]
    N.add("vs_drift_min", min(early), pairs, 1)
    N.add("vs_drift_max", max(early), pairs, 1)
    harmonic_orders(rows, pairs)
    bound = [r for r in rows if r["bound_holds"] in ("True", "False")]
    N.add("vs_bound_rows", len(bound), pairs)
    N.add("vs_bound_violations", sum(r["bound_holds"] == "False" for r in bound), pairs)
    raw = [
        r
        for r in rows
        if r["segment"] == "sweep_-3dBFS" and r["metric"] == "harmonic_total"
    ]
    pos = [
        float(r["score"])
        for r in raw
        if r["kind"] == "positive" and r["low_signal"] != "True"
    ]
    neg = [float(r["score"]) for r in raw if r["kind"] == "negative_control"]
    a, p = auc(pos, neg)
    N.add("vs_auc_sweep3_harm_raw", a, pairs, 2)
    N.add("vs_p_sweep3_harm_raw", p, pairs, text=p_text(p))

    extra = P + "real_extra_vs100r.csv"
    rows = csv_rows(extra)
    for segment, stag in (("di_test", "di"), ("pink_noise_-30dBFS_rms", "pink")):
        for metric, mtag in (
            ("magnitude", "mag"),
            ("phase", "phase"),
            ("envelope", "env"),
        ):
            use = [
                r
                for r in rows
                if r["segment"] == segment
                and r["metric"] == metric
                and r["input_common"] != "False"
            ]
            pos = [
                float(r["score"])
                for r in use
                if r["kind"] == "positive" and r["low_signal"] != "True"
            ]
            neg = [float(r["score"]) for r in use if r["kind"] == "negative_control"]
            a, p = auc(pos, neg)
            N.add(f"vs_auc_{stag}_{mtag}", a, extra, 2)
            N.add(f"vs_p_{stag}_{mtag}", p, extra, text=p_text(p))

    sweep = load(P + "gap_checks_sweep.json")
    src = P + "gap_checks_sweep.json"
    N.add(
        "slip_rep_order3_m10",
        sweep["repeat|sweep_-10dBFS"]["order_3"]["complex_rel"],
        src,
        3,
    )
    N.add(
        "slip_rep_order3_m3",
        sweep["repeat|sweep_-3dBFS"]["order_3"]["complex_rel"],
        src,
        2,
    )
    setting = sorted(k.split("|")[0] for k in sweep if not k.startswith("repeat"))[0]
    N.add("slip_gain_setting", float(setting.removeprefix("gain_")), src, 1)
    odd = [f"order_{k}" for k in (3, 5, 7, 9)]
    m10 = [sweep["gain_7.5|sweep_-10dBFS"][o]["complex_rel"] for o in odd]
    m3 = [sweep["gain_7.5|sweep_-3dBFS"][o]["complex_rel"] for o in odd]
    N.add("slip_gain_m10_min", min(m10), src, 2)
    N.add("slip_gain_m10_max", max(m10), src, 2)
    N.add("slip_gain_m3_min", min(m3), src, 2)
    N.add("slip_gain_m3_max", max(m3), src, 2)
    harm = load(P + "gap_checks_slip_harm.json")["sweep_-10dBFS"]["slips"]
    src = P + "gap_checks_slip_harm.json"

    def worst(times: list[str]) -> float:
        return max(
            v["of_repeat"]
            for t in times
            for k, v in harm[t].items()
            if k.startswith("order_")
        )

    times = sorted(harm, key=lambda t: float(t.rstrip("s")))
    missed = [t for t in times if not harm[t]["guard_flags"]]
    early_missed = [t for t in missed if float(t.rstrip("s")) < 1.3]
    late_missed = [t for t in missed if float(t.rstrip("s")) > 4.0]
    flagged = [t for t in times if harm[t]["guard_flags"]]
    N.add("slip_missed_early_max", worst(early_missed), src, 2)
    N.add("slip_missed_late_max", worst(late_missed), src, 2)
    N.add("slip_detected_max", worst(flagged), src, 0)
    N.add("slip_grid_positions", len(times), src)
    integrity = load(P + "sweep_integrity.json")
    N.add(
        "slip_flagged_sections",
        sum(not r["intact"] for r in integrity),
        P + "sweep_integrity.json",
    )


def jvm() -> None:
    real = load(RN)["real_jvm"]
    pre, spk = real["jvm410h_PreampOut"], real["jvm410h_SpeakerOut"]
    N.add("jvm_ladders", pre["ladders"], RN)
    N.add(
        "jvm_spearman_min",
        min(pre["ladder_spearman_min"], spk["ladder_spearman_min"]),
        RN,
        0,
    )
    N.add("jvm_h5_capt_pre", pre["h5_captured_median"], RN, 2)
    N.add("jvm_h5_capt_spk", spk["h5_captured_median"], RN, 2)
    N.add("jvm_h5_unrep_pre", pre["h5_unrepresented_median"], RN, 2)
    N.add("jvm_h5_unrep_spk", spk["h5_unrepresented_median"], RN, 2)
    N.add("jvm_bound_violations", pre["bound_violations"] + spk["bound_violations"], RN)
    pairs = P + "real_pairs_jvm.csv"
    rows = csv_rows(pairs)
    N.add(
        "jvm_bound_rows",
        sum(r["bound_holds"] in ("True", "False") for r in rows),
        pairs,
    )
    N.add("jvm_eval_settings", len({r["eval"] for r in rows}), pairs)

    extra = P + "real_extra_jvm.csv"
    rows = csv_rows(extra)
    rhos = []
    for device in sorted({r["device"] for r in rows}):
        for knob in "GBMT":
            for side in (1, -1):
                for metric in ("magnitude", "phase", "envelope"):
                    sel = []
                    for r in rows:
                        if (
                            r["device"] != device
                            or r["knob"] != knob
                            or r["metric"] != metric
                        ):
                            continue
                        value = int(r["eval"].split("_")["BMTG".index(knob)][1:])
                        if (value - 50) * side > 0:
                            sel.append((abs(value - 50), float(r["score"])))
                    if len(sel) > 2:
                        rhos.append(spearmanr(*zip(*sel, strict=True)).statistic)
    N.add("jvm_extra_ladders", len(rhos), extra)
    N.add("jvm_extra_spearman_min", min(rhos), extra, 0)

    sat = load(P + "gap_checks_jvm_saturating.json")
    src = P + "gap_checks_jvm_saturating.json"
    N.add("jvm_h5_res_g0", sat["B050_M050_T050_G000"]["hammerstein_5"], src, 2)
    N.add("jvm_h5_res_g5", sat["B050_M050_T050_G050"]["hammerstein_5"], src, 2)
    N.add("jvm_h5_res_g10", sat["B050_M050_T050_G100"]["hammerstein_5"], src, 2)
    N.add("jvm_h5_esr_g5", sat["B050_M050_T050_G050"]["hammerstein_5"] ** 2, src, 2)
    N.add("jvm_tanh_res_g5", sat["B050_M050_T050_G050"]["saturating_basis"], src, 2)
    N.add("jvm_tanh_res_g10", sat["B050_M050_T050_G100"]["saturating_basis"], src, 2)
    gap = load(P + "gap_checks_jvm.json")
    gsrc = P + "gap_checks_jvm.json"
    N.add("jvm_h5_1024_res_g5", gap["capacity_centre"]["hammerstein_5_1024"], gsrc, 2)
    vl3 = [
        v["volterra_laguerre_3"]["unrepresented"]
        for k, v in gap.items()
        if k != "capacity_centre"
    ]
    h5 = [
        v["hammerstein_5"]["unrepresented"]
        for k, v in gap.items()
        if k != "capacity_centre"
    ]
    N.add("jvm_vl3_unrep_min", min(vl3), gsrc, 2)
    N.add("jvm_vl3_unrep_max", max(vl3), gsrc, 2)
    N.add("jvm_h5_unrep_pairs_min", min(h5), gsrc, 2)
    N.add("jvm_h5_unrep_pairs_max", max(h5), gsrc, 2)
    par = load(P + "gap_checks_jvm_parallel.json")
    psrc = P + "gap_checks_jvm_parallel.json"
    ins = [v["in_sample"] for g in par.values() for v in g.values()]
    rd = [v["reading"] for g in par.values() for v in g.values()]
    N.add("jvm_par_in_min", min(ins), psrc, 2)
    N.add("jvm_par_in_max", max(ins), psrc, 2)
    N.add("jvm_par_read_min", min(rd), psrc, 2)
    N.add("jvm_par_read_max", max(rd), psrc, 2)
    causes = load(P + "gap_checks_jvm_causes.json")
    csrc = P + "gap_checks_jvm_causes.json"
    N.add("jvm_insample_150", causes["in_sample_h5"]["0-150 s"], csrc, 2)
    N.add("jvm_insample_30", causes["in_sample_h5"]["0-30 s"], csrc, 2)
    silence = causes["output_in_input_silence_db"]
    for gain in ("G050", "G000"):
        value = silence[gain]
        key = "jvm_silence_g5_db" if gain == "G050" else "jvm_silence_g0_db"
        N.add(key, value, csrc, text=fr(value, 0).replace("-", "\N{MINUS SIGN}"))
    history = causes["residual_by_level_and_history"]
    N.add("jvm_after_louder", history["after_louder_6db"], csrc, 2)
    N.add("jvm_no_louder", history["no_louder"], csrc, 2)
    lags = list(causes["best_lag_samples_by_window"].values())
    N.add("jvm_lag_abs_max", max(abs(v) for v in lags), csrc)
    N.add("jvm_wh_in", causes["wiener_hammerstein"]["in_sample_1s"], csrc, 2)
    N.add("jvm_wh_read", causes["wiener_hammerstein"]["reading_1s"], csrc, 2)
    reading = [
        sat["B050_M050_T050_G050"]["hammerstein_5"],
        sat["B050_M050_T050_G100"]["hammerstein_5"],
        sat["B050_M050_T050_G050"]["saturating_basis"],
        sat["B050_M050_T050_G100"]["saturating_basis"],
        gap["capacity_centre"]["hammerstein_5_1024"],
        *rd,
        causes["wiener_hammerstein"]["reading_1s"],
    ]
    best = min(reading)
    src_all = "; ".join((src, gsrc, psrc, csrc))
    N.add("jvm_best_reading_res", best, src_all, 2)
    N.add("jvm_best_reading_esr", best**2, src_all, 2)
    linear_in_sample = [
        *ins,
        causes["in_sample_h5"]["0-150 s"],
        causes["in_sample_h5"]["0-30 s"],
    ]
    N.add("jvm_best_insample_linear", min(linear_in_sample), psrc + "; " + csrc, 2)


def cost() -> None:
    src = P + "cost.json"
    c = load(src)
    for name, tag in (
        ("esr", "esr"),
        ("mrstft", "mrstft"),
        ("fir", "fir"),
        ("imd", "imd"),
        ("hammerstein_3", "h3"),
        ("hammerstein_5", "h5"),
        ("harmonic_signature", "harm"),
        ("volterra_laguerre_2", "vl2"),
        ("volterra_laguerre_3", "vl3"),
    ):
        value = c[name]
        text = sci(value, 1) if value < 0.01 else sig(value)
        N.add(f"cost_{tag}_s", value, src, text=text)
    N.add("cost_load_min", min(c["load_before"], c["load_after"]), src, 0)
    N.add("cost_load_max", max(c["load_before"], c["load_after"]), src, 0)
    src = P + "cost_scaling.json"
    s = load(src)
    for name, tag in (
        ("esr", "esr"),
        ("mrstft", "mrstft"),
        ("fir", "fir"),
        ("hammerstein_3", "h3"),
        ("hammerstein_5", "h5"),
        ("volterra_laguerre_2", "vl2"),
        ("volterra_laguerre_3", "vl3"),
        ("wiener_hammerstein", "wh"),
    ):
        seconds = s["metrics"][name]["seconds"]
        longest = max(seconds, key=float)
        per_minute = seconds[longest] / float(longest) * 60.0
        N.add(f"scale_{tag}_exponent", s["metrics"][name]["exponent"], src, 1)
        N.add(f"scale_{tag}_per_min_s", per_minute, src, text=sig(per_minute))
    N.add(
        "scale_wh_per_min_min",
        s["metrics"]["wiener_hammerstein"]["seconds"]["4.0"] / 4.0,
        src,
        0,
    )


def main() -> None:
    protocol()
    bench_counts()
    captured_shares()
    table_readings()
    repeatability()
    wiener_hammerstein()
    decomposition()
    dynamics()
    vs100r()
    jvm()
    cost()
    N.write(HERE / "numbers.json")
    print(f"{len(N.data)} keys -> {HERE / 'numbers.json'}")


if __name__ == "__main__":
    main()
