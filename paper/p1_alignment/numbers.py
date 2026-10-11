#!/usr/bin/env python3
"""Numbers of dossier P1 (alignment and polarity traps), read from existing records.

CPU only, a few seconds: no training, no model inference, no audio. Every value comes
from a JSON written by an earlier run or evaluation, or, for a pre-registered threshold,
from the Markdown that froze it. Writes paper/p1_alignment/numbers.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This file is named numbers.py: drop its own directory from the import path so that
# the standard library's `numbers` (imported by statistics) is not shadowed by it.
HERE = Path(__file__).resolve().parent
sys.path[0] = str(HERE.parent)

import math  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402

from numbers_lib import ROOT, Numbers, fr, load, sci  # noqa: E402

N = Numbers()
MINUS, TIMES = "\u2212", "\u00d7"


def signed(x: float, digits: int = 3) -> str:
    """+0,039 or -0,118 with a typographic minus sign."""
    return f"{x:+.{digits}f}".replace(".", ",").replace("-", MINUS)


def neg(x: float, digits: int = 1) -> str:
    """Fixed point with a typographic minus."""
    return fr(x, digits).replace("-", MINUS)


def grab(path: str, pattern: str) -> float:
    """A threshold frozen in a tracked Markdown file (French decimal comma)."""
    match = re.search(pattern, (ROOT / path).read_text(), re.S)
    if not match:
        raise ValueError(f"{path}: pattern {pattern!r} not found")
    return float(match.group(1).replace(",", "."))


def bench(run: str) -> str:
    return f"demo/nablafx_bench/{run}.json"


# --- 1. F-Fuzz: 4-sample latency of the targets (E-0027, E-0035) -------------------

SEG, ADV = "diagnosis/ffuzz/segments.json", "diagnosis/ffuzz/segments_adv4.json"
SSM_R2, S4_R2 = "param_ssm_wavenet_ffuzz_r2", "param_s4_ffuzz_r2"
SSM_A4, S4_A4 = "param_ssm_wavenet_ffuzz_adv4_r2", "param_s4_ffuzz_adv4"
FIX = "FIX_ffuzz.md"
seg, adv = load(SEG), load(ADV)
r2 = {k: load(bench(r)) for k, r in (("ssm", SSM_R2), ("s4", S4_R2))}
a4 = {k: load(bench(r)) for k, r in (("ssm", SSM_A4), ("s4", S4_A4))}
src_r2 = f"{bench(SSM_R2)}; {bench(S4_R2)}"
src_a4 = f"{bench(SSM_A4)}; {bench(S4_A4)}"


def mean(rows: list[dict], field: str) -> float:
    return statistics.mean(r[field] for r in rows)


def lag_count(rows: list[dict], lag: int) -> int:
    return sum(r["lag"] == lag for r in rows)


rows_ssm, rows_s4 = seg[SSM_R2], seg[S4_R2]
N.add("ffuzz_n_segments", len(rows_ssm), SEG)
N.add("ffuzz_n_settings", len({r["setting"] for r in rows_ssm}), SEG)
N.add("ffuzz_n_held_out", len(r2["ssm"]["held_out_settings"]), bench(SSM_R2))
N.add("ffuzz_seed", r2["ssm"]["seed"], src_r2)
N.add("ffuzz_split", r2["ssm"]["split_seed"], src_r2)
N.add("ffuzz_params_ssm", r2["ssm"]["parameters"], bench(SSM_R2))
N.add("ffuzz_params_s4", r2["s4"]["parameters"], bench(S4_R2))
N.add("ffuzz_w_l1", r2["ssm"]["loss_weights"]["l1"], src_r2, 1)
N.add("ffuzz_w_mrstft", r2["ssm"]["loss_weights"]["mrstft"], src_r2, 1)
N.add("ffuzz_lag4_ssm", lag_count(rows_ssm, 4), SEG)
N.add("ffuzz_lag4_s4", lag_count(rows_s4, 4), SEG)
N.add("ffuzz_lag3_s4", lag_count(rows_s4, 3), SEG)
esr_ssm = r2["ssm"]["test_last"]["metric/test/esr"]
esr_s4 = r2["s4"]["test_last"]["metric/test/esr"]
N.add("ffuzz_esr_raw_ssm", esr_ssm, bench(SSM_R2), 4)
N.add("ffuzz_esr_raw_s4", esr_s4, bench(S4_R2), 4)
N.add("ffuzz_esr_raw_ratio", esr_s4 / esr_ssm, src_r2, 2)
mr_ssm = r2["ssm"]["test_last"]["metric/test/mrstft"]
mr_s4 = r2["s4"]["test_last"]["metric/test/mrstft"]
N.add("ffuzz_mrstft_raw_ssm", mr_ssm, bench(SSM_R2), 2)
N.add("ffuzz_mrstft_raw_s4", mr_s4, bench(S4_R2), 2)
N.add("ffuzz_mrstft_raw_ratio", mr_s4 / mr_ssm, src_r2, 1)
lag0_ssm, lag0_s4 = (
    mean(rows_ssm, "esr_interior_lag0"),
    mean(rows_s4, "esr_interior_lag0"),
)
best_ssm, best_s4 = (
    mean(rows_ssm, "esr_interior_best_lag"),
    mean(rows_s4, "esr_interior_best_lag"),
)
N.add("ffuzz_esr_interior_lag0_ssm", lag0_ssm, SEG, 3)
N.add("ffuzz_esr_interior_lag0_s4", lag0_s4, SEG, 3)
N.add("ffuzz_esr_realigned_ssm", best_ssm, SEG, 4)
N.add("ffuzz_esr_realigned_s4", best_s4, SEG, 4)
N.add("ffuzz_realign_factor_ssm", lag0_ssm / best_ssm, SEG, 0)
N.add("ffuzz_realign_factor_s4", lag0_s4 / best_s4, SEG, 0)
by_setting = {}
for r in rows_ssm:
    by_setting.setdefault(r["setting"], []).append(r["esr_interior_lag0"])
setting_means = [statistics.mean(v) for _, v in sorted(by_setting.items())]
N.add("ffuzz_esr_lag0_f000_ssm", setting_means[0], SEG, 3)
N.add("ffuzz_esr_lag0_f100_ssm", setting_means[-1], SEG, 3)

TRAIN = "diagnosis/ffuzz/train_lag.json"
train = load(TRAIN)
N.add("ffuzz_train_n_segments", len(train[SSM_R2]), TRAIN)
N.add("ffuzz_train_lag4_ssm", lag_count(train[SSM_R2], 4), TRAIN)
N.add("ffuzz_train_lag4_s4", lag_count(train[S4_R2], 4), TRAIN)
DATA = "diagnosis/ffuzz/data_lag.json"
data_lags = [v["lag"] for v in load(DATA).values()]
N.add(
    "ffuzz_data_lag_min",
    min(data_lags),
    DATA,
    text=str(min(data_lags)).replace("-", MINUS),
)
N.add("ffuzz_data_lag_max", max(data_lags), DATA, text=f"+{max(data_lags)}")

# pre-registered fix: thresholds frozen in FIX_ffuzz.md section 1, results section 6
N.add("ffuzz_thr_esr", grab(FIX, r"ESR de test moyenne ≤ (\d+,\d+)"), FIX, 2)
N.add(
    "ffuzz_thr_gap",
    grab(FIX, r"ESR\(S4\) ≥ (\d+,\d+) " + TIMES + r" ESR\(SSM\)"),
    FIX,
    1,
)
N.add("ffuzz_thr_f100", int(grab(FIX, r"F100/F000 de l'ESR de SSM ≤ (\d+)")), FIX)
N.add("ffuzz_target_advance", a4["ssm"]["target_advance"], src_a4)
adv_ssm = a4["ssm"]["test_last"]["metric/test/esr"]
adv_s4 = a4["s4"]["test_last"]["metric/test/esr"]
N.add("ffuzz_adv4_esr_ssm", adv_ssm, bench(SSM_A4), 5)
N.add("ffuzz_adv4_esr_s4", adv_s4, bench(S4_A4), 5)
N.add("ffuzz_adv4_ratio", adv_s4 / adv_ssm, src_a4, 2)
by = a4["ssm"]["test_last_by_setting"]
N.add(
    "ffuzz_adv4_f100_f000_ssm",
    by["F100_V100"]["esr"] / by["F000_V100"]["esr"],
    bench(SSM_A4),
    2,
)
N.add("ffuzz_adv4_lag0_ssm", lag_count(adv[SSM_A4], 0), ADV)
N.add("ffuzz_adv4_lag0_s4", lag_count(adv[S4_A4], 0), ADV)
N.add("ffuzz_adv4_interior_ssm", mean(adv[SSM_A4], "esr_interior_lag0"), ADV, 5)
N.add("ffuzz_adv4_interior_s4", mean(adv[S4_A4], "esr_interior_lag0"), ADV, 5)
N.add("ffuzz_adv4_steps_ssm", a4["ssm"]["global_step"], bench(SSM_A4))
N.add("ffuzz_adv4_steps_s4", a4["s4"]["global_step"], bench(S4_A4))

# alternative causes refuted by their written thresholds (DIAGNOSIS_ffuzz.md section 4)
esr = sorted((r["esr"] for r in rows_ssm), reverse=True)
N.add("ffuzz_alt_mean", statistics.mean(esr), SEG, 3)
N.add("ffuzz_alt_median", statistics.median(esr), SEG, 3)
N.add("ffuzz_alt_worst10_share", sum(esr[: len(esr) // 10]) / sum(esr), SEG, 2)
N.add("ffuzz_alt_hf_share", mean(rows_ssm, "hf_error_share"), SEG, 2)
N.add(
    "ffuzz_alt_lowpass_ratio", mean(rows_ssm, "esr_interior_lowpass") / lag0_ssm, SEG, 2
)
markers = [r for r in rows_ssm if r["marker"]]
N.add("ffuzz_alt_marker_n", len(markers), SEG)
N.add("ffuzz_alt_marker_pct", 100 * sum(r["esr"] for r in markers) / sum(esr), SEG, 1)
N.add("ffuzz_alt_interior", lag0_ssm, SEG, 4)
N.add("ffuzz_alt_full", statistics.mean(esr), SEG, 4)

# controls: the same lag instrument on Big Muff and Marshall (other_devices_lag.py)
OTHER = "diagnosis/ffuzz/other_devices_lag.json"
other = load(OTHER)
bm, bm_s4 = other["paired_ssm_wavenet_s925"], other["paired_s4_tf_l_16_s925"]
ma, ma_s4 = other["param_ssm_wavenet_marshall"], other["param_s4_marshall"]
N.add("ctrl_bigmuff_n", len(bm), OTHER)
N.add("ctrl_bigmuff_lag0_ssm", lag_count(bm, 0), OTHER)
N.add("ctrl_bigmuff_lag0_s4", lag_count(bm_s4, 0), OTHER)
N.add("ctrl_bigmuff_esr_lag0", mean(bm, "esr_lag0"), OTHER, 4)
N.add("ctrl_bigmuff_esr_best", mean(bm, "esr_best"), OTHER, 4)
N.add("ctrl_marshall_n", len(ma), OTHER)
N.add("ctrl_marshall_lag0_ssm", lag_count(ma, 0), OTHER)
N.add("ctrl_marshall_lag0_s4", lag_count(ma_s4, 0), OTHER)
N.add("ctrl_marshall_esr_lag0", mean(ma, "esr_lag0"), OTHER, 3)
N.add("ctrl_marshall_esr_best", mean(ma, "esr_best"), OTHER, 3)
N.add(
    "ctrl_marshall_gain_pct",
    100 * (1 - mean(ma, "esr_best") / mean(ma, "esr_lag0")),
    OTHER,
    0,
)

# --- 2. VS100R at 48 kHz: sub-sample oracle delay (E-0037, E-0040, F-0032) ----------

MEAS = {
    42: "diagnosis/vs100r_192k/measures.json",
    43: "diagnosis/vs100r_192k/measures_s43.json",
    44: "diagnosis/vs100r_192k/measures_s44.json",
}
src_meas = "; ".join(MEAS.values())
EVAL192 = "demo/capture/valvestate/eval_192k.json"
ev192 = load(EVAL192)
N.add("vs100r_n_test_takes", len(ev192["takes"]), EVAL192)
N.add(
    "vs100r_m1_d", ev192["readings"]["d"], EVAL192, text=signed(ev192["readings"]["d"])
)
N.add(
    "vs100r_m1_bound",
    -ev192["readings"]["bound"],
    EVAL192,
    text=neg(-ev192["readings"]["bound"], 3),
)
PRED192 = "demo/capture/valvestate/predictions_192k.md"
thr_ln = -math.log(grab(PRED192, "seuils " + MINUS + r"ln (\d+,\d+)"))
N.add("vs100r_m1_thr_ln", thr_ln, PRED192, text=neg(thr_ln, 3))
N.add("vs100r_m1_label", ev192["readings"]["M1"], EVAL192, None)
delay_a, delay_b, ratios, ratios_42, n_ge2, small = [], [], [], [], [], []
better_aligned, better_raw = {}, {}
for seed, path in MEAS.items():
    takes = load(path)["takes"]
    others = {t: v for t, v in takes.items() if t != "16"}
    delay_a.append(statistics.median(v["A48"]["oracle_delay"] for v in others.values()))
    delay_b.append(
        statistics.median(v["B192"]["oracle_delay"] for v in others.values())
    )
    r = [v["A48"]["esr"] / v["A48"]["esr_after_delay"] for v in takes.values()]
    big = [x for x in r if x >= 2]
    ratios += big
    small += [x for x in r if x < 2]
    n_ge2.append(len(big))
    if seed == 42:
        ratios_42 = big
    better_aligned[seed] = sum(
        v["A48"]["esr_after_delay"] < v["B192"]["esr_after_delay"]
        for v in takes.values()
    )
    better_raw[seed] = sum(v["B192"]["esr"] < v["A48"]["esr"] for v in takes.values())
N.add("vs100r_delay_a48_min", min(delay_a), src_meas, 2)
N.add("vs100r_delay_a48_max", max(delay_a), src_meas, 2)
N.add("vs100r_delay_b192_min", min(delay_b), src_meas, 2)
N.add("vs100r_delay_b192_max", max(delay_b), src_meas, 2)
N.add("vs100r_ratio_s42_min", min(ratios_42), MEAS[42], 1)
N.add("vs100r_ratio_s42_max", max(ratios_42), MEAS[42], 1)
N.add("vs100r_ratio_all_min", min(ratios), src_meas, 1)
N.add("vs100r_ratio_all_max", max(ratios), src_meas, 1)
N.add("vs100r_ratio_small_max", max(small), src_meas, 2)
assert len(set(n_ge2)) == 1, n_ge2
N.add("vs100r_n_takes_ge2", n_ge2[0], src_meas)
for seed in MEAS:
    N.add(f"vs100r_a48_better_aligned_s{seed}", better_aligned[seed], MEAS[seed])
    N.add(f"vs100r_b192_better_raw_s{seed}", better_raw[seed], MEAS[seed])
t16 = load(MEAS[42])["takes"]["16"]
N.add("vs100r_t16_delay_a48", t16["A48"]["oracle_delay"], MEAS[42], 2)
N.add("vs100r_t16_delay_b192", t16["B192"]["oracle_delay"], MEAS[42], 2)
N.add(
    "vs100r_t16_ln_ratio_aligned",
    math.log(t16["B192"]["esr_after_delay"] / t16["A48"]["esr_after_delay"]),
    MEAS[42],
    2,
)

SHIFT = "demo/capture/valvestate/timing_shift.json"
shift = load(SHIFT)
N.add("vs100r_shift_n", sum(not r["low_signal"] for r in shift["rows"]), SHIFT)
N.add(
    "vs100r_shift_common",
    shift["summary"]["common_shift"],
    SHIFT,
    1,
    text=signed(shift["summary"]["common_shift"], 1),
)
N.add("vs100r_shift_us", 1e6 * shift["summary"]["common_shift"] / 48_000, SHIFT, 0)
N.add("vs100r_shift_esr0", shift["summary"]["median_esr_0"], SHIFT, 3)
N.add("vs100r_shift_esr1", shift["summary"]["median_esr_common_shift"], SHIFT, 4)
CTRL = "demo/capture/valvestate/timing_controls.json"
ctrl = load(CTRL)["summary"]
N.add(
    "vs100r_ctrl_bigmuff_shift",
    ctrl["pilotC_decide_seed42"]["median_best_shift"],
    CTRL,
    text=neg(ctrl["pilotC_decide_seed42"]["median_best_shift"]),
)
N.add(
    "vs100r_ctrl_marshall_shift",
    ctrl["param_ssm_wavenet_marshall"]["median_best_shift"],
    CTRL,
    text=neg(ctrl["param_ssm_wavenet_marshall"]["median_best_shift"]),
)
N.add(
    "vs100r_ctrl_training_shift",
    ctrl["vs100r_r2_training"]["median_best_shift"],
    CTRL,
    2,
    text=signed(ctrl["vs100r_r2_training"]["median_best_shift"], 2),
)
GD = "demo/capture/valvestate/group_delay.json"
gd = load(GD)["summary"]["model_to_device"]
band = [gd[b][0] for b in ("500", "1000", "2000", "4000", "8000")]
N.add("vs100r_gd_min", min(band), GD, 2)
N.add("vs100r_gd_max", max(band), GD, 2)
LEAD = "diagnosis/vs100r_192k/bigmuff_lead.json"
lead = [v["median_delay"] for v in load(LEAD).values()]
N.add("vs100r_bigmuff_lead_min", min(lead), LEAD, 3, text=neg(min(lead), 3))
N.add("vs100r_bigmuff_lead_max", max(lead), LEAD, 3, text=neg(max(lead), 3))
N.add("vs100r_bigmuff_lead_runs", len(lead), LEAD)

# --- 3. Training length per take (E-0050, F-0037) -----------------------------------

P = "demo/train_length/eval_train_length.json"
A = "demo/train_length/eval_train_length_aligned.json"
H = "demo/train_length/eval_train_length_posthoc_test_tau.json"
PAR = "demo/train_length/parity_gpu_cpu.json"
PRED = "demo/train_length/predictions.md"
p, a, h = load(P), load(A), load(H)
N.add("trainlen_n_takes", len(p["takes"]), P)
N.add("trainlen_n_pairs", p["runs"]["vs100r48fast_film_s42"]["last"]["n_pairs"], P)
N.add("trainlen_n_runs", len(p["runs"]), P)
N.add("trainlen_rule", a["rule"], A, None)
N.add(
    "trainlen_thr_verdict",
    grab(PRED, "« meilleur » si d ≤ " + MINUS + r"(\d+,\d+)"),
    PRED,
    1,
)
N.add("trainlen_thr_accept", grab(PRED, r"« acceptable » si d ≤ \+(\d+,\d+)"), PRED, 2)
N.add("trainlen_thr_accept_seed", grab(PRED, r"chaque d_i ≤ \+(\d+,\d+)"), PRED, 2)


def reading(key: str, src: str, block: dict) -> None:
    sp = block["s_pairs"]
    N.add(f"{key}_d", sp["d"], src, text=signed(sp["d"]))
    N.add(
        f"{key}_di",
        " ; ".join(signed(sp["d_i"][s]) for s in ("42", "43", "44")),
        src,
        None,
    )
    N.add(f"{key}_label", sp["verdict"], src, None)
    N.add(f"{key}_acceptable", "oui" if sp["acceptable"] else "non", src, None)


reading("trainlen_p_l10", P, p["readings"]["L10_last"])
reading("trainlen_p_l20", P, p["readings"]["L20_last"])
reading("trainlen_a_l10", A, a["readings"]["L10"])
reading("trainlen_a_l20", A, a["readings"]["L20"])
reading("trainlen_h_l10", H, h["readings"]["L10"])
reading("trainlen_h_l20", H, h["readings"]["L20"])
N.add(
    "trainlen_a_l20_d42",
    a["readings"]["L20"]["s_pairs"]["d_i"]["42"],
    A,
    text=signed(a["readings"]["L20"]["s_pairs"]["d_i"]["42"]),
)
N.add("trainlen_a_l10_ratio", math.exp(a["readings"]["L10"]["s_pairs"]["d"]), A, 2)
N.add("trainlen_h_l10_ratio", math.exp(h["readings"]["L10"]["s_pairs"]["d"]), H, 2)
for arm in ("C", "L10", "L20"):
    runs = [r for r in a["runs"].values() if r["arm"] == arm]
    taus = [r["tau"] for r in runs]
    k = arm.lower()
    N.add(f"trainlen_tau_{k}_min", min(taus), A, 2)
    N.add(f"trainlen_tau_{k}_max", max(taus), A, 2)
    shifts = [
        p["runs"][name]["last"]["s_pairs"] - run["s_pairs"]
        for name, run in a["runs"].items()
        if run["arm"] == arm
    ]
    N.add(f"trainlen_div_{k}", math.exp(statistics.mean(shifts)), f"{P}; {A}", 1)
    htaus = [r["tau"] for name, r in h["runs"].items() if a["runs"][name]["arm"] == arm]
    N.add(f"trainlen_htau_{k}_min", min(htaus), H, 2)
    N.add(f"trainlen_htau_{k}_max", max(htaus), H, 2)
N.add("trainlen_grid_edge_runs", sum(r["grid_edge"] for r in a["runs"].values()), A)
N.add(
    "trainlen_parity",
    load(PAR)["max_abs_ln_esr_gap"],
    PAR,
    text=sci(load(PAR)["max_abs_ln_esr_gap"]),
)
edge = a["edge_effect"]
N.add("trainlen_edge", edge["energy"], A, text=sci(edge["energy"]))
N.add("trainlen_edge_bound", edge["bound"], A, text=sci(edge["bound"]))

# --- 4. Polarity trap (E-0005, F-0005, T2) ------------------------------------------

INV = "paper/icassp2027/data/polarity_ssmzoh_seed42.json"
GUARD42 = "paper/icassp2027/data/polarity_ssmzoh_guard_seed42.json"
PUB = "paper/icassp2027/data/published_bigmuff.json"
inv, g42, pub = load(INV), load(GUARD42), load(PUB)
N.add("polar_inv_step", inv["global_step"], INV)
N.add("polar_inv_esr", inv["test_as_is"]["metric/test/esr"], INV, 2)
N.add("polar_inv_esr_negated", inv["test_negated"]["metric/test/esr"], INV, 3)
N.add("polar_inv_l1", inv["test_as_is"]["metric/test/l1"], INV, 4)
N.add("polar_inv_l1_negated", inv["test_negated"]["metric/test/l1"], INV, 4)
N.add("polar_inv_mrstft", inv["test_as_is"]["metric/test/mrstft"], INV, 3)
N.add("polar_inv_mrstft_negated", inv["test_negated"]["metric/test/mrstft"], INV, 3)
mean_abs = inv["test_target_mean_abs"]
N.add("polar_mean_abs_y", mean_abs, INV, 4)
N.add("polar_two_mean_abs_y", 2 * mean_abs, INV, 4)
N.add("polar_guard42_esr", g42["test_as_is"]["metric/test/esr"], GUARD42, 4)
N.add("polar_guard42_esr_negated", g42["test_negated"]["metric/test/esr"], GUARD42, 2)
inverted = {m: v["test"] for m, v in pub["models"].items() if v["test"]["esr"] > 1}
N.add("polar_pub_n_models", len(pub["models"]), PUB)
N.add("polar_pub_n_inverted", len(inverted), PUB)
N.add("polar_pub_names", ", ".join(sorted(inverted)), PUB, None)
l1s = [v["l1"] for v in inverted.values()]
esrs = [v["esr"] for v in inverted.values()]
N.add("polar_pub_l1_min", min(l1s), PUB, 4)
N.add("polar_pub_l1_max", max(l1s), PUB, 4)
N.add("polar_pub_esr_min", min(esrs), PUB, 2)
N.add("polar_pub_esr_max", max(esrs), PUB, 2)
N.add("polar_pub_l1_ratio_min", min(l1s) / (2 * mean_abs), f"{PUB}; {INV}", 2)
N.add("polar_pub_l1_ratio_max", max(l1s) / (2 * mean_abs), f"{PUB}; {INV}", 2)
others = [v["test"]["esr"] for v in pub["models"].values() if v["test"]["esr"] <= 1]
N.add("polar_pub_others_esr_max", max(others), PUB, 2)

GUARDED = [
    "ssmzoh_guard_seed42",
    "ssmzoh_guard_seed42_repeat",
    "ssmzoh_guard_seed43",
    "ssmzoh_guard_seed43_repeat",
    "ssmzoh_guard_seed44",
    "ssmzoh_guard_seed44_repeat",
    "s4tfl16_nowd_guard_seed42",
    "s4tfl16_nowd_guard_seed43",
    "s4tfl16_nowd_guard_seed44",
]
flips = {r: load(bench(r))["polarity_flips"] for r in GUARDED}
assert all(load(bench(r))["polarity_guard"] for r in GUARDED)
src_guard = "; ".join(bench(r) for r in GUARDED)
N.add("polar_guard_runs", len(GUARDED), src_guard)
N.add("polar_guard_flips", sum(len(f) for f in flips.values()), src_guard)
N.add("polar_guard_last_epoch", max(e for f in flips.values() for e in f), src_guard)
zoh = [len(flips[r]) for r in GUARDED if r.startswith("ssmzoh")]
N.add("polar_guard_zoh_flips_min", min(zoh), src_guard)
N.add("polar_guard_zoh_flips_max", max(zoh), src_guard)
w = load(bench("ssmzoh_guard_seed42"))["loss_weights"]
N.add("polar_w_l1", w["l1"], bench("ssmzoh_guard_seed42"), 1)
N.add("polar_w_mrstft", w["mrstft"], bench("ssmzoh_guard_seed42"), 1)
FREE = ["ssmwavenet_seed42", "ssmwavenet_seed43", "ssmwavenet_seed44"]
free = [load(bench(r))["test_last"]["metric/test/esr"] for r in FREE]
src_free = "; ".join(bench(r) for r in FREE)
N.add("polar_free_runs", len(FREE), src_free)
N.add("polar_free_esr_max", max(free), src_free, 3)
N.add("polar_free_inverted", sum(x > 1 for x in free), src_free)

# --- 5. Quiet DI: quantisation residue amplified (E-0031, F-0029) --------------------

QF, CZ = "diagnosis/noise/quiet_frames.json", "diagnosis/noise/control_zero_di.json"
qf, cz = load(QF), load(CZ)
levels = [r["quiet"]["L"] for r in qf["rows"].values()]
med_l = qf["median"]["quiet"]["L"]
target_db = statistics.median(r["quiet_target_dbfs"] for r in qf["rows"].values())
N.add("quiet_n_models", len(qf["rows"]), QF)
N.add("quiet_n_quiet_windows", qf["windows"]["quiet"], QF)
N.add("quiet_l_median", med_l, QF, 1)
N.add("quiet_l_min", min(levels), QF, 1)
N.add("quiet_l_max", max(levels), QF, 1)
N.add(
    "quiet_excess_db",
    10 * math.log10(med_l),
    QF,
    1,
    text=signed(10 * math.log10(med_l), 1),
)
N.add("quiet_coherence", qf["median"]["quiet"]["C"], QF, 4)
N.add("quiet_coherence_loud", qf["median"]["loud"]["C"], QF, 2)
N.add("quiet_target_dbfs", target_db, QF, text=neg(target_db))
N.add(
    "quiet_model_dbfs",
    target_db + 10 * math.log10(med_l),
    QF,
    text=neg(target_db + 10 * math.log10(med_l)),
)
N.add("quiet_di_floor_dbfs", qf["di_floor_dbfs"], QF, text=neg(qf["di_floor_dbfs"]))
N.add("quiet_gain_db", target_db + 10 * math.log10(med_l) - qf["di_floor_dbfs"], QF, 0)
N.add("quiet_zero_di_db", cz["median_db"], CZ, text=neg(cz["median_db"]))
N.add("quiet_verdict", qf["verdict"], QF, None)

N.write(HERE / "numbers.json")
print(f"{len(N.data)} keys -> {(HERE / 'numbers.json').relative_to(ROOT)}")
