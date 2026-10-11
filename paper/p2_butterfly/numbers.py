#!/usr/bin/env python3
"""Numbers of dossier P2 (effet papillon), written to numbers.json.

Reads existing run records and analysis outputs only; trains nothing, runs no model.
Light recomputations (standard deviations, rank correlations, the two-way split of
diagnosis/seeds/variance_sources.py, its bootstrap with the same generator) reproduce
recorded values. Pre-registered thresholds are read from the pre-registration texts.
The local plateau fluctuation (correction of 2026-10-02) and the monitored-quantity
replay need the validation curves of demo/runs/*/logs/metrics.csv, which are not
tracked by git.

    uv run python paper/p2_butterfly/numbers.py
"""

# ruff: noqa: RUF001  (French typography in labels and patterns)
from __future__ import annotations

import sys
from pathlib import Path

# This file is named numbers.py: drop its directory from the import path so that
# the stdlib module `numbers` (imported by numpy, statistics, ...) is not shadowed.
sys.path[0] = str(Path(__file__).resolve().parents[1])

import csv  # noqa: E402
import itertools  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import statistics as st  # noqa: E402

import numpy as np  # noqa: E402
from numbers_lib import ROOT, Numbers, fr, load, sci  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
BENCH = "demo/nablafx_bench"
BUTTERFLY = "demo/butterfly"
SEEDS = "diagnosis/seeds"
BFLY = "diagnosis/butterfly"
PILOT_A = f"{BFLY}/pilot_A_results.json"
PILOT_C = f"{BFLY}/pilot_C_results.json"
PILOT_CX = f"{BFLY}/pilot_C_extension_results.json"
PILOT_CX_CH = f"{BFLY}/pilot_C_extension_channel.json"
GLOBAL = f"{SEEDS}/global_factor.json"
VARSRC = f"{SEEDS}/variance_sources.json"
PATIENCE = f"{BFLY}/patience_law.json"


def bench(run: str) -> dict:
    return load(f"{BENCH}/{run}.json")


def esr(run: str) -> float:
    return bench(run)["test_last"]["metric/test/esr"]


def bpath(*runs: str) -> str:
    return "; ".join(f"{BENCH}/{r}.json" for r in runs)


def sd_log(values) -> float:
    return st.stdev(math.log(v) for v in values)


def gmean(values) -> float:
    return math.exp(st.fmean(math.log(v) for v in values))


def md(path: str, pattern: str) -> float:
    """A number written in a pre-registration or result text (French decimal comma)."""
    match = re.search(pattern, (ROOT / path).read_text())
    if match is None:
        raise ValueError(f"{path}: pattern {pattern!r} not found")
    return float(match.group(1).replace(",", ".").replace("−", "-"))


def frs(x: float, digits: int = 3) -> str:
    """fr() with a typographic minus and an explicit plus."""
    text = fr(x, digits)
    return text.replace("-", "−") if x < 0 else "+" + text


def pct(x: float, digits: int = 0) -> str:
    return fr(100 * x, digits)


def tens(x: float) -> str:
    """Rounded to the nearest ten, as the records write ratios to the threshold."""
    return str(int(round(x, -1)))


def chi2_ci(s: float, df: int) -> tuple[float, float]:
    """95 % interval of a standard deviation, chi-square with df degrees of freedom."""
    return (
        s * math.sqrt(df / stats.chi2.ppf(0.975, df)),
        s * math.sqrt(df / stats.chi2.ppf(0.025, df)),
    )


def exact_spearman(x, y) -> tuple[float, float]:
    """Spearman rho and its exact two-sided permutation p (all n! orderings)."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    perms = np.array(list(itertools.permutations(ry)))
    rx_c = rx - rx.mean()
    pc = perms - perms.mean(1, keepdims=True)
    r = pc @ rx_c / np.sqrt((pc**2).sum(1) * (rx_c**2).sum())
    return rho, float(np.mean(np.abs(r) >= abs(rho) - 1e-12))


def bootstrap_ranking(scores: np.ndarray, logged: bool) -> dict:
    """Bootstrap over segments (columns) of the ranking of runs (rows) by mean score.

    Same generator and draw order as diagnosis/seeds/variance_sources.py and the
    first loop of diagnosis/butterfly/validation_noise.py, so "kept" reproduces their
    recorded share. Added here (post hoc, 2026-10-11): how often the best run keeps its
    rank, the median number of discordant pairs, and the largest gap in mean score
    between two runs whose order flips in at least 5 % of draws.
    """
    means = scores.mean(1)
    order = np.argsort(means)
    rows, cols = scores.shape
    rng = np.random.default_rng(0)
    kept = best = 0
    flips = np.zeros((rows, rows))
    discordant = []
    pairs = list(itertools.combinations(range(rows), 2))
    for _ in range(10_000):
        m = scores[:, rng.integers(0, cols, cols)].mean(1)
        kept += np.array_equal(np.argsort(m), order)
        best += int(np.argmin(m) == order[0])
        d = 0
        for i, j in pairs:
            if (means[i] - means[j]) * (m[i] - m[j]) < 0:
                flips[i, j] += 1
                d += 1
        discordant.append(d)
    # Gaps in log units: test scores are already log ESR; validation losses are logged
    # after averaging.
    level = means if logged else np.log(means)
    gaps = [abs(level[i] - level[j]) for i, j in pairs if flips[i, j] >= 500]
    return {
        "kept": kept / 10_000,
        "best": best / 10_000,
        "pairs": len(pairs),
        "discordant": float(np.median(discordant)),
        "gap": max(gaps) if gaps else 0.0,
        "range": float(level.max() - level.min()),
    }


def decompose(table: np.ndarray) -> dict:
    """diagnosis/seeds/variance_sources.py:decompose, same generator and draws."""
    logs = np.log(table)
    grand = logs.mean()
    run, segment = logs.mean(1) - grand, logs.mean(0) - grand
    interaction = logs - grand - run[:, None] - segment[None, :]
    boot = bootstrap_ranking(logs, logged=True)
    return {
        "run": float(run.std(ddof=1)),
        "segment": float(segment.std(ddof=1)),
        "interaction": float(interaction.std(ddof=1)),
        **boot,
    }


def metrics_csv(run: str) -> str:
    return f"demo/runs/nablafx_{run}/logs/metrics.csv"


def validation_by_epoch(run: str, column: str = "loss/val/tot") -> dict[int, float]:
    with (ROOT / metrics_csv(run)).open() as stream:
        return {
            int(r["epoch"]): float(r[column])
            for r in csv.DictReader(stream)
            if r.get(column) and r.get("epoch")
        }


def protocol(n: Numbers) -> None:
    ref = bench("ssmzoh_guard_seed42")
    src = bpath("ssmzoh_guard_seed42")
    n.add("proto_lr", ref["lr"], src, 2)
    n.add("proto_mrstft_weight", ref["loss_weights"]["mrstft"], src, 1)
    n.add("proto_clip", int(ref["gradient_clip_val"]), src)
    n.add("proto_params_ssm", ref["parameters"], src, text="11 329")
    s4 = bench("s4tfl16_seed42")
    n.add("proto_params_s4", s4["parameters"], bpath("s4tfl16_seed42"), text="70 193")
    n.add(
        "proto_params_ratio",
        s4["parameters"] / ref["parameters"],
        bpath("ssmzoh_guard_seed42", "s4tfl16_seed42"),
        1,
    )
    theta = bench("pilotC_decide_seed42")["plateau_threshold"]
    n.add("proto_theta", theta, bpath("pilotC_decide_seed42"), text=sci(theta, 0))

    pol = "paper/icassp2027/data/polarity_ssmzoh_seed42.json"
    p = load(pol)
    n.add("proto_inverted_esr", p["test_as_is"]["metric/test/esr"], pol, 2)
    n.add("proto_inverted_negated_esr", p["test_negated"]["metric/test/esr"], pol, 3)
    n.add("proto_inverted_step", p["global_step"], pol, text="5 100")
    n.add("proto_two_mean_abs", 2 * p["test_target_mean_abs"], pol, 4)
    pub_path = "paper/icassp2027/data/published_bigmuff.json"
    pub = load(pub_path)["models"]
    inverted = {m: v for m, v in pub.items() if v["test"]["esr"] > 1}
    l1 = [v["test"]["l1"] for v in inverted.values()]
    n.add("proto_published_inverted_n", len(inverted), pub_path)
    n.add("proto_published_l1_min", min(l1), pub_path, 4)
    n.add("proto_published_l1_max", max(l1), pub_path, 4)

    guarded = [
        f"ssmzoh_guard_seed{s}{r}" for s in (42, 43, 44) for r in ("", "_repeat")
    ]
    guarded += [f"s4tfl16_nowd_guard_seed{s}" for s in (42, 43, 44)]
    flips = [bench(r)["polarity_flips"] for r in guarded]
    n.add("proto_guard_runs", len(guarded), bpath(*guarded))
    n.add("proto_guard_flips", sum(len(f) for f in flips), bpath(*guarded))
    n.add("proto_guard_last_epoch", max(max(f) for f in flips if f), bpath(*guarded))


def bench_comparison(n: Numbers) -> None:
    ssm = [f"ssmzoh_guard_seed{s}{r}" for s in (42, 43, 44) for r in ("", "_repeat")]
    s4c = [f"s4tfl16_nowd_guard_seed{s}" for s in (42, 43, 44)]
    s4r = [f"s4tfl16_seed{s}" for s in (42, 43, 44)]
    g_ssm, g_s4c, g_s4r = (gmean(esr(r) for r in g) for g in (ssm, s4c, s4r))
    n.add("bench_ssm_gmean", g_ssm, bpath(*ssm), 3)
    n.add("bench_s4_changes_gmean", g_s4c, bpath(*s4c), 3)
    n.add("bench_s4_released_gmean", g_s4r, bpath(*s4r), 3)
    n.add("bench_gmean_ratio", g_s4c / g_ssm, bpath(*ssm, *s4c), 1)
    lower = all(
        max(esr(f"ssmzoh_guard_seed{s}"), esr(f"ssmzoh_guard_seed{s}_repeat"))
        < esr(f"s4tfl16_nowd_guard_seed{s}")
        for s in (42, 43, 44)
    )
    assert lower, "SSM-WaveNet no longer lower at each seed"
    p_min = stats.mannwhitneyu(
        [1, 2, 3], [4, 5, 6], alternative="two-sided", method="exact"
    ).pvalue
    n.add("bench_mw_pmin", p_min, bpath(*ssm[::2], *s4c), 2)
    n.add("bench_raw_sd", sd_log(esr(r) for r in ssm), bpath(*ssm), 3)
    spread = f"{SEEDS}/spread.txt"
    shares = [
        int(x)
        for x in re.findall(r"\((\d+)% of the mean\)", (ROOT / spread).read_text())
    ]
    n.add("bench_se_share_min", min(shares), spread)
    n.add("bench_se_share_max", max(shares), spread)


def seed_structure(n: Numbers) -> None:
    g = load(GLOBAL)
    fs = [
        v["anova_log_esr"]["ms_run"] / v["anova_log_esr"]["ms_run_x_segment"]
        for k, v in g.items()
        if "anova_log_esr" in v
    ]
    n.add("seed_global_f_min", min(fs), GLOBAL, 0)
    n.add("seed_global_f_max", max(fs), GLOBAL, 0)
    n.add("seed_run_sd_ssm", g["ssm-changes"]["anova_log_esr"]["run_sd"], GLOBAL, 2)
    n.add(
        "seed_run_sd_s4_changes", g["s4-changes"]["anova_log_esr"]["run_sd"], GLOBAL, 2
    )
    n.add(
        "seed_run_sd_s4_released",
        g["s4-released"]["anova_log_esr"]["run_sd"],
        GLOBAL,
        2,
    )

    nest = g["nested"]["ssm-changes"]
    within, seed_sd, f = nest["within_seed_sd"], nest["seed_sd"], nest["F"]
    n.add("nested_within_sd", within, GLOBAL, 2)
    n.add("nested_seed_sd", seed_sd, GLOBAL, 2)
    n.add("nested_f", f, GLOBAL, 1)
    n.add("nested_p", stats.f.sf(f, *nest["df"]), GLOBAL, 2)
    share = seed_sd**2 / (seed_sd**2 + within**2)
    n.add("nested_seed_share", share, GLOBAL, text=pct(share))
    gpu = max(abs(v) for v in nest["log_ratio_mean_esr"].values())
    n.add("nested_gpu_factor", math.exp(gpu), GLOBAL, 2)
    lo, hi = chi2_ci(within, 3)
    n.add("nested_within_ci_lo", lo, GLOBAL, 2)
    n.add("nested_within_ci_hi", hi, GLOBAL, 2)

    # Per-segment ratio seed 43 / seed 42 (first runs), where_audible.md qualification.
    v = load(VARSRC)["graines"]
    a = np.array(v["ssmzoh_guard_seed42"]["last"])
    b = np.array(v["ssmzoh_guard_seed43"]["last"])
    ratio = b / a
    n.add("seed_segment_ratio_min", ratio.min(), VARSRC, 2)
    n.add("seed_segment_ratio_max", ratio.max(), VARSRC, 2)
    # The record's "median" is the geometric one, exp(median of the log ratios).
    logmed = float(np.median(np.log(ratio)))
    n.add("seed_segment_ratio_median", math.exp(logmed), VARSRC, 2)
    n.add("seed_segment_ratio_logmedian", logmed, VARSRC, 2)
    n.add("seed_segment_ratio_sdlog", float(np.log(ratio).std(ddof=1)), VARSRC, 3)


def s4_structure(n: Numbers) -> None:
    path = f"{SEEDS}/s4_nested_verdict.json"
    s = load(path)["three_pairs"]
    n.add("s4_within_sd", s["within_sd"], path, 3)
    n.add("s4_seed_sd", s["seed_sd"], path, text="0")
    n.add("s4_f", s["F"], path, 2)
    n.add("s4_p", s["p"], path, 2)
    n.add(
        "s4_gpu_factor", math.exp(max(abs(x) for x in s["delta_log"].values())), path, 2
    )
    runs = ("s4tfl16_seed42_repeat", "s4tfl16_seed42_redo")
    n.add("s4_stop_repeat", bench(runs[0])["stopped_epoch"], bpath(*runs))
    n.add("s4_stop_redo", bench(runs[1])["stopped_epoch"], bpath(*runs))


def evaluation(n: Numbers) -> dict:
    v = load(VARSRC)
    seeds = decompose(np.array([r["last"] for r in v["graines"].values()]))
    kids = decompose(np.array([r["last"] for r in v["enfants f100"].values()]))
    n.add("rank_seed_run_effect", seeds["run"], VARSRC, 3)
    n.add("rank_seed_segment_effect", seeds["segment"], VARSRC, 3)
    n.add("rank_seed_interaction", seeds["interaction"], VARSRC, 3)
    share = seeds["interaction"] / seeds["run"]
    n.add("rank_interaction_share", share, VARSRC, text=pct(share))
    n.add("rank_child_run_effect", kids["run"], VARSRC, 3)
    n.add("rank_seed_kept", seeds["kept"], VARSRC, text=pct(seeds["kept"]))
    n.add("rank_seed_reordered", 1 - seeds["kept"], VARSRC, text=pct(1 - seeds["kept"]))
    n.add("rank_child_kept", kids["kept"], VARSRC, text=pct(kids["kept"]))
    seeds_best = decompose(np.array([r["best"] for r in v["graines"].values()]))
    kids_best = decompose(np.array([r["best"] for r in v["enfants f100"].values()]))
    n.add("rank_seed_run_effect_best", seeds_best["run"], VARSRC, 3)
    n.add("rank_child_run_effect_best", kids_best["run"], VARSRC, 3)
    gaps = [
        abs(np.mean(r["best"]) - np.mean(r["last"]))
        for group in ("graines", "enfants f100")
        for r in v[group].values()
        if "best" in r
    ]
    n.add("rank_best_last_runs", len(gaps), VARSRC)
    n.add("rank_best_last_max_gap", max(gaps), VARSRC, 4)
    noise = f"{BFLY}/validation_noise.json"
    vn = load(noise)
    kept = vn["ranking_unchanged"]
    n.add("rank_val_children_kept", kept, noise, text=fr(100 * kept, 1))
    # Best child by mean validation loss, and how often it keeps its name (recorded).
    names = list(vn["per_segment"])
    val = np.array([vn["per_segment"][r] for r in names])
    best_name = names[int(np.argmin(val.mean(1)))][-12:]
    best_share = vn["bootstrap_winners"][best_name] / sum(
        vn["bootstrap_winners"].values()
    )
    n.add("rank_val_best_kept", best_share, noise, text=fr(100 * best_share, 1))
    # Size of the inversions (computed here; the full-ranking share matches the record).
    vb = bootstrap_ranking(val, logged=False)
    assert abs(vb["kept"] - kept) < 1e-12
    n.add("rank_val_pairs", vb["pairs"], noise)
    n.add("rank_val_discordant_median", round(vb["discordant"]), noise)
    n.add("rank_val_flip_gap", vb["gap"], noise, 3)
    n.add("rank_val_spread", vb["range"], noise, 3)
    for name, group in (("seed", seeds), ("child", kids)):
        n.add(
            f"rank_{name}_best_kept", group["best"], VARSRC, text=pct(group["best"], 2)
        )
        n.add(f"rank_{name}_pairs", group["pairs"], VARSRC)
        n.add(f"rank_{name}_discordant_median", round(group["discordant"]), VARSRC)
        n.add(f"rank_{name}_flip_gap", group["gap"], VARSRC, 3)
        n.add(f"rank_{name}_spread", group["range"], VARSRC, 3)
    return seeds


def data_chain(n: Numbers, seeds: dict) -> None:
    runs = [f"splitfix_seed{s}" for s in range(42, 48)]
    values = [esr(r) for r in runs]
    s = sd_log(values)
    n.add("pilot_e_sd", s, bpath(*runs), 3)
    n.add("pilot_e_gmean", gmean(values), bpath(*runs), 4)
    n.add(
        "pilot_e_threshold",
        md(f"{SEEDS}/pilot_E_split.md", r"s ≤ (0,\d+)\. La disp"),
        f"{SEEDS}/pilot_E_split.md",
        2,
    )
    lo, hi = chi2_ci(s, len(values) - 1)
    n.add("pilot_e_ci_lo", lo, bpath(*runs), 2)
    n.add("pilot_e_ci_hi", hi, bpath(*runs), 2)
    n.add(
        "pilot_e_nondeterministic",
        sum(not bench(r)["deterministic"] for r in runs),
        bpath(*runs),
    )
    n.add("pilot_e_runs", len(runs), bpath(*runs))


def threshold_tuning(n: Numbers) -> None:
    runs = [f"plateau2e2_seed{s}{r}" for s in (42, 43, 44) for r in ("", "_repeat")]
    ref = [f"ssmzoh_guard_seed{s}{r}" for s in (42, 43, 44) for r in ("", "_repeat")]
    theta = bench(runs[0])["plateau_threshold"]
    n.add("pilot_f_theta", theta, bpath(runs[0]), text=sci(theta, 0))
    n.add("pilot_f_sd", sd_log(esr(r) for r in runs), bpath(*runs), 3)
    g, g_ref = gmean(esr(r) for r in runs), gmean(esr(r) for r in ref)
    n.add("pilot_f_gmean", g, bpath(*runs), 3)
    n.add("pilot_f_quality_factor", g / g_ref, bpath(*runs, *ref), 1)
    pre_f = f"{SEEDS}/pilot_F_threshold.md"
    n.add(
        "pilot_f_g_threshold",
        md(pre_f, r"\*\*Correctif utile\*\* : s ≤ 0,25 \*\*et\*\* g ≤ (0,\d+)"),
        pre_f,
        3,
    )
    n.add(
        "pilot_f_s_threshold",
        md(pre_f, r"\*\*Correctif utile\*\* : s ≤ (0,\d+)"),
        pre_f,
        2,
    )
    for key, group in (("pilot_f_steps", runs), ("pilot_f_steps_ref", ref)):
        steps = round(st.fmean(bench(r)["global_step"] for r in group))
        n.add(key, steps, bpath(*group), text=f"{steps:,}".replace(",", " "))

    sim_path = f"{SEEDS}/plateau_simulation.json"
    sim = load(sim_path)
    kids = {
        k: v
        for k, v in sim.items()
        if k.startswith("enfants") and k.split("|")[1] == "20"
    }
    n.add(
        "plateau_sim_sd_protocol",
        sim["enfants f100 (un pas float32)|20|0.0001"]["sd_first"],
        sim_path,
        0,
    )
    best = min(kids.values(), key=lambda v: v["sd_first"])
    n.add("plateau_sim_sd_min", best["sd_first"], sim_path, 0)
    n.add("plateau_sim_count_at_min_lo", min(best["count"]), sim_path)
    n.add("plateau_sim_count_at_min_hi", max(best["count"]), sim_path)
    n.add("plateau_sim_thresholds", len(kids), sim_path)


def plateau_noise(n: Numbers) -> None:
    margin_path = f"{BFLY}/plateau_margin.json"
    margin = load(margin_path)
    whole = [v["median_relative_jump"] / v["threshold"] for v in margin.values()]
    n.add("plateau_whole_run_ratio_min", min(whole), margin_path, 0)
    n.add("plateau_whole_run_ratio_max", max(whole), margin_path, 0)
    first, margins, sources = [], [], [margin_path]
    for run, v in margin.items():
        losses = validation_by_epoch(run)
        sources.append(metrics_csv(run))
        epochs = sorted(losses)
        for i, h in enumerate(v["halvings"]):
            window = [e for e in epochs if h["halving"] - 40 <= e < h["halving"]]
            local = st.median(
                abs(losses[b] - losses[a]) / losses[a]
                for a, b in itertools.pairwise(window)
            )
            margins.append(h["closest_approach"] / local)
            if i == 0:
                first.append(local)
                if run.endswith("decide_k0"):
                    k0_first = h["closest_approach"] / local
    src = "; ".join(sources)
    theta = next(iter(margin.values()))["threshold"]
    n.add("plateau_local_fluct_min", min(first), src, text=pct(min(first), 1))
    n.add("plateau_local_fluct_max", max(first), src, text=pct(max(first), 1))
    n.add(
        "plateau_local_ratio_min",
        min(first) / theta,
        src,
        text=tens(min(first) / theta),
    )
    n.add(
        "plateau_local_ratio_max",
        max(first) / theta,
        src,
        text=tens(max(first) / theta),
    )
    margins.sort()
    n.add("plateau_halvings_n", len(margins), src)
    n.add("plateau_margin_min", margins[0], src, 2)
    n.add("plateau_margin_max_main", margins[-2], src, 2)
    n.add("plateau_margin_outlier", margins[-1], src, 2)
    n.add("plateau_margins_below_one", sum(m < 1 for m in margins), src)
    n.add("plateau_k0_first_margin", k0_first, src, 2)


def monitored_quantity(n: Numbers) -> None:
    sys.path.insert(0, str(ROOT / SEEDS))
    from plateau_simulation import halvings

    observed = {s: v["observed"] for s, v in _pilot_c_first_halving().items()}
    seeds = list(range(42, 50))
    runs = [f"pilotC_decide_seed{s}" for s in seeds]

    def firsts(column: str) -> list[int]:
        out = []
        for run in runs:
            curve = validation_by_epoch(run, column)
            fired = halvings([curve[e] for e in sorted(curve)], 1e-4, 20)
            out.append(fired[0] if fired else -1)
        return out

    total, esr_col = firsts("loss/val/tot"), firsts("metric/val/esr")
    mrstft, l1 = firsts("loss/val/mrstft"), firsts("loss/val/l1")
    src = (
        "diagnosis/seeds/plateau_simulation.py; diagnosis/seeds/monitored_quantity.py; "
        + "; ".join(metrics_csv(r) for r in runs)
    )
    # monitored_quantity.py compares with the real epochs read from the lr column.
    real = [observed[s] - 1 for s in seeds]
    n.add("mq_replay_exact", sum(a == b for a, b in zip(total, real, strict=True)), src)
    n.add("mq_runs", len(runs), src)
    n.add("mq_total_median", round(st.median(total)), src)
    n.add("mq_esr_median", round(st.median(esr_col)), src)
    n.add("mq_total_sd", st.stdev(total), src, 0)
    n.add("mq_esr_sd", st.stdev(esr_col), src, 0)
    n.add(
        "mq_mrstft_same", sum(a == b for a, b in zip(mrstft, total, strict=True)), src
    )
    n.add("mq_l1_same", sum(a == b for a, b in zip(l1, total, strict=True)), src)


def _pilot_c_first_halving() -> dict[int, dict]:
    runs = load(PATIENCE)["predictions"]["independent"]["per_run"]
    return {
        int(k.removeprefix("pilotC_decide_seed")): v
        for k, v in runs.items()
        if k.startswith("pilotC")
    }


def pilot_a(n: Numbers) -> None:
    a = load(PILOT_A)
    g, verdict = a["groups"], a["verdict"]
    pre = f"{BFLY}/pilot_A.md"
    n.add("pilot_a_s_decide", verdict["s_decide"], PILOT_A, 3)
    n.add("pilot_a_s_replay", verdict["s_replay"], PILOT_A, 3)
    n.add("pilot_a_delta_ratio", verdict["mean_abs_delta_ratio"], PILOT_A, 1)
    n.add("pilot_a_threshold_s", md(pre, r"s\(décide\) ≥ (0,\d+), s\(rejoue\)"), pre, 2)
    n.add("pilot_a_replay_bound", verdict["s_decide"] / 2, PILOT_A, 3)
    n.add("pilot_a_p0_s", verdict["fork5"]["s_decide"], PILOT_A, 3)
    dec, rep = g["fork100_decide"]["test_esr"], g["fork100_replay"]["test_esr"]
    n.add("pilot_a_control_esr", dec[0], PILOT_A, 4)
    n.add("pilot_a_decide_esr_max", max(dec), PILOT_A, 4)
    n.add("pilot_a_replay_esr_min", min(rep), PILOT_A, 4)
    n.add(
        "pilot_a_out_decide_min",
        min(g["fork100_decide"]["output_esr_vs_k0"]),
        PILOT_A,
        text=sci(min(g["fork100_decide"]["output_esr_vs_k0"])),
    )
    n.add(
        "pilot_a_out_decide_max",
        max(g["fork100_decide"]["output_esr_vs_k0"]),
        PILOT_A,
        text=sci(max(g["fork100_decide"]["output_esr_vs_k0"])),
    )
    n.add(
        "pilot_a_out_replay_min",
        min(g["fork100_replay"]["output_esr_vs_k0"]),
        PILOT_A,
        text=sci(min(g["fork100_replay"]["output_esr_vs_k0"])),
    )
    n.add(
        "pilot_a_out_replay_max",
        max(g["fork100_replay"]["output_esr_vs_k0"]),
        PILOT_A,
        text=sci(max(g["fork100_replay"]["output_esr_vs_k0"])),
    )
    n.add("pilot_a_k1_delta", g["fork100_decide"]["delta"][0], PILOT_A, 3)
    n.add("pilot_a_k1_first_halving", g["fork100_decide"]["halvings"][1][0], PILOT_A)
    n.add(
        "pilot_a_control_first_halving", g["fork100_decide"]["halvings"][0][0], PILOT_A
    )
    n.add("pilot_a_runs_per_arm", len(dec), PILOT_A)
    n.add("pilot_a_perturbed_per_arm", len(g["fork100_decide"]["delta"]), PILOT_A)
    for arm in ("decide", "replay"):
        delta = g[f"fork100_{arm}"]["delta"]
        n.add(f"pilot_a_delta_{arm}_min", min(delta), PILOT_A, text=frs(min(delta), 3))
        n.add(f"pilot_a_delta_{arm}_max", max(delta), PILOT_A, text=frs(max(delta), 3))
    ulp = g["fork100_decide"]["nudge"][1]["largest_change"]
    n.add("pilot_a_largest_nudge", ulp, PILOT_A, text=sci(ulp, 2))
    assert (
        verdict["A_supported"]
        and verdict["replay_valid"]
        and verdict["P0_positive_control"]
    )

    # Early forks (preliminary).
    f5, f25 = verdict["fork5"], verdict["fork25"]
    n.add("fork5_s_decide", f5["s_decide"], PILOT_A, 3)
    n.add("fork5_s_replay", f5["s_replay"], PILOT_A, 3)
    n.add("fork5_delta_ratio", f5["mean_abs_delta_ratio"], PILOT_A, 2)
    n.add("fork5_replay_bound", f5["s_decide"] / 2, PILOT_A, 3)
    n.add("fork5_replay_k1_delta", g["fork5_replay"]["delta"][0], PILOT_A, 2)
    n.add("fork25_s_decide", f25["s_decide"], PILOT_A, 3)
    n.add("fork25_s_replay", f25["s_replay"], PILOT_A, 3)
    n.add("fork25_s_ratio", f25["s_replay"] / f25["s_decide"], PILOT_A, 2)
    n.add("fork25_delta_ratio", f25["mean_abs_delta_ratio"], PILOT_A, 2)
    for fork, s in (("5", f5), ("25", f25), ("100", verdict)):
        n.add(f"fork{fork}_containment", s["s_replay"] / s["s_decide"], PILOT_A, 2)


def fork100_readings(n: Numbers) -> None:
    dw_path = f"{BFLY}/decided_when.json"
    dw = load(dw_path)
    pairs = dw["fork100_decide"]["pairs"]
    big = [p["settles_at_epoch"] for p in pairs if p["gap_log_esr"] >= 0.29]
    small = [p for p in pairs if p["gap_log_esr"] <= 0.12]
    n.add("decided_big_pairs", len(big), dw_path)
    n.add("decided_big_first", min(big), dw_path)
    n.add("decided_big_last", max(big), dw_path)
    n.add(
        "decided_big_gap_min",
        min(p["gap_log_esr"] for p in pairs if p["gap_log_esr"] >= 0.29),
        dw_path,
        2,
    )
    n.add("decided_small_gap_max", max(p["gap_log_esr"] for p in small), dw_path, 2)
    settled = [
        p["settles_at_epoch"] for p in small if p["settles_at_epoch"] is not None
    ]
    n.add("decided_small_first", min(settled), dw_path)
    n.add("decided_small_last", max(settled), dw_path)
    n.add(
        "decided_small_never",
        sum(p["settles_at_epoch"] is None for p in small),
        dw_path,
    )
    n.add("decided_control_stop", dw["fork100_decide"]["terminal"][0]["epoch"], dw_path)
    # decided_when.py reads each child's terminal validation loss from its metrics.csv
    # (decided_when.json keeps it rounded to 5 decimals, too coarse for the replay arm).
    a = load(PILOT_A)["groups"]
    for arm in ("decide", "replay"):
        runs = [f"butterfly_ssm_seed42_f100_{arm}_k{k}" for k in range(5)]
        terminal = []
        for run in runs:
            curve = validation_by_epoch(run)
            terminal.append(curve[max(curve)])
        sd_val = st.stdev(math.log(x) for x in terminal)
        sd_esr = st.stdev(math.log(x) for x in a[f"fork100_{arm}"]["test_esr"])
        src = f"{PILOT_A}; " + "; ".join(metrics_csv(r) for r in runs)
        n.add(f"decided_val_sd_{arm}", sd_val, src, 3)
        n.add(f"decided_test_over_val_{arm}", sd_esr / sd_val, src, 1)
        n.add(f"decided_beyond_two_{arm}", sd_esr / sd_val / 2, src, 1)
    dilation = (
        n.data["decided_val_sd_decide"]["value"]
        / n.data["decided_val_sd_replay"]["value"]
    )
    n.add(
        "decided_val_dilation",
        dilation,
        n.data["decided_val_sd_decide"]["source"]
        + "; "
        + n.data["decided_val_sd_replay"]["source"],
        0,
    )

    def slope(paths: list[str]) -> float:
        xy = []
        for p in paths:
            t = load(p)["test_last"]
            xy.append((math.log(t["metric/test/l1"]), math.log(t["metric/test/esr"])))
        x, y = np.array(xy).T
        return float(np.polyfit(x, y, 1)[0])

    fixe = [f"{BENCH}/pilotC_fixe_seed{s}.json" for s in range(42, 58)]
    decide = [f"{BENCH}/pilotC_decide_seed{s}.json" for s in range(42, 58)]
    kids = [f"{BUTTERFLY}/butterfly_ssm_seed42_f100_decide_k{k}.json" for k in range(5)]
    n.add("decided_slope_fixe", slope(fixe), "; ".join(fixe), 2)
    n.add("decided_slope_decide", slope(decide), "; ".join(decide), 2)
    n.add("decided_slope_children", slope(kids), "; ".join(kids), 2)

    lmc_path = f"{BFLY}/mode_connectivity.json"
    lmc = load(lmc_path)["children"]
    k1 = lmc["butterfly_ssm_seed42_f100_decide_k1"]
    k3 = lmc["butterfly_ssm_seed42_f100_decide_k3"]
    n.add("lmc_mid_esr", k1["esr"][2], lmc_path, 4)
    n.add("lmc_end_control", k1["esr"][0], lmc_path, 4)
    n.add("lmc_end_child", k1["esr"][-1], lmc_path, 4)
    n.add("lmc_segments_above", k1["segments_above_midline"], lmc_path)
    rho, p = stats.spearmanr(
        [c["l2_distance"] for c in lmc.values()], [c["barrier"] for c in lmc.values()]
    )
    n.add("lmc_rho_distance", rho, lmc_path, 2)
    n.add("lmc_p_distance", p, lmc_path, 3)
    n.add("lmc_children", len(lmc), lmc_path)
    n.add("lmc_barrier_ratio", k1["barrier"] / k3["barrier"], lmc_path, 1)
    n.add("lmc_distance_k1", k1["l2_distance"], lmc_path, 2)
    n.add("lmc_distance_k3", k3["l2_distance"], lmc_path, 2)

    eq_path = f"{BFLY}/is_it_eq.json"
    eq = load(eq_path)["decide_k1"]
    n.add("eq_esr_raw", eq["esr_raw"], eq_path, 4)
    n.add("eq_esr_filtered", eq["esr_after_best_filter"], eq_path, 4)
    n.add(
        "eq_removed",
        1 - eq["esr_after_best_filter"] / eq["esr_raw"],
        eq_path,
        text=pct(1 - eq["esr_after_best_filter"] / eq["esr_raw"]),
    )
    bands = eq["bands_db"]
    n.add(
        "eq_band_db_min", min(bands.values()), eq_path, text=frs(min(bands.values()), 2)
    )
    n.add(
        "eq_band_db_max", max(bands.values()), eq_path, text=frs(max(bands.values()), 2)
    )
    n.add("eq_band_lo_hz", min(int(b) for b in bands), eq_path)
    n.add("eq_band_hi_hz", max(int(b) for b in bands), eq_path)

    when_path = f"{BFLY}/when_audible.json"
    when = load(when_path)
    for i in (0, 1):
        n.add(f"aud_fall_db_seg{i}", -when[i]["by_kind_db"]["chute"], when_path, 1)
        n.add(f"aud_hold_db_seg{i}", -when[i]["by_kind_db"]["tenue"], when_path, 1)

    where_path = f"{BFLY}/where_audible.json"
    rows = load(where_path)
    share = [r["share"] for r in rows]
    for key, name in (
        ("rms_dbfs", "rms"),
        ("crest_db", "crest"),
        ("centroid_hz", "centroid"),
        ("quiet_share", "quiet"),
    ):
        rho = stats.spearmanr([r[key] for r in rows], share)[0]
        n.add(f"aud_rho_{name}", rho, where_path, text=frs(rho, 2))
    n.add("aud_segments", len(rows), where_path)
    # ESR of the control against the device on the concatenated test set (the measure of
    # output_esr_vs_k0): segment ESRs weighted by the target energy (equal lengths).
    weights = [10 ** (r["rms_dbfs"] / 10) for r in rows]
    concat = sum(
        w * r["esr_control_device"] for w, r in zip(weights, rows, strict=True)
    ) / sum(weights)
    n.add("pilot_a_control_concat_esr", concat, where_path, 4)
    n.add("aud_seg0_share", rows[0]["share"], where_path, 2)
    n.add("aud_seg11_share", rows[11]["share"], where_path, 2)
    n.add(
        "aud_seg0_quiet",
        rows[0]["quiet_share"],
        where_path,
        text=pct(rows[0]["quiet_share"]),
    )
    n.add(
        "aud_seg11_quiet",
        rows[11]["quiet_share"],
        where_path,
        text=pct(rows[11]["quiet_share"]),
    )

    listen = f"{BFLY}/listening.md"
    n.add(
        "listen_real_control",
        md(listen, r"\| 2 \| ([\d,]+)e−2 \|") * 1e-2,
        listen,
        text=sci(md(listen, r"\| 2 \| ([\d,]+)e−2 \|") * 1e-2, 2),
    )
    n.add(
        "listen_control_child",
        md(listen, r"\| 2 \| [\d,]+e−2 \| ([\d,]+)e−2 \|") * 1e-2,
        listen,
        text=sci(md(listen, r"\| 2 \| [\d,]+e−2 \| ([\d,]+)e−2 \|") * 1e-2, 2),
    )


def pilot_c(n: Numbers) -> None:
    c = load(PILOT_C)
    seeds = list(range(42, 50))
    dec = [c["esr"]["decide"][str(s)] for s in seeds]
    fix = [c["esr"]["fixe"][str(s)] for s in seeds]
    pre = f"{BFLY}/pilot_C_common_schedule.md"
    n.add("pilot_c_sd_decide", c["s"]["decide"], PILOT_C, 3)
    n.add("pilot_c_sd_fixe", c["s"]["fixe"], PILOT_C, 3)
    n.add("pilot_c_ratio", c["ratio"], PILOT_C, 2)
    n.add("pilot_c_support", md(pre, r"ni soutenue \(≤ (0,\d+)\)"), pre, 2)
    n.add("pilot_c_refute", md(pre, r"ni réfutée \(≥ (0,\d+)\)"), pre, 2)
    f = (c["s"]["decide"] / c["s"]["fixe"]) ** 2
    n.add("pilot_c_f", f, PILOT_C, 2)
    n.add("pilot_c_f_p", stats.f.sf(f, 7, 7), PILOT_C, 3)
    loo = []
    for drop in range(8):
        keep = [i for i in range(8) if i != drop]
        loo.append(sd_log(fix[i] for i in keep) / sd_log(dec[i] for i in keep))
    n.add("pilot_c_loo_min", min(loo), PILOT_C, 2)
    n.add("pilot_c_loo_max", max(loo), PILOT_C, 2)
    gain = st.fmean(math.log(fx / d) for d, fx in zip(dec, fix, strict=True))
    n.add("pilot_c_mean_gain", gain, PILOT_C, text=frs(gain, 3))
    n.add("pilot_c_mean_factor", math.exp(gain), PILOT_C, 2)
    keep = [i for i, s in enumerate(seeds) if s not in (48, 49)]
    sd_d6, sd_f6 = sd_log(dec[i] for i in keep), sd_log(fix[i] for i in keep)
    n.add("pilot_c_sd_decide_wo", sd_d6, PILOT_C, 3)
    n.add("pilot_c_sd_fixe_wo", sd_f6, PILOT_C, 3)
    n.add("pilot_c_ratio_wo", sd_f6 / sd_d6, PILOT_C, 2)
    diffs = [math.log(fx) - math.log(d) for d, fx in zip(dec, fix, strict=True)]
    n.add("pilot_c_arm_residual", st.stdev(diffs) / math.sqrt(2), PILOT_C, 3)
    n.add("pilot_c_esr_48", c["esr"]["decide"]["48"], PILOT_C, 3)
    n.add("pilot_c_esr_49", c["esr"]["decide"]["49"], PILOT_C, 3)

    # Channel test (post hoc), each arm against a split property.
    props_path = f"{SEEDS}/split_property.json"
    props = load(props_path)
    quiet = [props[str(s)]["val_quiet_share_mean"] for s in seeds]
    halving = [_pilot_c_first_halving()[s]["observed"] for s in seeds]
    src = f"{PILOT_C}; {props_path}; {PATIENCE}"
    for name, x in (("quiet", quiet), ("halving", halving)):
        for arm, y in (("decide", dec), ("fixe", fix)):
            rho, p = exact_spearman(x, [math.log(v) for v in y])
            n.add(f"pilot_c_rho_{name}_{arm}", rho, src, text=frs(rho, 2))
            n.add(f"pilot_c_p_{name}_{arm}", p, src, 3)


def pilot_c_extension(n: Numbers) -> None:
    x = load(PILOT_CX)
    pre = f"{BFLY}/pilot_C_extension.md"
    n.add("pilot_cx_r_early", x["ratios"]["precoce"], PILOT_CX, 2)
    n.add("pilot_cx_r_late", x["ratios"]["tardif"], PILOT_CX, 2)
    n.add("pilot_cx_pred_early", md(pre, r"R_précoce ≤ (0,\d+) et"), pre, 1)
    n.add("pilot_cx_pred_late", md(pre, r"R_tardif ≥ (0,\d+)\.\*\*"), pre, 1)
    n.add("pilot_cx_r_eight", x["ratio_new_eight"], PILOT_CX, 2)
    n.add("pilot_cx_r_sixteen", x["ratio_sixteen"], PILOT_CX, 2)
    seeds = sorted(x["esr"], key=int)
    for group, key in (("precoce", "early"), ("tardif", "late")):
        members = [s for s in seeds if x["groups"][s] == group]
        loo = []
        for drop in members:
            keep = [s for s in members if s != drop]
            loo.append(
                sd_log(x["esr"][s]["fixe"] for s in keep)
                / sd_log(x["esr"][s]["decide"] for s in keep)
            )
        n.add(f"pilot_cx_loo_{key}_min", min(loo), PILOT_CX, 2)
        n.add(f"pilot_cx_loo_{key}_max", max(loo), PILOT_CX, 2)
    worst = max(x["esr"][s]["decide"] for s in seeds)
    n.add("pilot_cx_worst_decide", worst, PILOT_CX, 3)
    gain = st.fmean(
        math.log(x["esr"][s]["fixe"] / x["esr"][s]["decide"]) for s in seeds
    )
    n.add("pilot_cx_mean_gain", gain, PILOT_CX, text=frs(gain, 3))
    n.add(
        "pilot_cx_sd_decide", sd_log(x["esr"][s]["decide"] for s in seeds), PILOT_CX, 2
    )
    n.add("pilot_cx_sd_fixe", sd_log(x["esr"][s]["fixe"] for s in seeds), PILOT_CX, 2)

    c = load(PILOT_C)
    all_dec = {
        **{int(s): v for s, v in c["esr"]["decide"].items()},
        **{int(s): x["esr"][s]["decide"] for s in seeds},
    }
    all_fix = {
        **{int(s): v for s, v in c["esr"]["fixe"].items()},
        **{int(s): x["esr"][s]["fixe"] for s in seeds},
    }
    keep = [s for s in all_dec if s not in (48, 49)]
    sd_d, sd_f = sd_log(all_dec[s] for s in keep), sd_log(all_fix[s] for s in keep)
    n.add("pilot_cx_sd_decide_wo", sd_d, f"{PILOT_C}; {PILOT_CX}", 3)
    n.add("pilot_cx_sd_fixe_wo", sd_f, f"{PILOT_C}; {PILOT_CX}", 3)
    n.add("pilot_cx_r_wo", sd_f / sd_d, f"{PILOT_C}; {PILOT_CX}", 2)

    ch = load(PILOT_CX_CH)["spearman"]
    for key, name in (
        ("quiet_share->decide", "quiet"),
        ("first_halving->decide", "halving"),
        ("first_halving->gain", "gain"),
        ("decide_last_epoch->gain", "stop_gain"),
    ):
        n.add(f"pilot_cx_rho_{name}", ch[key][0], PILOT_CX_CH, text=frs(ch[key][0], 2))
        n.add(f"pilot_cx_p_{name}", ch[key][1], PILOT_CX_CH, 3)

    # Sign of ln(fixe/decide) against the halving group, sixteen seeds (post hoc).
    first = _pilot_c_first_halving()
    groups_path = f"{BFLY}/pilot_C_extension_groups.json"
    groups = load(groups_path)
    follow = 0
    for s in sorted(all_dec):
        early = (
            groups[str(s)]["group"] == "precoce"
            if str(s) in groups
            else first[s]["observed"] - 1 < 200
        )
        follow += (all_fix[s] < all_dec[s]) == early
    n.add(
        "pilot_cx_sign_follows",
        follow,
        f"{PILOT_C}; {PILOT_CX}; {groups_path}; {PATIENCE}",
    )
    n.add("pilot_cx_seeds_total", len(all_dec), f"{PILOT_C}; {PILOT_CX}")


def paired(n: Numbers) -> None:
    p3_path, p5_path = f"{SEEDS}/paired_protocol.json", f"{SEEDS}/paired_extension.json"
    p3, p5 = load(p3_path), load(p5_path)
    pre = f"{SEEDS}/paired_protocol.md"
    n.add("paired_r3", p3["residual"], p3_path, 3)
    n.add("paired_r3_ci_lo", p3["residual_ci95"][0], p3_path, 3)
    n.add("paired_r3_ci_hi", p3["residual_ci95"][1], p3_path, 2)
    n.add(
        "paired_support", md(pre, r"\*\*Appariement soutenu\*\* : r ≤ (0,\d+)"), pre, 2
    )
    n.add("paired_refute", md(pre, r"\*\*Appariement réfuté\*\* : r > (0,\d+)"), pre, 2)
    n.add("paired_factor3", math.exp(-p3["mean_difference"]), p3_path, 2)
    n.add("paired_r5", p5["r5"], p5_path, 3)
    n.add("paired_r5_ci_lo", p5["ci95"][0], p5_path, 3)
    n.add("paired_r5_ci_hi", p5["ci95"][1], p5_path, 3)
    splits = sorted(p5["differences"], key=int)
    runs = [f"paired_{m}_s{s}" for s in splits for m in ("ssm_wavenet", "s4_tf_l_16")]
    ssm = np.log([esr(f"paired_ssm_wavenet_s{s}") for s in splits])
    s4 = np.log([esr(f"paired_s4_tf_l_16_s{s}") for s in splits])
    assert np.allclose(ssm - s4, [p5["differences"][s] for s in splits])
    src = f"{p5_path}; " + bpath(*runs)
    n.add("paired_factor5", math.exp(-float(np.mean(ssm - s4))), src, 2)
    n.add("paired_sd_ssm", float(ssm.std(ddof=1)), src, 3)
    n.add("paired_sd_s4", float(s4.std(ddof=1)), src, 3)
    corr = float(np.corrcoef(ssm, s4)[0, 1])
    z, half = math.atanh(corr), 1.96 / math.sqrt(len(splits) - 3)
    n.add("paired_corr", corr, src, 2)
    n.add(
        "paired_corr_ci_lo", math.tanh(z - half), src, text=frs(math.tanh(z - half), 2)
    )
    n.add("paired_corr_ci_hi", math.tanh(z + half), src, 2)
    r0 = math.sqrt((ssm.var(ddof=1) + s4.var(ddof=1)) / 2)
    n.add("paired_r_uncorrelated", r0, src, 3)
    n.add(
        "paired_removed_share",
        1 - (p5["r5"] / r0) ** 2,
        src,
        text=pct(1 - (p5["r5"] / r0) ** 2),
    )
    n.add("paired_splits", len(splits), src)
    same = all(all(v.values()) for v in p5["order_same"].values())
    assert same
    n.add("paired_pilot_c_residual_ref", md(pre, r"au pilote C \((0,\d+)\)"), pre, 3)


def pilot_g(n: Numbers) -> None:
    path = f"{SEEDS}/split_property_verdict.json"
    v = load(path)["correlations"]
    pre = f"{SEEDS}/pilot_G_split_property.md"
    n.add(
        "pilot_g_rho",
        v["val_rms_min"]["rho"],
        path,
        text=frs(v["val_rms_min"]["rho"], 2),
    )
    n.add("pilot_g_p", v["val_rms_min"]["p"], path, 2)
    n.add("pilot_g_support", md(pre, r"\|ρ\| ≥ (0,\d+), seuil"), pre, 2)
    n.add("pilot_g_refute", md(pre, r"\*\*H2 réfutée\*\* : \|ρ\| < (0,\d+)"), pre, 2)
    props = load(f"{SEEDS}/split_property.json")
    c = load(PILOT_C)
    seeds = [str(s) for s in range(42, 50)]
    rho, p = exact_spearman(
        [props[s]["val_quiet_share_mean"] for s in seeds],
        [math.log(c["esr"]["decide"][s]) for s in seeds],
    )
    src = f"{path}; {SEEDS}/split_property.json; {PILOT_C}"
    tested = sum(1 for x in v.values() if x.get("rho") is not None)
    n.add("pilot_g_quiet_rho", rho, src, text=frs(rho, 2))
    n.add("pilot_g_quiet_p", p, src, 3)
    n.add("pilot_g_tested", tested, path)
    n.add("pilot_g_quiet_sidak", 1 - (1 - p) ** tested, src, 3)
    within = load(GLOBAL)["nested"]["ssm-changes"]["within_seed_sd"]
    reliability = 1 - within**2 / c["s"]["decide"] ** 2
    true_rho = md(pre, r"Un ρ vrai de (0,\d+) se")
    n.add("pilot_g_reliability", reliability, f"{GLOBAL}; {PILOT_C}", 2)
    n.add("pilot_g_attenuation", math.sqrt(reliability), f"{GLOBAL}; {PILOT_C}", 2)
    n.add("pilot_g_true_rho", true_rho, pre, 2)
    n.add(
        "pilot_g_seen_rho",
        true_rho * math.sqrt(reliability),
        f"{GLOBAL}; {PILOT_C}; {pre}",
        2,
    )


def theory_t3(n: Numbers) -> None:
    law = load(PATIENCE)
    pre = f"{BFLY}/patience_law.md"
    ind = law["predictions"]["independent"]
    runs = ind["per_run"]
    n.add("t3_runs", len(runs), PATIENCE)
    n.add(
        "t3_gate_exact",
        law["gate"]["exact_by_shift"][str(law["gate"]["shift"])],
        PATIENCE,
    )
    n.add("t3_vr_median", law["variance_ratio"]["median"], PATIENCE, 3)
    n.add("t3_vr_max", max(law["variance_ratio"]["per_run"].values()), PATIENCE, 3)
    n.add(
        "t3_vr_below",
        sum(v <= 0.3 for v in law["variance_ratio"]["per_run"].values()),
        PATIENCE,
    )
    n.add("t3_vr_independent", md(pre, r"VR\(10\) ≤ (0,\d+) → \*\*bruit"), pre, 1)
    covered = sum(v["covered"] for v in runs.values())
    n.add("t3_covered", covered, PATIENCE)
    n.add("t3_coverage", ind["coverage"], PATIENCE, 2)
    n.add("t3_support", md(pre, r"≥ (0,\d+) \(au moins 13"), pre, 2)
    n.add("t3_refute", md(pre, r"≤ (0,\d+) \(au plus 8\)"), pre, 2)
    late = sum(
        (not v["covered"]) and v["observed"] > v["0.0001"]["q90"] for v in runs.values()
    )
    n.add("t3_misses_late", late, PATIENCE)
    n.add("t3_shift_up", int(ind["median_shift_vs_1e-4"]["0.001"]), PATIENCE)
    n.add(
        "t3_shift_down", int(ind["median_shift_vs_1e-4"]["1e-05"]), PATIENCE, text="−1"
    )
    widths = [v["0.0001"]["q90"] - v["0.0001"]["q10"] for v in runs.values()]
    n.add("t3_width_median", st.median(widths), PATIENCE, 0)
    # One curve, one child: first halving of decide k3 replayed at the protocol
    # threshold and at ten times it (plateau_simulation.json, curve index), plus
    # the gate's one-epoch convention (patience_law.json): epochs of the child.
    sim_path = f"{SEEDS}/plateau_simulation.json"
    sim = load(sim_path)
    shift = law["gate"]["shift"]
    key = "enfants f100 (un pas float32)|20|"
    for name, theta in (("proto", "0.0001"), ("up", "0.001")):
        first = sim[key + theta]["first"][3] + shift
        n.add(f"t3_counter_k3_{name}", first, f"{sim_path}; {PATIENCE}")
    # VR(10) of an AR(1) residual with the lag-1 autocorrelation seen by the 2026-10-01
    # re-reading: (1 - phi^10) / (10 (1 - phi)); white noise gives 1/10.
    phi = md(pre, r"autocorrélation de 0,2 à (0,\d) au retard 1")
    n.add("t3_ar1_phi", phi, pre, 1)
    n.add("t3_ar1_vr10", (1 - phi**10) / (10 * (1 - phi)), pre, 3)


def determinism(n: Numbers) -> None:
    path = f"{BFLY}/determinism.json"
    d = load(path)
    for model in ("ssm", "s4"):
        comps = {tuple(c["pair"]): c for c in d[model]["comparisons"]}
        det = comps[(f"determinism_{model}_det_a", f"determinism_{model}_det_b")]
        nondet = comps[
            (f"determinism_{model}_nondet_a", f"determinism_{model}_nondet_b")
        ]
        assert (
            det["weights_equal"]
            and det["adam_moments_equal"]
            and not nondet["weights_equal"]
        )
        n.add(f"det_{model}_nondet_max_diff", nondet["max_weight_diff"], path, 2)
        minutes = {r["run_id"]: r["minutes"] for r in d[model]["runs"]}
        slow = st.fmean(
            minutes[f"determinism_{model}_det_{x}"] for x in "ab"
        ) / st.fmean(minutes[f"determinism_{model}_nondet_{x}"] for x in "ab")
        n.add(f"det_{model}_cost", slow - 1, path, text=pct(slow - 1))
    gate_path = f"{SEEDS}/repro_gate.json"
    gate = load(gate_path)["results"]
    n.add("det_gate_paths", sum(v["identical"] for v in gate.values()), gate_path)
    n.add(
        "det_gate_values_min",
        min(v["values_compared"] for v in gate.values()),
        gate_path,
    )


def retired(n: Numbers) -> None:
    """Values of claims withdrawn from the icassp2027-draft tag, superseded readings."""
    g = load(GLOBAL)
    conds = [v for v in g.values() if "worst_over_best" in v]
    quiet = [v["worst_over_best"]["quiet_share_of_gap"] for v in conds]
    one = [v["worst_over_best"]["quiet_share_if_one_factor"] for v in conds]
    n.add("retired_quiet_share_min", min(quiet), GLOBAL, text=pct(min(quiet)))
    n.add("retired_quiet_share_max", max(quiet), GLOBAL, text=pct(max(quiet)))
    n.add("retired_one_factor_min", min(one), GLOBAL, text=pct(min(one)))
    n.add("retired_one_factor_max", max(one), GLOBAL, text=pct(max(one)))
    runs = ("ssmzoh_guard_seed42", "ssmzoh_guard_seed42_repeat")
    n.add("retired_repeat_delta", esr(runs[1]) - esr(runs[0]), bpath(*runs), 3)
    rep = g["nested"]["ssm-changes"]["log_ratio_mean_esr"]["42"]
    n.add("retired_repeat42_log", rep, GLOBAL, text=frs(rep, 2))
    first = [f"ssmzoh_guard_seed{s}" for s in (42, 43, 44)]
    n.add("retired_ssm_mean", st.fmean(esr(r) for r in first), bpath(*first), 3)
    n.add("retired_ssm_std", st.stdev(esr(r) for r in first), bpath(*first), 3)
    cv = {k: g[k]["cv"] for k in ("ssm-changes", "s4-changes")}
    n.add(
        "retired_h4_cv_ssm",
        cv["ssm-changes"]["mean_esr"] / 100,
        GLOBAL,
        text=fr(cv["ssm-changes"]["mean_esr"], 0),
    )
    n.add(
        "retired_h4_cv_ssm_energy",
        cv["ssm-changes"]["energy_weighted_esr"] / 100,
        GLOBAL,
        text=fr(cv["ssm-changes"]["energy_weighted_esr"], 0),
    )
    n.add(
        "retired_h4_cv_s4",
        cv["s4-changes"]["mean_esr"] / 100,
        GLOBAL,
        text=fr(cv["s4-changes"]["mean_esr"], 0),
    )
    n.add(
        "retired_h4_cv_s4_energy",
        cv["s4-changes"]["energy_weighted_esr"] / 100,
        GLOBAL,
        text=fr(cv["s4-changes"]["energy_weighted_esr"], 0),
    )
    red = 1 - cv["ssm-changes"]["energy_weighted_esr"] / cv["ssm-changes"]["mean_esr"]
    n.add("retired_h4_reduction", red, GLOBAL, text=pct(red))
    diag = "DIAGNOSIS_seeds.md"
    n.add(
        "retired_h4_predicted", int(md(diag, r"réduit le CV d'au moins (\d+) %")), diag
    )
    margin_path = f"{BFLY}/plateau_margin.json"
    k0 = load(margin_path)["butterfly_ssm_seed42_f100_decide_k0"]
    n.add(
        "retired_k0_old_margin",
        k0["halvings"][0]["closest_approach"] / k0["median_relative_jump"],
        margin_path,
        2,
    )


def main() -> None:
    n = Numbers()
    protocol(n)
    bench_comparison(n)
    seed_structure(n)
    s4_structure(n)
    seeds = evaluation(n)
    data_chain(n, seeds)
    threshold_tuning(n)
    plateau_noise(n)
    monitored_quantity(n)
    pilot_a(n)
    fork100_readings(n)
    pilot_c(n)
    pilot_c_extension(n)
    paired(n)
    pilot_g(n)
    theory_t3(n)
    determinism(n)
    retired(n)
    n.write(HERE / "numbers.json")
    print(f"{len(n.data)} keys -> {HERE / 'numbers.json'}")


if __name__ == "__main__":
    main()
