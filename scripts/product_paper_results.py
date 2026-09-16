#!/usr/bin/env python3
"""Build the paper's result macros, table and figure from recorded runs.

Inputs, all under version control: paper/icassp2027/data/published_bigmuff.json
(appendix tables of Comunità et al.), demo/nablafx_bench/*.json
(scripts/product_nablafx_bench.py), paper/icassp2027/data/polarity_ssmzoh_seed42.json
(scripts/product_polarity_check.py) and recurrence_check.json
(scripts/product_ssm_recurrence_check.py).
Outputs: paper/icassp2027/results.tex and paper/icassp2027/fig_pareto.pdf.
Variants without runs print as [pending], so the draft always compiles.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fssr_nam.models.ssm_wavenet import SSMWaveNet

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/icassp2027"
PUBLISHED = json.loads(
    (PAPER / "data/published_bigmuff.json").read_text(encoding="utf-8")
)
INVERTED = json.loads(
    (PAPER / "data/polarity_ssmzoh_seed42.json").read_text(encoding="utf-8")
)
RECURRENCE = PAPER / "data/recurrence_check.json"
VALIDATION = ROOT / "diagnosis/seeds/validation_loss.json"
SEGMENTS = PAPER / "data/test_segments.json"
RUNS = ROOT / "demo/nablafx_bench"

ESR, L1, MRSTFT = "metric/test/esr", "metric/test/l1", "metric/test/mrstft"
# (model, discretization, states per channel, no weight decay on state-space
# parameters, polarity guard) -> variant. Any other configuration raises.
VARIANT_OF = {
    ("s4-tf-l-16", None, None, False, False): "s4-released",
    ("s4-tf-l-16", None, None, True, True): "s4-changes",
    ("ssm-wavenet", "zoh", 4, True, True): "ssm",
    ("ssm-wavenet", "free", 4, False, False): "ssm-v1",
}
SECTIONS = [
    ("Re-run, released training", [("s4-released", "S4-TF-L-16")]),
    (
        "Both training changes (Sec.~\\ref{sec:setup})",
        [("s4-changes", "S4-TF-L-16"), ("ssm", "SSM-WaveNet")],
    ),
    ("Ablation: learned $b$, released training", [("ssm-v1", "SSM-WaveNet")]),
]
PUBLISHED_S4 = ["S4-S-16", "S4-L-16", "S4-TF-S-16", "S4-TF-L-16"]


def params_value(text: str) -> float:
    return float(text[:-1]) * 1e3 if text.endswith("k") else float(text)


def params_label(count: int) -> str:
    return f"{count / 1e3:.1f}k"


def variant(record: dict) -> str:
    ssm = record["ssm"] or {}
    key = (
        record["model"],
        ssm.get("discretization", "free") if ssm else None,
        ssm.get("state_dim"),
        bool(record.get("honor_optim")),
        bool(record.get("polarity_guard")),
    )
    return VARIANT_OF[key]


def load_runs() -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {key: [] for key in VARIANT_OF.values()}
    for path in sorted(RUNS.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        # A repeat trains a seed already trained, to measure how much of the
        # spread is GPU non-determinism; it is not a further seed.
        if record["run_id"].endswith("_repeat"):
            continue
        runs[variant(record)].append(record)
    return runs


def mean_std(values: list[float], digits: int) -> str:
    if not values:
        return r"\pending"
    mean = statistics.fmean(values)
    if len(values) == 1:
        return f"{mean:.{digits}f}"
    std = statistics.stdev(values)
    return f"{mean:.{digits}f}\\,{{\\scriptsize$\\pm$\\,{std:.{digits}f}}}"


def best_per_family() -> list[str]:
    best: dict[str, str] = {}
    for name, model in PUBLISHED["models"].items():
        family = name.split("-")[0]
        if family in ("GB", "S4"):
            continue
        if (
            family not in best
            or model["test"]["esr"] < PUBLISHED["models"][best[family]]["test"]["esr"]
        ):
            best[family] = name
    return [best[family] for family in ("LSTM", "TCN", "GCN")] + PUBLISHED_S4


def published_outliers() -> list[str]:
    """Neural models with test ESR above 1, which an inverted output produces."""
    return [
        name
        for name, model in sorted(
            PUBLISHED["models"].items(), key=lambda item: item[1]["test"]["esr"]
        )
        if model["test"]["esr"] > 1 and not name.startswith("GB")
    ]


def table(runs: dict[str, list[dict]]) -> str:
    rows = [
        r"\begin{table}[t]",
        r"\caption{Test results on the Big Muff under the NablAFx protocol, last"
        r" checkpoint. Losses are unweighted. Our runs: mean $\pm$ standard deviation"
        r" over $n$ seeds.}",
        r"\label{tab:results}",
        r"\smallskip",
        r"\centering\footnotesize\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}lrlll@{}}",
        r"\toprule",
        r"Model & Params & L1 ($10^{-3}$) & MR-STFT & ESR \\",
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\emph{Published~\cite{comunita2025frontiers}}} \\",
    ]
    for name in best_per_family():
        test = PUBLISHED["models"][name]["test"]
        rows.append(
            f"{name} & {PUBLISHED['models'][name]['params']} & {test['l1'] * 1e3:.1f}"
            f" & {test['mrstft']:.3f} & {test['esr']:.3f} \\\\"
        )
    for title, members in SECTIONS:
        rows += [r"\midrule", f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\"]
        for key, label in members:
            group = runs[key]
            params = params_label(group[0]["parameters"]) if group else r"\pending"
            tests = [record["test_last"] for record in group]
            rows.append(
                f"{label} ($n={len(group)}$) & {params}"
                f" & {mean_std([t[L1] * 1e3 for t in tests], 1)}"
                f" & {mean_std([t[MRSTFT] for t in tests], 3)}"
                f" & {mean_std([t[ESR] for t in tests], 3)} \\\\"
            )
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(rows)


# Label offsets in points, chosen so the four S4 labels do not overlap.
S4_LABEL_OFFSETS = {
    "S4-S-16": (4, -2),
    "S4-L-16": (-4, 3),
    "S4-TF-S-16": (-4, -6),
    "S4-TF-L-16": (4, -7),
}
ESR_AXIS = (0.03, 1.0)
STYLES = {
    "s4-released": ("D", "tab:blue", "S4-TF-L-16 re-run, released training"),
    "s4-changes": ("s", "tab:cyan", "S4-TF-L-16 re-run, both changes"),
    "ssm": ("*", "tab:red", "SSM-WaveNet, both changes"),
}


def figure(runs: dict[str, list[dict]]) -> list[str]:
    """Draw ESR against parameters; return the published models off the axis."""
    plt.rcParams.update({"font.size": 7, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    off_scale = []
    for name, model in PUBLISHED["models"].items():
        if name.startswith("GB"):
            continue
        x, y = params_value(model["params"]), model["test"]["esr"]
        if y > ESR_AXIS[1]:
            off_scale.append(f"{name} ({y:.2f})")
            continue
        ax.scatter(x, y, s=10, color="0.65", zorder=2)
        if name in PUBLISHED_S4:
            dx, dy = S4_LABEL_OFFSETS[name]
            ax.annotate(
                name,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx > 0 else "right",
                fontsize=6,
                color="0.35",
            )
    for key, (marker, color, label) in STYLES.items():
        group = runs[key]
        if not group:
            continue
        esr = [record["test_last"][ESR] for record in group]
        mean = statistics.fmean(esr)
        ax.errorbar(
            group[0]["parameters"],
            mean,
            yerr=[[mean - min(esr)], [max(esr) - mean]],
            fmt=marker,
            color=color,
            markersize=6 if marker == "*" else 4,
            capsize=2,
            zorder=3,
            label=f"{label} (n={len(group)})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(*ESR_AXIS)
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Test ESR")
    ax.grid(True, which="both", linewidth=0.3, color="0.85")
    # Below the axes: every corner of the plot holds published models.
    ax.legend(
        fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24)
    )
    fig.savefig(PAPER / "fig_pareto.pdf", bbox_inches="tight", pad_inches=0.02)
    return off_scale


def main() -> None:
    runs = load_runs()
    s4tfl = PUBLISHED["models"]["S4-TF-L-16"]
    ssm, rerun, changes = runs["ssm"], runs["s4-released"], runs["s4-changes"]
    parameters = {
        key: SSMWaveNet(8, 16, 4, discretization=discretization).param_count()
        for key, discretization in (("ssm", "zoh"), ("ssm-v1", "free"))
    }
    for key, count in parameters.items():
        if any(record["parameters"] != count for record in runs[key]):
            raise RuntimeError(f"{key} runs do not all have {count} parameters")

    def esr(group: list[dict]) -> str:
        return mean_std([r["test_last"][ESR] for r in group], 3)

    def extreme(group: list[dict], pick) -> str:
        return (
            f"{pick(r['test_last'][ESR] for r in group):.3f}" if group else r"\pending"
        )

    def peak(group: list[dict]) -> str:
        return f"{max(r['peak_gpu_gib'] for r in group):.1f}" if group else r"\pending"

    rerun_losses = [r["test_last"]["loss_scaled/test/tot"] for r in rerun]
    loss_gap = r"\pending"
    if rerun_losses:
        excess = statistics.fmean(rerun_losses) / s4tfl["test"]["tot"] - 1
        loss_gap = f"{100 * excess:.0f}\\,\\%"

    def mean_esr(group: list[dict]) -> float:
        return statistics.fmean(r["test_last"][ESR] for r in group)

    def pair(left: list[dict], right: list[dict]) -> tuple[list[dict], list[dict]]:
        """The runs of both groups whose seed appears in the other, seed by seed."""
        seeds = {r["seed"] for r in left} & {r["seed"] for r in right}
        key = lambda r: r["seed"]  # noqa: E731
        return (
            sorted((r for r in left if r["seed"] in seeds), key=key),
            sorted((r for r in right if r["seed"] in seeds), key=key),
        )

    comparisons = {}
    if ssm and changes and rerun:
        ours, theirs = pair(ssm, changes)
        lower = sum(
            a["test_last"][ESR] < b["test_last"][ESR]
            for a, b in zip(ours, theirs, strict=True)
        )
        after, before = pair(changes, rerun)
        reduction = 1 - mean_esr(after) / mean_esr(before)
        helped = sum(
            a["test_last"][ESR] < b["test_last"][ESR]
            for a, b in zip(after, before, strict=True)
        )
        mine, ablated = pair(ssm, runs["ssm-v1"])
        comparisons = {
            "SSMLowerThanVoneSeeds": str(
                sum(
                    a["test_last"][ESR] < b["test_last"][ESR]
                    for a, b in zip(mine, ablated, strict=True)
                )
            ),
            "VoneLowerSeeds": str(
                sum(
                    b["test_last"][ESR] < a["test_last"][ESR]
                    for a, b in zip(mine, ablated, strict=True)
                )
            ),
            "VonePairedSeeds": str(len(mine)),
            "ChangesReduction": f"{100 * reduction:.0f}\\,\\%",
            "ChangesPairedSeeds": str(len(after)),
            "ChangesLowerSeeds": str(helped),
            "SSMRatio": f"{mean_esr(theirs) / mean_esr(ours):.1f}",
            "ParamRatio": f"{changes[0]['parameters'] / parameters['ssm']:.1f}",
            "SSMLowerSeeds": str(lower),
            "PairedSeeds": str(len(ours)),
        }

    guarded = ssm + changes
    seeds = sorted({r["seed"] for group in runs.values() for r in group})
    seed_list = ", ".join(map(str, seeds[:-1])) + " and " + str(seeds[-1])
    flips = [epoch for r in guarded for epoch in r["polarity_flips"]]
    outliers = published_outliers()
    outlier_l1 = [PUBLISHED["models"][name]["test"]["l1"] for name in outliers]
    recurrence = r"\pending"
    if RECURRENCE.exists():
        check = json.loads(RECURRENCE.read_text(encoding="utf-8"))
        if check["discretization"] == "zoh" and "guard" in check["checkpoint"]:
            mantissa, exponent = f"{check['max_abs_deviation']:.0e}".split("e")
            recurrence = f"{mantissa}\\times10^{{{int(exponent)}}}"

    # An inverted output has L1 = 2 E|y| on any set; the published tables give
    # validation losses too, so the same test applies to them.
    published_val = r"\pending"
    twice_trainval = r"\pending"
    repeat = RUNS / "ssmzoh_guard_seed42_repeat.json"
    repeat_macros = {"RepeatEsr": r"\pending", "RepeatDelta": r"\pending"}
    if repeat.exists():
        again = json.loads(repeat.read_text(encoding="utf-8"))["test_last"][ESR]
        base = next(r["test_last"][ESR] for r in ssm if r["seed"] == 42)
        repeat_macros = {
            "RepeatEsr": f"{again:.3f}",
            "RepeatDelta": f"{abs(again - base):.3f}",
        }

    # Share of the best-to-worst seed gap that sits on the six quietest segments.
    quiet_share = r"\pending"
    if SEGMENTS.exists():
        segments = json.loads(SEGMENTS.read_text(encoding="utf-8"))["runs"]
        shares = []
        for group in runs.values():
            if len(group) < 2:
                continue
            order = sorted(group, key=lambda r: r["test_last"][ESR])
            best, worst = order[0]["run_id"], order[-1]["run_id"]
            gap = order[-1]["test_last"][ESR] - order[0]["test_last"][ESR]
            quiet = segments[worst]["quiet"] - segments[best]["quiet"]
            shares.append(quiet / 2 / gap)
        quiet_share = f"{100 * min(shares):.0f}--{100 * max(shares):.0f}\\,\\%"

    # The validation segment whose relative error is worst, and what it weighs
    # in the validation loss that drives stopping and selection.
    blind_esr = blind_share = r"\pending"
    if VALIDATION.exists():
        checks = json.loads(VALIDATION.read_text(encoding="utf-8"))
        _, check = max(
            checks.items(), key=lambda kv: max(s["esr"] for s in kv[1]["per_segment"])
        )
        blind_esr = f"{max(s['esr'] for s in check['per_segment']):.1f}"
        blind_share = f"{100 * check['share_of_quietest']:.1f}\\,\\%"

    quiet_loud = r"\pending"
    if SEGMENTS.exists():
        ratios = [
            run["quiet"] / run["loud"]
            for run in json.loads(SEGMENTS.read_text(encoding="utf-8"))["runs"].values()
        ]
        quiet_loud = f"{min(ratios):.1f}--{max(ratios):.1f}"
        level = json.loads(SEGMENTS.read_text(encoding="utf-8"))[
            "trainval_target_mean_abs"
        ]
        twice_trainval = f"{2 * level:.4f}"
        inverted_on_val = sorted(
            PUBLISHED["models"][name]["val"]["l1"]
            for name in published_outliers()
            if PUBLISHED["models"][name]["val"]["l1"] > 1.5 * level
        )
        published_val = f"{min(inverted_on_val):.4f} and {max(inverted_on_val):.4f}"

    macros = {
        "pending": r"\textbf{[pending]}",
        "QuietLoudRatioRange": quiet_loud,
        "SpreadQuietShare": quiet_share,
        "ValBlindEsr": blind_esr,
        "ValBlindShare": blind_share,
        **repeat_macros,
        "PubOutlierValLone": published_val,
        "TwiceMeanAbsTrainval": twice_trainval,
        "PubSFourTFLesr": f"{s4tfl['test']['esr']:.3f}",
        "PubSFourTFLparams": s4tfl["params"],
        "SSMparams": params_label(parameters["ssm"]),
        "SSMparamsExact": f"{parameters['ssm']:,}".replace(",", "{,}"),
        "VoneSSMparamsExact": f"{parameters['ssm-v1']:,}".replace(",", "{,}"),
        "SSMn": str(len(ssm)),
        "SeedList": seed_list if len(seeds) > 1 else str(seeds[0]),
        "SSMesrMeanStd": esr(ssm),
        "SSMmrstftMeanStd": mean_std([r["test_last"][MRSTFT] for r in ssm], 3),
        "RerunSFourTFLesrMeanStd": esr(rerun),
        "RerunSFourTFLn": str(len(rerun)),
        "RerunSFourTFLlossGap": loss_gap,
        "ChangesSFourTFLesrMeanStd": esr(changes),
        "ChangesSFourTFLn": str(len(changes)),
        "SSMesrMin": extreme(ssm, min),
        "SSMesrMax": extreme(ssm, max),
        "ChangesSFourTFLesrMin": extreme(changes, min),
        "ChangesSFourTFLesrMax": extreme(changes, max),
        "ChangesSFourTFLmrstftMeanStd": mean_std(
            [r["test_last"][MRSTFT] for r in changes], 3
        ),
        "VoneSSMesr": esr(runs["ssm-v1"]),
        "SSMpeakGiB": peak(ssm),
        "SFourTFLpeakGiB": peak(rerun + changes),
        "GuardFlips": str(len(flips)),
        "GuardRuns": str(len(guarded)),
        "GuardLastFlipEpoch": str(max(flips)) if flips else r"\pending",
        "InvertedStep": f"{INVERTED['global_step']:,}".replace(",", "{,}"),
        "InvertedEsr": f"{INVERTED['test_as_is'][ESR]:.2f}",
        "InvertedLone": f"{INVERTED['test_as_is'][L1]:.4f}",
        "InvertedNegatedEsr": f"{INVERTED['test_negated'][ESR]:.3f}",
        "InvertedNegatedLone": f"{INVERTED['test_negated'][L1]:.4f}",
        "TwiceMeanAbsTarget": f"{2 * INVERTED['test_target_mean_abs']:.4f}",
        "PubOutliers": ", ".join(outliers[:-1]) + " and " + outliers[-1],
        "PubOutlierLoneRange": f"{min(outlier_l1):.4f}--{max(outlier_l1):.4f}",
        "RecurrenceDeviation": recurrence,
        "ResultsTable": table(runs),
    }
    for name in (
        "ChangesReduction",
        "ChangesPairedSeeds",
        "ChangesLowerSeeds",
        "SSMLowerThanVoneSeeds",
        "VoneLowerSeeds",
        "VonePairedSeeds",
        "SSMRatio",
        "ParamRatio",
        "SSMLowerSeeds",
        "PairedSeeds",
    ):
        macros[name] = comparisons.get(name, r"\pending")
    macros["PubOffScale"] = ", ".join(figure(runs))
    lines = ["% Generated by scripts/product_paper_results.py; do not edit."]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()]
    (PAPER / "results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({key: len(group) for key, group in runs.items()})


if __name__ == "__main__":
    main()
