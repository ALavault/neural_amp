#!/usr/bin/env python3
"""Build the paper's result macros, table and figure from recorded runs.

Inputs: paper/icassp2027/data/published_bigmuff.json (appendix tables of
Comunità et al.) and demo/nablafx_bench/*.json (scripts/product_nablafx_bench.py).
Outputs: paper/icassp2027/results.tex and paper/icassp2027/fig_pareto.pdf.
Groups without runs print as [pending], so the draft always compiles.
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
RUNS = ROOT / "demo/nablafx_bench"

ESR, L1, MRSTFT = "metric/test/esr", "loss_scaled/test/l1", "loss_scaled/test/mrstft"
GROUPS = [
    ("s4-l-16", "S4-L-16"),
    ("s4-tf-l-16", "S4-TF-L-16"),
    ("ssm-wavenet", "SSM-WaveNet"),
]
PUBLISHED_S4 = ["S4-S-16", "S4-L-16", "S4-TF-S-16", "S4-TF-L-16"]


def params_value(text: str) -> float:
    return float(text[:-1]) * 1e3 if text.endswith("k") else float(text)


def params_label(count: int) -> str:
    return f"{count / 1e3:.1f}k"


def load_runs() -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {key: [] for key, _ in GROUPS}
    for path in sorted(RUNS.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        runs[record["model"]].append(record)
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


def table(runs: dict[str, list[dict]]) -> str:
    rows = [
        r"\begin{table}[t]",
        r"\caption{Test results on the Big Muff under the NablAFx protocol. Losses are"
        r" unweighted. Re-runs and SSM-WaveNet: mean $\pm$ standard deviation over"
        r" $n$ seeds.}",
        r"\label{tab:results}",
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
            f" & {test['mrstft']:.4f} & {test['esr']:.4f} \\\\"
        )
    for key, label in GROUPS:
        group = runs[key]
        if key == "s4-l-16":
            rows += [
                r"\midrule",
                r"\multicolumn{5}{@{}l}{\emph{Re-run with the same code as ours}} \\",
            ]
        if key == "ssm-wavenet":
            rows += [r"\midrule", r"\multicolumn{5}{@{}l}{\emph{Proposed}} \\"]
        params = params_label(group[0]["parameters"]) if group else r"\pending"
        tests = [record["test_last"] for record in group]
        rows.append(
            f"{label} ($n={len(group)}$) & {params}"
            f" & {mean_std([t[L1] * 1e3 for t in tests], 1)}"
            f" & {mean_std([t[MRSTFT] for t in tests], 4)}"
            f" & {mean_std([t[ESR] for t in tests], 4)} \\\\"
        )
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(rows)


def figure(runs: dict[str, list[dict]]) -> None:
    plt.rcParams.update({"font.size": 7, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    for name, model in PUBLISHED["models"].items():
        if name.startswith("GB"):
            continue
        x, y = params_value(model["params"]), model["test"]["esr"]
        ax.scatter(x, y, s=10, color="0.65", zorder=2)
        if name in PUBLISHED_S4:
            ax.annotate(
                name,
                (x, y),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=6,
                color="0.35",
            )
    styles = {
        "s4-l-16": ("s", "tab:blue"),
        "s4-tf-l-16": ("D", "tab:blue"),
        "ssm-wavenet": ("*", "tab:red"),
    }
    for key, label in GROUPS:
        group = runs[key]
        if not group:
            continue
        esr = [record["test_last"][ESR] for record in group]
        mean = statistics.fmean(esr)
        marker, color = styles[key]
        ax.errorbar(
            group[0]["parameters"],
            mean,
            yerr=[[mean - min(esr)], [max(esr) - mean]],
            fmt=marker,
            color=color,
            markersize=6 if marker == "*" else 4,
            capsize=2,
            zorder=3,
            label=f"{label} (ours, n={len(group)})"
            if key == "ssm-wavenet"
            else f"{label} re-run (n={len(group)})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Test ESR")
    ax.grid(True, which="both", linewidth=0.3, color="0.85")
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    fig.tight_layout(pad=0.2)
    fig.savefig(PAPER / "fig_pareto.pdf")


def main() -> None:
    runs = load_runs()
    s4tfl = PUBLISHED["models"]["S4-TF-L-16"]
    ssm, rerun = runs["ssm-wavenet"], runs["s4-tf-l-16"]
    ssm_parameters = SSMWaveNet(num_blocks=8, channels=16, state_dim=4).param_count()
    if any(record["parameters"] != ssm_parameters for record in ssm):
        raise RuntimeError("SSM-WaveNet runs do not all use the paper configuration")

    def peak(group: list[dict]) -> str:
        return f"{max(r['peak_gpu_gib'] for r in group):.1f}" if group else r"\pending"

    macros = {
        "pending": r"\textbf{[pending]}",
        "PubSFourTFLesr": f"{s4tfl['test']['esr']:.4f}",
        "PubSFourTFLparams": s4tfl["params"],
        "SSMparams": params_label(ssm_parameters),
        "SSMparamsExact": f"{ssm_parameters:,}".replace(",", "{,}"),
        "SSMn": str(len(ssm)),
        "SSMesrMeanStd": mean_std([r["test_last"][ESR] for r in ssm], 4),
        "RerunSFourTFLesrMeanStd": mean_std([r["test_last"][ESR] for r in rerun], 4),
        "SSMpeakGiB": peak(ssm),
        "SFourTFLpeakGiB": peak(rerun),
        "ResultsTable": table(runs),
    }
    lines = ["% Generated by scripts/product_paper_results.py; do not edit."]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()]
    (PAPER / "results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figure(runs)
    print({key: len(group) for key, group in runs.items()})


if __name__ == "__main__":
    main()
