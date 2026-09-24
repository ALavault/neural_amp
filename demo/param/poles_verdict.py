#!/usr/bin/env python3
"""Verdict of G2 (demo/param/predictions_poles.md). Read-only on the runs, CPU.

Written 2026-09-24 before any G2 run. Refuses to conclude before every arm of the
requested splits exists. Each run's last checkpoint is re-evaluated on the nine unseen
test settings, batches of eight as the bench does, with B030_M070_T100_G050 evaluated at
treble 0.1 (primary, demo/param/label_check.md) and 1.0 (secondary); pole arms are also
evaluated with their pole shifts frozen at the central setting (H6). Seen settings come
from the records' val_last_by_setting.

Usage: poles_verdict.py [--splits 925 [736 688]]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import product_nablafx_bench as bench  # noqa: E402
from nablafx.core import BlackBoxModel  # noqa: E402
from nablafx.evaluation.registry import EvaluationRegistry  # noqa: E402

ARMS = {  # arm -> (cond, head)
    "A": ("film", "joint"),
    "B": ("film", "additive"),
    "C": ("poles", "joint"),
    "D": ("poles", "additive"),
}
POLES = ("C", "D")
MISLABELLED = "B030_M070_T100_G050"
TREBLE = 2
EXTREME, INTERPOLATION = "B100_M000_T000_G050", "B010_M030_T070_G050"
MARGIN, LARGE, STOP = math.log(1.15), math.log(1.5), -math.log(2)
ESR = EvaluationRegistry.get_function("esr_loss")
MRSTFT = EvaluationRegistry.get_function("mrstft_loss")


def run_id(arm: str, split: int) -> str:
    if arm == "A" and split == 925:
        return "param_ssm_wavenet_marshall"
    cond, head = ARMS[arm]
    return f"g2_{cond}_{head}_s{split}"


def load(run: str) -> BlackBoxModel:
    record = json.loads((bench.RESULTS_DIR / f"{run}.json").read_text())
    args = argparse.Namespace(model=record["model"], device="marshall", **record["ssm"])
    processor = bench.build_processor(args)
    path = bench.RUNS_DIR / f"nablafx_{run}/checkpoints/last.ckpt"
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    prefix = "model.processor."
    processor.load_state_dict(
        {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    )
    return BlackBoxModel(processor).eval()


@torch.no_grad()
def unseen(model: BlackBoxModel, dataset, frozen: bool = False) -> dict:
    """ESR and MR-STFT per test setting; the mislabelled one under both labels."""
    processor = model.processor
    if frozen:
        moving = processor.pole_shifts
        processor.pole_shifts = lambda c: moving(torch.full_like(c, 0.5))
    groups: dict[str, list[int]] = {}
    for index in range(len(dataset)):
        setting = Path(dataset.samples[index]["target_file"]).parent.name
        groups.setdefault(setting, []).append(index)
    out = {}
    for setting, members in sorted(groups.items()):
        trebles = (0.1, 1.0) if setting == MISLABELLED else (None,)
        for treble in trebles:
            totals = {"esr": 0.0, "mrstft": 0.0}
            for start in range(0, len(members), 8):
                items = [dataset[i] for i in members[start : start + 8]]
                x, y, controls = (torch.stack(p) for p in zip(*items, strict=True))
                if treble is not None:
                    controls = controls.clone()
                    controls[:, TREBLE] = treble
                model.reset_states()
                pred = model(x, controls)
                totals["esr"] += float(ESR(pred, y)) * len(items)
                totals["mrstft"] += float(MRSTFT(pred, y)) * len(items)
            key = setting if treble in (None, 0.1) else f"{setting}@treble1.0"
            out[key] = {k: v / len(members) for k, v in totals.items()}
    if frozen:
        del processor.pole_shifts
    return out


def band(value: float, low: float, high: float, labels: tuple[str, str, str]) -> str:
    return labels[0] if value >= high else labels[2] if value <= low else labels[1]


def evaluate(splits: list[int]) -> dict | None:
    cells = {(arm, s): run_id(arm, s) for s in splits for arm in ARMS}
    present = {
        k: v for k, v in cells.items() if (bench.RESULTS_DIR / f"{v}.json").exists()
    }
    if splits == [925] and len(present) < 4:
        missing = sorted(set(cells.values()) - set(present.values()))
        print(f"G2.1 incomplet ; il manque {', '.join(missing)}.")
        return None
    if splits != [925]:
        # G2.2: every arm retained at 925 must exist at every requested split.
        g21 = Path(__file__).with_name("poles_verdict_925.json")
        if not g21.exists():
            print("verdict G2.1 absent : lancer d'abord --splits 925.")
            return None
        retained = json.loads(g21.read_text())["g2_2_arms"]
        missing = [
            cells[(a, s)] for a in retained for s in splits if (a, s) not in present
        ]
        if missing:
            print(f"G2.2 incomplet ; il manque {', '.join(missing)}.")
            return None
        present = {k: v for k, v in present.items() if k[0] in retained}
    data = bench.data_module("test", device="marshall")
    data.setup("test")
    results = {}
    for (arm, split), run in sorted(present.items()):
        model = load(run)
        entry = {"run": run, "unseen": unseen(model, data.test_dataset)}
        if arm in POLES:
            entry["unseen_frozen"] = unseen(model, data.test_dataset, frozen=True)
        record = json.loads((bench.RESULTS_DIR / f"{run}.json").read_text())
        entry["seen"] = {k: v["esr"] for k, v in record["val_last_by_setting"].items()}
        entry["polarity_flips"] = record["polarity_flips"]
        results[f"{arm}_{split}"] = entry
        print(f"{run}: evalue", flush=True)
    return results


def log_ratio(a: float, b: float) -> float:
    return math.log(a / b)


def spread(values: list[float]) -> float:
    """Standard deviation across splits; 0 with a single split."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def arm_verdict(arm: str, results: dict, splits: list[int], settings: list) -> dict:
    """d = ln(ESR_A / ESR_arm) per setting, averaged over the splits (G2.2) before
    any median, as predictions_poles.md writes it. With several splits, the paired
    spread of d across splits is reported, and an effect smaller than it is flagged
    "indistinct" next to its label."""
    esr = {s: results[f"{arm}_{s}"]["unseen"] for s in splits}
    ref = {s: results[f"A_{s}"]["unseen"] for s in splits}
    seen = {s: results[f"{arm}_{s}"]["seen"] for s in splits}
    seen_ref = {s: results[f"A_{s}"]["seen"] for s in splits}
    values = [x["esr"] for s in splits for x in esr[s].values()]
    values += [x for s in splits for x in seen[s].values()]
    finite = all(math.isfinite(x) and x > 0 for x in values)
    late_flip = any(
        e > 20 for s in splits for e in results[f"{arm}_{s}"]["polarity_flips"]
    )
    v = {"splits": splits, "gate_run": finite and not late_flip}
    if not finite:
        v["stopped"] = arm in POLES
        return v
    by_split = {
        k: [log_ratio(ref[s][k]["esr"], esr[s][k]["esr"]) for s in splits]
        for k in settings
    }
    per_setting = {k: statistics.mean(x) for k, x in by_split.items()}
    sd_setting = {k: spread(x) for k, x in by_split.items()}
    median_d = statistics.median(per_setting.values())
    mr_by_split = [
        log_ratio(
            statistics.mean(ref[s][k]["mrstft"] for k in settings),
            statistics.mean(esr[s][k]["mrstft"] for k in settings),
        )
        for s in splits
    ]
    mr = statistics.mean(mr_by_split)
    ratio = statistics.mean(
        esr[s][EXTREME]["esr"] / esr[s][INTERPOLATION]["esr"] for s in splits
    )
    # H4 reference: arm A's own ratio at the same splits (3.64 at split 925).
    ratio_a = statistics.mean(
        ref[s][EXTREME]["esr"] / ref[s][INTERPOLATION]["esr"] for s in splits
    )
    seen_by_setting = {
        k: [log_ratio(seen_ref[s][k], seen[s][k]) for s in splits]
        for k in seen_ref[splits[0]]
    }
    seen_d = statistics.median(statistics.mean(x) for x in seen_by_setting.values())
    seen_sd = statistics.median(spread(x) for x in seen_by_setting.values())
    typical_sd = statistics.median(sd_setting.values())
    v |= {
        "d_by_setting": per_setting,
        "sd_by_setting": sd_setting,
        "median_d_unseen": median_d,
        "d_extreme": per_setting[EXTREME],
        "d_mrstft_unseen": mr,
        "ratio_extreme_interpolation": ratio,
        "ratio_extreme_interpolation_A": ratio_a,
        "median_d_seen": seen_d,
    }
    if len(splits) > 1:
        v["indistinct"] = {
            "H1": abs(per_setting[EXTREME]) < sd_setting[EXTREME],
            "H2": abs(median_d) < typical_sd,
            "H3": abs(mr) < spread(mr_by_split),
            "H5": abs(seen_d) < seen_sd,
        }
    if arm not in POLES:
        return v
    three = ("soutenue", "intermediaire", "refutee")
    v["H1"] = band(v["d_extreme"], 0.0, LARGE, three)
    v["H2"] = "non inferieur" if median_d >= -MARGIN else "inferieur"
    v["H3"] = band(mr, 0.0, MARGIN, three)
    v["H4"] = (
        "soutenue"
        if ratio <= 0.6 * ratio_a
        else "refutee"
        if ratio >= ratio_a
        else "intermediaire"
    )
    v["H5"] = "non inferieur" if seen_d >= -MARGIN else "inferieur"
    frozen = statistics.median(
        statistics.mean(
            results[f"{arm}_{s}"]["unseen_frozen"][k]["esr"] / esr[s][k]["esr"]
            for s in splits
        )
        for k in settings
    )
    v["frozen_over_moving_median"] = frozen
    v["H6"] = band(
        math.log(frozen),
        MARGIN,
        LARGE,
        ("se sert des poles", "intermediaire", "ne s'en sert pas"),
    )
    v["stopped"] = median_d <= STOP or not v["gate_run"]
    return v


def structure_reading(
    joint: str, additive: str, results: dict, requested: list[int], settings: list
) -> dict | None:
    splits = [
        s
        for s in requested
        if f"{joint}_{s}" in results and f"{additive}_{s}" in results
    ]
    if not splits:
        return None
    values = [
        results[f"{a}_{s}"]["unseen"][k]["esr"]
        for a in (joint, additive)
        for s in splits
        for k in settings
    ]
    if not all(math.isfinite(x) and x > 0 for x in values):
        return {"median_ln_joint_over_additive": math.nan, "reading": "non evaluable"}
    m = statistics.median(
        statistics.mean(
            log_ratio(
                results[f"{joint}_{s}"]["unseen"][k]["esr"],
                results[f"{additive}_{s}"]["unseen"][k]["esr"],
            )
            for s in splits
        )
        for k in settings
    )
    labels = ("additif meilleur", "indifferent", "joint meilleur")
    return {
        "median_ln_joint_over_additive": m,
        "reading": band(m, -MARGIN, MARGIN, labels),
    }


def g22_arms(verdicts: dict, structure: dict) -> list[str]:
    """Arms of G2.2, by the rule of predictions_poles.md."""
    alive = [a for a in POLES if a in verdicts and not verdicts[a]["stopped"]]
    best = sorted(alive, key=lambda a: -verdicts[a]["median_d_unseen"])
    chosen = best[:1]
    if len(best) == 2:
        gap = (
            verdicts[best[0]]["median_d_unseen"] - verdicts[best[1]]["median_d_unseen"]
        )
        if abs(gap) < MARGIN:
            chosen = best
    arms = ["A", *chosen]
    if structure.get("film", {}).get("reading") == "additif meilleur":
        arms.append("B")
    return arms


def report(verdicts: dict, structure: dict) -> None:
    for arm, v in verdicts.items():
        print(f"\nbras {arm} ({', '.join(map(str, v['splits']))})")
        for key in (
            "median_d_unseen",
            "d_extreme",
            "d_mrstft_unseen",
            "ratio_extreme_interpolation",
            "median_d_seen",
        ):
            if key in v:
                print(f"  {key:28s} {v[key]:+.3f}")
        if "indistinct" in v:
            print(f"  {'indistinct':28s} {v['indistinct']}")
        for key in ("H1", "H2", "H3", "H4", "H5", "H6", "frozen_over_moving_median"):
            if key in v:
                print(f"  {key:28s} {v[key]}")
        print(f"  {'gate_run':28s} {v['gate_run']}")
        if "stopped" in v:
            print(f"  {'stopped':28s} {v['stopped']}")
    for family, s in structure.items():
        m = s["median_ln_joint_over_additive"]
        print(f"\nstructure {family} : {m:+.3f} -> {s['reading']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, nargs="+", default=[925])
    args = parser.parse_args()
    torch.set_num_threads(16)
    results = evaluate(args.splits)
    if results is None:
        return 1
    settings = sorted(
        k for k in results[f"A_{args.splits[0]}"]["unseen"] if "@" not in k
    )
    verdicts = {}
    for arm in ("B", "C", "D"):
        splits = [
            s for s in args.splits if f"{arm}_{s}" in results and f"A_{s}" in results
        ]
        if splits:
            verdicts[arm] = arm_verdict(arm, results, splits, settings)
    structure = {}
    for family, (joint, additive) in {"film": ("A", "B"), "poles": ("C", "D")}.items():
        reading = structure_reading(joint, additive, results, args.splits, settings)
        if reading is not None:
            structure[family] = reading
    report(verdicts, structure)
    out = {"results": results, "verdicts": verdicts, "structure": structure}
    if args.splits == [925]:
        out["g2_2_arms"] = g22_arms(verdicts, structure)
        print(f"\nG2.2 (partitions 736 et 688) : bras {', '.join(out['g2_2_arms'])}")
    name = "poles_verdict_" + "_".join(map(str, args.splits)) + ".json"
    (Path(__file__).parent / name).write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
