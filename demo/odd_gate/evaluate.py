#!/usr/bin/env python3
"""VS100R and KIPR readings of the two KIPR-inspired tracks, against arm C.

--arm odd: demo/odd_gate/predictions.md; --arm grey: demo/greybox/predictions.md.
Read-only, CPU. Needs, after the arm's last run:
- diagnosis/hammerstein/vs100r_hammerstein.json regenerated with the arm (its RUNS);
- demo/kipr_known_answer/known_answer_<odd|grey>.json and known_answer_r2.json
  (known_answer.py measure --tag ...).
VS100R ESR: P of demo/train_length/eval_train_length.py (every non-low-signal take,
di_test, mean of ln ESR per gain-contour pair, the unseen pair excluded, last
checkpoint), d_i = s(arm) - s(C) per seed. 2nd harmonic: signed D2 gap to the device
(dB), median over bands 200 Hz - 8 kHz per (take, run), then over the 8 test takes and
3 seeds, per sweep level; delta = arm - C. Writes evaluation.json in demo/odd_gate
or demo/greybox.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "demo/train_length"))
import eval_train_length as etl  # noqa: E402

RUNS = {
    "C": "vs100r48fast_film_s{}",
    "odd": "vs100r48odd_film_s{}",
    "grey": "vs100r48grey_s{}",
}
HAMMERSTEIN_ARM = {"C": "C", "odd": "ODD", "grey": "GREY"}
OUT = {"odd": ROOT / "demo/odd_gate", "grey": ROOT / "demo/greybox"}
LEVELS = ("-20", "-10", "-3")


def esr_reading(arm: str) -> dict:
    takes = [t for t in etl.ev.takes() if not t["low_signal"]]
    data = {t["order"]: etl.ev.audio(t) for t in takes}
    s, by_take, parameters = {}, {}, {}
    for name in ("C", arm):
        for seed in etl.SEEDS:
            run_id = RUNS[name].format(seed)
            scored = etl.score(etl.load(run_id, "last"), takes, data)
            s[run_id] = scored["s_pairs"]
            by_take[run_id] = scored["ln_esr_by_take"]
            record = json.loads((etl.bench.RESULTS_DIR / f"{run_id}.json").read_text())
            parameters[name] = record["parameters"]
            print(f"{run_id}: s {s[run_id]:.3f}", flush=True)
    d_i = {
        seed: s[RUNS[arm].format(seed)] - s[RUNS["C"].format(seed)]
        for seed in etl.SEEDS
    }
    values = list(d_i.values())
    # Descriptive: gain 2.5 test takes, ln ESR arm - C per seed.
    gain25 = {
        take: {
            seed: by_take[RUNS[arm].format(seed)][take]
            - by_take[RUNS["C"].format(seed)][take]
            for seed in etl.SEEDS
        }
        for take in (16, 36)
    }
    return {
        "s": s,
        "d_i": d_i,
        "d": statistics.mean(values),
        "esr_ratio": math.exp(statistics.mean(values)),
        "verdict": etl.verdict(values),
        "acceptable": etl.acceptable(values),
        "gain25_ln_esr_minus_c": gain25,
        "parameters": parameters,
    }


def harmonic_reading(arm: str) -> dict:
    data = json.loads(
        (ROOT / "diagnosis/hammerstein/vs100r_hammerstein.json").read_text()
    )
    centres = np.array(data["centres"])
    band = (centres >= 200) & (centres <= 8000)

    def gaps(name: str, order: int, level: str) -> list[float]:
        label = HAMMERSTEIN_ARM[name]
        out = []
        for take in data["takes"].values():
            for model in take[level]["models"].values():
                if model["arm"] == label:
                    err = np.array(model["d_errors"]["level_db"])[order, band]
                    out.append(float(np.nanmedian(err)))
        return out

    reading = {}
    for level in LEVELS:
        c2, a2 = gaps("C", 1, level), gaps(arm, 1, level)
        if len(a2) != 24 or len(c2) != 24:
            raise SystemExit(f"{level}: {len(a2)} arm and {len(c2)} C cases, not 24")
        c3, a3 = gaps("C", 2, level), gaps(arm, 2, level)
        reading[level] = {
            "d2_c": statistics.median(c2),
            "d2_arm": statistics.median(a2),
            "delta_d2": statistics.median(a2) - statistics.median(c2),
            "abs_d3_c": statistics.median(abs(v) for v in c3),
            "abs_d3_arm": statistics.median(abs(v) for v in a3),
        }
    # Descriptive: gain 2.5 test takes, signed D2 gap at -10 dBFS, median over seeds.
    for take in ("16", "36"):
        models = data["takes"][take]["-10"]["models"].values()
        reading[f"take{take}_-10"] = {
            name: statistics.median(
                float(np.nanmedian(np.array(m["d_errors"]["level_db"])[1, band]))
                for m in models
                if m["arm"] == HAMMERSTEIN_ARM[name]
            )
            for name in ("C", arm)
        }
    deltas = [reading[lv]["delta_d2"] for lv in LEVELS]
    if all(v <= -6 for v in deltas):
        label = "réduit"
    elif all(abs(v) <= 3 for v in deltas):
        label = "inchangé"
    else:
        label = "partiel"
    return {"by_level": reading, "verdict": label}


def kipr_reading(arm: str) -> dict:
    here = ROOT / "demo/kipr_known_answer"
    tagged = json.loads((here / f"known_answer_{arm}.json").read_text())["readings"]
    r2 = json.loads((here / "known_answer_r2.json").read_text())["readings"]
    out = {"arm": tagged, "r2": r2}
    # Only a complete, valid reading (instrument, alignment, control) is compared.
    valid = {"plancher", "pas de plancher", "plancher partiel"}
    if tagged.get("verdict") not in valid:
        out["comparison"] = "aucune lecture (bras)"
    elif r2.get("verdict") not in valid:
        out["comparison"] = "descriptif (r2 sans lecture valide)"
    else:
        gap = tagged["sym"]["e_primary"] - r2["sym"]["e_primary"]
        out["e_sym_minus_r2"] = gap
        out["comparison"] = "réduit" if gap <= -10 else "non réduit"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["odd", "grey"])
    arm = parser.parse_args().arm
    torch.set_num_threads(16)
    result = {
        "esr": esr_reading(arm),
        "harmonics": harmonic_reading(arm),
        "kipr": kipr_reading(arm),
    }
    esr_ok = result["esr"]["acceptable"]
    reduced = result["harmonics"]["verdict"] == "réduit"
    if arm == "odd":
        if not esr_ok:
            verdict = "porte impaire rejetée (ESR)"
        elif reduced:
            verdict = "porte impaire retenue"
        else:
            verdict = "porte impaire non retenue (harmoniques paires non réduites)"
    else:
        verdict = (
            "boîte grise équivalente au bras C (ESR)"
            if esr_ok
            else "boîte grise moins bonne que le bras C (ESR)"
        )
    result["verdict"] = verdict
    print(json.dumps(result, indent=1, ensure_ascii=False, default=str))
    (OUT[arm] / "evaluation.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n"
    )


if __name__ == "__main__":
    main()
