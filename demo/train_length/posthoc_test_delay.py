#!/usr/bin/env python3
"""Post hoc check of the co-reading, outside the rule of predictions.md. Read-only.

The co-reading estimates tau on di_validation, mostly trained audio for C and mostly
unseen audio for L10 and L20. Here tau is estimated on di_test itself (one common delay
per run, never per take; unseen by every arm), then the co-reading aggregates are
recomputed. Reported apart; changes no label. GPU as in coreading_delay.py.
Writes eval_train_length_posthoc_test_tau.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coreading_delay as cr  # noqa: E402

tl = cr.tl


@torch.no_grad()
def main() -> None:
    p_result = json.loads((HERE / "eval_train_length.json").read_text())
    takes = [t for t in cr.ev.takes() if not t["low_signal"]]
    data = {t["order"]: cr.ev.audio(t) for t in takes}
    test48 = {k: tuple(resample_poly(s, 1, 4) for s in v) for k, v in data.items()}
    result = {"rule": "post hoc, hors règle : tau commun par run estimé sur di_test"}
    runs = {}
    for pattern in tl.ARMS.values():
        for seed in tl.SEEDS:
            run_id = pattern.format(seed)
            model = tl.load(run_id, "last").to(cr.DEVICE)
            tau, _ = cr.common_delay(model, takes, test48)
            edge = tau in (cr.DELAYS[0], cr.DELAYS[-1])
            run = {"tau": tau, "grid_edge": edge} | cr.score(model, takes, data, tau)
            runs[run_id] = run
            print(f"{run_id}: tau {tau:+.2f} s {run['s_pairs']:.3f}", flush=True)
    result["runs"] = runs
    result["readings"] = cr.readings(runs, p_result["readings"])
    print(json.dumps(result["readings"], indent=1, ensure_ascii=False))
    (HERE / "eval_train_length_posthoc_test_tau.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
