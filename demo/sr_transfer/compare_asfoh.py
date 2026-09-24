#!/usr/bin/env python3
"""FIX_foh.md section 1 (a): the --as-foh arms against the recorded two-coefficient arms,
tolerance 1e-3 relative, and the 48 kHz identity (ESR <= 1e-12). Read-only, records only."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
PAIRS = {
    "44k1_delta_interior": "44k1_delta_2tap_interior",
    "88k_delta_band48_vs_48k_interior": "88k_delta_2tap_band48_vs_48k_interior",
    "96k_delta_band48_vs_48k_interior": "96k_delta_2tap_band48_vs_48k_interior",
    "192k_delta_band48_vs_48k_interior": "192k_delta_2tap_band48_vs_48k_interior",
}
TOLERANCE = 1e-3
ok = True
for seed in (925, 736, 688):
    stem = f"sr_transfer_paired_ssm_wavenet_s{seed}_last"
    path = HERE / f"{stem}_asfoh.json"
    if not path.exists():
        print(f"s{seed}: pas encore de relance")
        continue
    foh = json.loads(path.read_text())["arms"]
    two = json.loads((HERE / f"{stem}_twotap.json").read_text())["arms"]
    identity = foh["48k_asfoh_identity_esr"]["esr"]
    ok = ok and identity <= 1e-12
    print(f"s{seed}: identite 48 kHz, ESR {identity:.1e}")
    for mine, theirs in PAIRS.items():
        a, b = foh[mine]["esr"], two[theirs]["esr"]
        rel = abs(a - b) / b
        ok = ok and rel <= TOLERANCE
        print(f"  {mine:36s} FOH {a:.7f}  2 coef. {b:.7f}  ecart relatif {rel:.1e}")
print(f"\ntolerances tenues : {ok}")
