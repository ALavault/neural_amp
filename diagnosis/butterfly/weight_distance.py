#!/usr/bin/env python3
"""Where do the children move, if not further? (Read-only, no inference.)

mode_connectivity.py found that the two arms of fork 100 differ by a factor 17 in the
spread of their test ESR and by almost nothing in the total distance travelled in weight
space. This splits that distance by family of parameter: the state-space time constants
and poles of the DSSM blocks, and the ordinary convolution and linear weights. If the
arms separate anywhere, it is by where they move, not by how far.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
from mode_connectivity import CHILDREN, CONTROL, weights  # noqa: E402


def family(name: str) -> str:
    """The 94 tensors end in one of: log_dt, log_A_real, A_imag, C, D, weight, bias."""
    return name.rsplit(".", 1)[-1]


def main() -> None:
    control = weights(CONTROL)
    families = sorted({family(k) for k in control})
    counts = {
        f: sum(control[k].numel() for k in control if family(k) == f) for f in families
    }
    print("parameters per family:", counts)

    out = {"control": CONTROL, "parameters": counts, "children": {}}
    for child in CHILDREN:
        theta = weights(child)
        # Distance per family, and the relative move against the control's own norm.
        rows = {}
        for f in families:
            keys = [k for k in control if family(k) == f]
            delta = torch.sqrt(sum((theta[k] - control[k]).pow(2).sum() for k in keys))
            norm = torch.sqrt(sum(control[k].pow(2).sum() for k in keys))
            rows[f] = {"l2": float(delta), "relative": float(delta / norm)}
        out["children"][child] = rows
        print(
            f"{child}: " + "  ".join(f"{f} {rows[f]['relative']:.3f}" for f in families)
        )

    (Path(__file__).parent / "weight_distance.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
