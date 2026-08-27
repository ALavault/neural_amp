"""Persist the M1 controlled metric-perturbation audit."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fssr_nam.metrics.validation import validate_metric_suite

DISPLAY_METRICS = (
    "esr",
    "magnitude_error",
    "phase_error_radians",
    "complex_harmonic_error",
    "inharmonic_energy_ratio",
    "envelope_error",
)


def _git_state() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return {"commit": commit, "dirty": dirty}


def main() -> None:
    validation = validate_metric_suite()
    validation["git"] = _git_state()
    if not validation["all_checks_passed"]:
        failed = [name for name, passed in validation["checks"].items() if not passed]
        raise RuntimeError(f"metric checks failed: {', '.join(failed)}")

    output_dir = Path("experiments/summaries/m1_metric_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    names = list(validation["results"])
    matrix = np.array(
        [
            [float(validation["results"][name][metric]) for metric in DISPLAY_METRICS]
            for name in names
        ]
    )
    matrix = np.log10(np.maximum(np.abs(matrix), 1.0e-12))
    figure, axis = plt.subplots(figsize=(9, 6))
    image = axis.imshow(matrix, aspect="auto", cmap="magma")
    axis.set_xticks(
        range(len(DISPLAY_METRICS)), DISPLAY_METRICS, rotation=35, ha="right"
    )
    axis.set_yticks(range(len(names)), names)
    axis.set_title("M1 metric responses (log10 absolute value)")
    figure.colorbar(image, ax=axis, label="log10 response")
    figure.tight_layout()
    figure.savefig(output_dir / "detection_matrix.png", dpi=160)
    plt.close(figure)
    print(json.dumps(validation["checks"], sort_keys=True))


if __name__ == "__main__":
    main()
