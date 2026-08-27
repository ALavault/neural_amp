"""Generate and persist the M1 synthetic corpus manifest and filter response."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import yaml
from scipy.signal import freqz

from fssr_nam.data.corpus import build_corpus_manifest
from fssr_nam.dsp.multirate import DEFAULT_DECIMATION_CONFIG, design_decimation_filter
from fssr_nam.reporting.provenance import git_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    config_path = Path("configs/data/synthetic.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    external_config = config["external_di_diagnostic"]
    external_path = Path(external_config["path"])
    if _sha256(external_path) != external_config["file_sha256"]:
        raise RuntimeError("external DI diagnostic SHA-256 does not match config")
    external_di, external_rate = sf.read(
        external_path, dtype="float32", always_2d=False
    )
    if external_rate != 48_000 or external_di.ndim != 1:
        raise RuntimeError("external DI diagnostic must be mono at 48 kHz")
    manifest = build_corpus_manifest(config, external_di_48k=external_di)
    manifest["external_di_diagnostic"].update(
        {
            "source": external_config["source"],
            "license": external_config["license"],
            "tier": external_config["tier"],
            "path": str(external_path),
            "file_sha256": external_config["file_sha256"],
        }
    )
    manifest["git"] = git_state(
        ignored_generated_paths=(
            "experiments/summaries/m1_metric_validation",
            "experiments/summaries/m1_synthetic",
        )
    )
    manifest["decimation"] = {
        "stages": [2, 2],
        "factor_per_stage": DEFAULT_DECIMATION_CONFIG.factor,
        "taps_per_stage": DEFAULT_DECIMATION_CONFIG.taps,
        "cutoff_fraction_of_target_nyquist": (
            DEFAULT_DECIMATION_CONFIG.cutoff_fraction_of_target_nyquist
        ),
        "kaiser_beta": DEFAULT_DECIMATION_CONFIG.kaiser_beta,
        "group_delay_input_samples_per_stage": (DEFAULT_DECIMATION_CONFIG.taps - 1)
        // 2,
        "boundary_definition": "zero extension; known linear-phase delay removed",
    }

    output_dir = Path("experiments/summaries/m1_synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "config-resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )

    kernel = design_decimation_filter(DEFAULT_DECIMATION_CONFIG)
    normalized_frequency, response = freqz(kernel, worN=16_384)
    input_nyquist_fraction = normalized_frequency / np.pi
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), 1.0e-10))
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(input_nyquist_fraction, magnitude_db)
    axis.axvline(0.5, color="black", linestyle="--", label="target Nyquist")
    axis.axvline(0.45, color="tab:orange", linestyle=":", label="declared cutoff")
    axis.set(xlabel="Input Nyquist fraction", ylabel="Magnitude (dB)", ylim=(-140, 2))
    axis.grid(True, alpha=0.25)
    axis.legend()
    axis.set_title("M1 x2 reference-decimation FIR")
    figure.tight_layout()
    figure.savefig(output_dir / "decimation_response.png", dpi=160)
    plt.close(figure)
    print(
        json.dumps(
            {
                "excitations": manifest["excitation_count"],
                "systems": manifest["system_count"],
                "pairs": manifest["pair_count"],
                "status": "passed",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
