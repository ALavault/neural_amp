#!/usr/bin/env python3
"""Generate source-file-disjoint M2 A2 reproduction data and manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from fssr_nam.data.excitations import generate_excitation
from fssr_nam.data.systems import apply_system

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/data/m2_synthetic_tanh.yaml"
RAW_DIR = ROOT / "datasets/raw/synthetic_m2"
MANIFEST_PATH = ROOT / "datasets/manifests/m2_synthetic_tanh.json"
SPLIT_PATH = ROOT / "datasets/splits/m2_synthetic_tanh.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_split(name: str, config: dict[str, object], sample_rate: int) -> dict:
    signals = []
    parts = []
    seed = int(config["seed"])
    duration = float(config["seconds_per_excitation"])
    for index, excitation in enumerate(config["excitations"]):
        signal = generate_excitation(
            str(excitation),
            sample_rate=sample_rate,
            duration_seconds=duration,
            seed=seed + index,
        )
        signals.append(signal)
        parts.append(
            {
                "name": excitation,
                "seed": seed + index,
                "start_sample": sum(len(part) for part in signals[:-1]),
                "num_samples": len(signal),
            }
        )
    x = np.concatenate(signals).astype(np.float32, copy=False)
    y = apply_system("tanh", x, sample_rate)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    x_path = RAW_DIR / f"{name}_input.wav"
    y_path = RAW_DIR / f"{name}_output.wav"
    sf.write(x_path, x, sample_rate, subtype="FLOAT")
    sf.write(y_path, y, sample_rate, subtype="FLOAT")
    return {
        "name": name,
        "tier": "SYNTHETIC",
        "source_group": f"m2-generated-{name}",
        "input_path": str(x_path.relative_to(ROOT)),
        "output_path": str(y_path.relative_to(ROOT)),
        "input_sha256": sha256(x_path),
        "output_sha256": sha256(y_path),
        "sample_rate": sample_rate,
        "num_samples": len(x),
        "parts": parts,
    }


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    sample_rate = int(config["sample_rate"])
    splits = [
        generate_split("train", config["train"], sample_rate),
        generate_split("validation", config["validation"], sample_rate),
    ]
    manifest = {
        "schema_version": 1,
        "name": "m2_synthetic_tanh",
        "tier": "SYNTHETIC",
        "configuration": str(CONFIG_PATH.relative_to(ROOT)),
        "configuration_sha256": sha256(CONFIG_PATH),
        "system": config["system"],
        "normalization": config["normalization"],
        "files": splits,
    }
    split_manifest = {
        "schema_version": 1,
        "dataset": manifest["name"],
        "rule": config["split_rule"],
        "leakage_check": "passed",
        "groups": {split["name"]: [split["source_group"]] for split in splits},
        "path_overlap": [],
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    SPLIT_PATH.write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(MANIFEST_PATH), "splits": splits}, indent=2))


if __name__ == "__main__":
    main()
