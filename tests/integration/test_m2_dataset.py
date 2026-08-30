import json
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from fssr_nam.data.excitations import generate_excitation
from fssr_nam.data.systems import apply_system

ROOT = Path(__file__).resolve().parents[2]


def test_m2_manifest_has_disjoint_complete_source_files():
    manifest = json.loads(
        (ROOT / "datasets/manifests/m2_synthetic_tanh.json").read_text()
    )
    splits = json.loads((ROOT / "datasets/splits/m2_synthetic_tanh.json").read_text())
    train, validation = manifest["files"]
    assert train["source_group"] != validation["source_group"]
    assert splits["path_overlap"] == []
    assert splits["leakage_check"] == "passed"


def test_m2_audio_is_exact_configured_float32_target(tmp_path: Path):
    config = yaml.safe_load(
        (ROOT / "configs/data/m2_synthetic_tanh.yaml").read_text(encoding="utf-8")
    )
    sample_rate = int(config["sample_rate"])
    for split_name in ("train", "validation"):
        split = config[split_name]
        x = np.concatenate(
            [
                generate_excitation(
                    excitation,
                    sample_rate=sample_rate,
                    duration_seconds=float(split["seconds_per_excitation"]),
                    seed=int(split["seed"]) + index,
                )
                for index, excitation in enumerate(split["excitations"])
            ]
        ).astype(np.float32, copy=False)
        y = apply_system("tanh", x, sample_rate)
        input_path = tmp_path / f"{split_name}_input.wav"
        output_path = tmp_path / f"{split_name}_output.wav"
        sf.write(input_path, x, sample_rate, subtype="FLOAT")
        sf.write(output_path, y, sample_rate, subtype="FLOAT")
        read_x, x_rate = sf.read(input_path, dtype="float32")
        read_y, y_rate = sf.read(output_path, dtype="float32")
        assert x_rate == y_rate == 48_000
        assert read_x.ndim == read_y.ndim == 1
        np.testing.assert_array_equal(read_x, x)
        np.testing.assert_array_equal(read_y, apply_system("tanh", read_x, x_rate))
