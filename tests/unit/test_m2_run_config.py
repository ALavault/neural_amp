import json
from pathlib import Path

import yaml

from fssr_nam.training.nam_a2 import make_configs

ROOT = Path(__file__).resolve().parents[2]


def test_m2_training_uses_disjoint_files_and_official_a2_config():
    campaign = yaml.safe_load((ROOT / "configs/training/m2_a2_smoke.yaml").read_text())
    manifest = json.loads(
        (ROOT / "datasets/manifests/m2_synthetic_tanh.json").read_text()
    )
    data, model, learning = make_configs(
        campaign,
        manifest,
        root=ROOT,
        model_config_path=(
            ROOT
            / "third_party/neural-amp-modeler/nam/train/_resources"
            / "config_model_packed.json"
        ),
    )
    assert data["train"]["x_path"] != data["validation"]["x_path"]
    assert model["net"]["name"] == "PackedWaveNet"
    assert [entry["name"] for entry in model["net"]["config"]["submodels"]] == [
        "channels_3",
        "channels_8",
    ]
    assert learning["trainer"]["precision"] == "32-true"
    assert learning["trainer"]["deterministic"] == "warn"
    assert campaign["deterministic_mode"] == "warn_only"
