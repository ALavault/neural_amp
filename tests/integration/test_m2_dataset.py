import json
from pathlib import Path

import numpy as np
import soundfile as sf

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


def test_m2_audio_is_exact_configured_float32_target():
    manifest = json.loads(
        (ROOT / "datasets/manifests/m2_synthetic_tanh.json").read_text()
    )
    for file_manifest in manifest["files"]:
        x, x_rate = sf.read(ROOT / file_manifest["input_path"], dtype="float32")
        y, y_rate = sf.read(ROOT / file_manifest["output_path"], dtype="float32")
        assert x_rate == y_rate == 48_000
        assert x.ndim == y.ndim == 1
        np.testing.assert_array_equal(y, apply_system("tanh", x, x_rate))
