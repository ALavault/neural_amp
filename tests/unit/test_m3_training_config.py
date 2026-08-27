from pathlib import Path

import numpy as np
import torch
import yaml

from fssr_nam.training.m3 import delay_target, model_factory, short_nonlinear_memory

ROOT = Path(__file__).resolve().parents[2]


def test_all_preregistered_m3_models_construct_and_are_finite():
    config = yaml.safe_load((ROOT / "configs/training/m3_synthetic.yaml").read_text())
    signal = torch.linspace(-0.5, 0.5, 257)
    for variant in ("S0", "S1", "S2", "S3", "S4"):
        model = model_factory(variant, config["model"])
        assert torch.isfinite(model(signal)).all()


def test_short_memory_target_is_causal_and_delay_target_preserves_length():
    signal = np.linspace(-0.5, 0.5, 128, dtype=np.float32)
    changed = signal.copy()
    changed[80:] *= -1
    target = short_nonlinear_memory(signal)
    changed_target = short_nonlinear_memory(changed)
    np.testing.assert_array_equal(target[:80], changed_target[:80])
    tensor = torch.from_numpy(target)
    delayed = delay_target(tensor, 16)
    assert delayed.shape == tensor.shape
    torch.testing.assert_close(delayed[:16], torch.zeros(16))
