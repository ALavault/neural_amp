import json
from pathlib import Path

import numpy as np
import torch
import yaml

from fssr_nam.training.m4 import (
    causal_predict,
    delay_target,
    model_factory,
    training_prediction,
)

ROOT = Path(__file__).resolve().parents[2]


def test_m4_matrix_and_equal_sample_budget() -> None:
    config = yaml.safe_load((ROOT / "configs/training/m4_smoke.yaml").read_text())
    assert len(config["devices"]) * len(config["models"]) * len(config["seeds"]) == 24
    assert config["context_samples"] == 6346
    samples_seen = (
        config["optimizer_steps"] * config["batch_size"] * config["output_samples"]
    )
    assert samples_seen == 3_276_800
    manifest = json.loads((ROOT / "datasets/manifests/m4_internal.json").read_text())
    assert {item["device"] for item in manifest["files"]} == set(config["devices"])


def test_m4_recovery_matches_a2_parameter_budget() -> None:
    recovery = yaml.safe_load((ROOT / "configs/training/m4_recovery.yaml").read_text())
    model_config = yaml.safe_load(
        (ROOT / "configs/training/m3_synthetic.yaml").read_text()
    )["model"]
    model_config["residual_channels"] = recovery["residual_channels"]
    model = model_factory("S3", root=ROOT, model_config=model_config)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters == recovery["parameters"] == 12_192
    assert abs(parameters - 12_145) / 12_145 < 0.005


def test_declared_target_delay() -> None:
    target = np.arange(6, dtype=np.float32)
    assert np.array_equal(delay_target(target, 2), [0, 0, 0, 1, 2, 3])


def test_all_m4_models_produce_expected_training_shape() -> None:
    model_config = yaml.safe_load(
        (ROOT / "configs/training/m3_synthetic.yaml").read_text()
    )["model"]
    windows = torch.randn(1, 6346 + 64)
    for code in ("B0", "B2", "S3", "S4"):
        model = model_factory(code, root=ROOT, model_config=model_config).eval()
        with torch.inference_mode():
            output = training_prediction(model, code, windows, 64)
        assert output.shape == (1, 64)
        assert torch.isfinite(output).all()


def test_b0_overlap_blocks_match_official_complete_inference() -> None:
    model_config = yaml.safe_load(
        (ROOT / "configs/training/m3_synthetic.yaml").read_text()
    )["model"]
    model = model_factory("B0", root=ROOT, model_config=model_config).eval()
    signal = np.random.default_rng(7).normal(size=10_123).astype(np.float32)
    with torch.inference_mode():
        complete = model(torch.from_numpy(signal), pad_start=True).numpy()
    blocked = causal_predict(
        model,
        "B0",
        signal,
        device=torch.device("cpu"),
        context_samples=6346,
        block_samples=1000,
    )
    assert np.max(np.abs(complete - blocked)) < 2.0e-6


def test_m4_memory_diagnostic_keeps_identity_init_and_block_parity() -> None:
    memory = yaml.safe_load((ROOT / "configs/training/m4_memory.yaml").read_text())
    smoke = yaml.safe_load((ROOT / "configs/training/m4_smoke.yaml").read_text())
    assert memory["devices"] == smoke["devices"]
    assert memory["seeds"] == smoke["seeds"]
    assert memory["cascade_memory_samples"] == 2 * memory["taps"] - 1 >= 63
    model_config = yaml.safe_load(
        (ROOT / "configs/training/m3_synthetic.yaml").read_text()
    )["model"]
    model_config["taps"] = memory["taps"]
    model = model_factory("S3", root=ROOT, model_config=model_config).eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters == memory["parameters"] == 1_276
    signal = torch.rand(20_000, generator=torch.Generator().manual_seed(3)) - 0.5
    with torch.inference_mode():
        output = model(signal)
    torch.testing.assert_close(output, signal, atol=3.0e-6, rtol=3.0e-6)
    blocked = causal_predict(
        model,
        "S3",
        signal.numpy(),
        device=torch.device("cpu"),
        context_samples=smoke["context_samples"],
        block_samples=4093,
    )
    assert np.max(np.abs(blocked - output.numpy())) < 2.0e-6


def test_m4_grid_diagnostic_keeps_identity_init_and_block_parity() -> None:
    grid = yaml.safe_load((ROOT / "configs/training/m4_grid.yaml").read_text())
    memory = yaml.safe_load((ROOT / "configs/training/m4_memory.yaml").read_text())
    smoke = yaml.safe_load((ROOT / "configs/training/m4_smoke.yaml").read_text())
    assert grid["models"] == ["S3"]
    assert grid["devices"] == smoke["devices"]
    assert grid["seeds"] == smoke["seeds"]
    assert grid["taps"] == memory["taps"] == 33
    assert grid["spline_range"] == 0.4
    assert grid["knot_spacing"] == 2 * grid["spline_range"] / (grid["num_knots"] - 1)
    model_config = yaml.safe_load(
        (ROOT / "configs/training/m3_synthetic.yaml").read_text()
    )["model"]
    default = model_factory("S3", root=ROOT, model_config=dict(model_config))
    assert default.core.shaper.knots[0].item() == -2.0
    model_config["taps"] = grid["taps"]
    model_config["spline_range"] = grid["spline_range"]
    model = model_factory("S3", root=ROOT, model_config=model_config).eval()
    knots = model.core.shaper.knots
    assert len(knots) == grid["num_knots"]
    torch.testing.assert_close(knots[0].item(), -0.4)
    torch.testing.assert_close(knots[-1].item(), 0.4)
    torch.testing.assert_close(model.core.shaper.spacing, 0.05)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters == grid["parameters"] == memory["parameters"] == 1_276
    signal = torch.rand(20_000, generator=torch.Generator().manual_seed(3)) - 0.5
    assert signal.abs().max() > grid["spline_range"]
    with torch.inference_mode():
        output = model(signal)
    torch.testing.assert_close(output, signal, atol=3.0e-6, rtol=3.0e-6)
    blocked = causal_predict(
        model,
        "S3",
        signal.numpy(),
        device=torch.device("cpu"),
        context_samples=smoke["context_samples"],
        block_samples=4093,
    )
    assert np.max(np.abs(blocked - output.numpy())) < 2.0e-6
