from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from fssr_nam.inference import (
    R1NativeReference,
    export_r1_native_model,
    load_r1_native_payload,
    verify_r1_cpp_parity,
)
from fssr_nam.models.r1 import R1Cascade, R1Mono


@pytest.fixture(scope="session")
def r1_cpp_tools(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    if shutil.which("cmake") is None or shutil.which("c++") is None:
        pytest.skip("CMake and a C++ compiler are required for native parity")
    build = tmp_path_factory.mktemp("r1-cpp-build")
    repository = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            "cmake",
            "-S",
            str(repository / "cpp"),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            "r1_block_runner",
            "r1_benchmark",
            "-j2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "runner": build / "r1_block_runner",
        "benchmark": build / "r1_benchmark",
    }


def _make_nontrivial(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for parameter_index, parameter in enumerate(model.parameters()):
            positions = torch.arange(parameter.numel(), dtype=parameter.dtype)
            values = 0.04 * torch.sin(0.37 * positions + 0.29 * parameter_index)
            parameter.copy_(values.reshape_as(parameter))


@pytest.mark.parametrize(
    ("model_type", "receptive_field", "expected_kind"),
    [
        (R1Mono, 31, "causal-full-convolution-tcn"),
        (R1Mono, 2047, "causal-depthwise-separable-tcn"),
        (R1Cascade, 2047, "causal-depthwise-separable-tcn"),
    ],
)
def test_native_export_matches_python_for_blocks_reset_and_slow_controller(
    tmp_path: Path,
    r1_cpp_tools: dict[str, Path],
    model_type: type[R1Mono] | type[R1Cascade],
    receptive_field: int,
    expected_kind: str,
) -> None:
    torch.manual_seed(20260827)
    model = model_type(
        taps=5,
        num_knots=7,
        receptive_field=receptive_field,
        slow_hidden_size=3,
        slow_decimation=16,
    )
    _make_nontrivial(model)
    signal = torch.linspace(-0.7, 0.8, 257) + 0.05 * torch.sin(torch.arange(257) * 0.23)
    model.reset_state()
    with torch.no_grad():
        expected = model(signal).numpy()
    model_path = export_r1_native_model(model, tmp_path / "model.json")
    payload = load_r1_native_payload(model_path)
    assert payload["residual"]["kind"] == expected_kind
    assert payload["slow_controller"]["kind"] == "fssr-slow-gru-v1"
    reference = R1NativeReference(payload).process(signal.numpy())
    np.testing.assert_allclose(reference, expected, atol=2.0e-5, rtol=2.0e-5)
    report_path = tmp_path / "parity.json"
    report = verify_r1_cpp_parity(
        model_path,
        r1_cpp_tools["runner"],
        signal.numpy(),
        report_path=report_path,
    )
    assert report["python_cpp_parity"] is True
    assert report["passed"] is True
    assert report["max_abs_error"] <= 2.0e-5
    assert report["reset_verified"] is True
    assert report["covered_compositions"] == [
        {
            "core": "cascade" if model_type is R1Cascade else "mono",
            "residual": expected_kind,
            "receptive_field": receptive_field,
            "precision": "float32",
            "slow_controller": "fssr-slow-gru-v1",
        }
    ]
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_native_benchmark_reports_block_64_median_and_p95(
    tmp_path: Path, r1_cpp_tools: dict[str, Path]
) -> None:
    model = R1Mono(taps=3, num_knots=5, receptive_field=31)
    model_path = export_r1_native_model(model, tmp_path / "benchmark-model.json")
    completed = subprocess.run(
        [str(r1_cpp_tools["benchmark"]), str(model_path), "1", "1"],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["format"] == "fssr-r1-benchmark-v1"
    assert result["block_size"] == 64
    assert result["median_block_ns"] > 0.0
    assert result["p95_block_ns"] >= result["median_block_ns"]
    assert result["latency_samples"] == 0
    assert result["state_size_bytes"] > 0


def test_blind_lstm_cost_benchmark_uses_registered_width_without_training_data(
    r1_cpp_tools: dict[str, Path],
) -> None:
    completed = subprocess.run(
        [str(r1_cpp_tools["benchmark"]), "--lstm-width", "16", "1", "1"],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["format"] == "fssr-r1-lstm-cost-benchmark-v1"
    assert result["blind"] is True
    assert result["selection_uses_audio_or_esr"] is False
    assert result["trained_weights"] is False
    assert result["width"] == 16
    assert result["block_size"] == 64
    assert result["estimated_macs_per_sample"] == 1104
    assert result["p95_block_ns"] >= result["median_block_ns"] > 0.0
