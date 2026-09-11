from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from fssr_nam.inference import export_r2_native_model, verify_r2_cpp_parity
from fssr_nam.models import AAFSSR, AAFSSRXL, AANAM


@pytest.fixture(scope="session")
def r2_cpp_tools(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    if shutil.which("cmake") is None or shutil.which("c++") is None:
        pytest.skip("CMake and a C++ compiler are required for native parity")
    build = tmp_path_factory.mktemp("r2-cpp-build")
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
            "r2_block_runner",
            "r2_benchmark",
            "-j2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {"runner": build / "r2_block_runner", "benchmark": build / "r2_benchmark"}


def _make_nontrivial(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for parameter_index, parameter in enumerate(model.parameters()):
            positions = torch.arange(parameter.numel(), dtype=parameter.dtype)
            values = 0.02 * torch.sin(0.19 * positions + 0.13 * parameter_index)
            parameter.copy_(values.reshape_as(parameter))


@pytest.mark.parametrize("mode", ["off", "adaa1", "full_island_x2", "teacher_x4"])
def test_r2_cpp_matches_python_for_regular_irregular_and_reset_blocks(
    tmp_path: Path, r2_cpp_tools: dict[str, Path], mode: str
) -> None:
    torch.manual_seed(20260828)
    model = AAFSSR(core_kind="cascade", aa_mode=mode)
    _make_nontrivial(model)
    signal = torch.linspace(-0.6, 0.7, 257) + 0.04 * torch.sin(torch.arange(257) * 0.21)
    model_path = export_r2_native_model(model, tmp_path / f"{mode}.json")
    report = verify_r2_cpp_parity(
        model_path,
        r2_cpp_tools["runner"],
        signal.numpy(),
        report_path=tmp_path / f"{mode}-parity.json",
    )
    assert report["python_cpp_parity"] is True
    assert report["max_abs_error"] <= 2.0e-5
    assert report["reset_verified"] is True
    assert report["aa_mode"] == mode
    assert report["sizes"]["weight_bytes"] > 0


@pytest.mark.parametrize("mode", ["full_island_x2", "teacher_x4"])
def test_quality_aa_latency32_cpp_matches_python(
    tmp_path: Path, r2_cpp_tools: dict[str, Path], mode: str
) -> None:
    torch.manual_seed(20260828)
    model = AAFSSR(core_kind="cascade", aa_mode=mode, aa_latency_samples=32)
    _make_nontrivial(model)
    signal = torch.linspace(-0.6, 0.7, 257) + 0.04 * torch.sin(torch.arange(257) * 0.21)
    model_path = export_r2_native_model(model, tmp_path / f"quality-{mode}.json")
    report = verify_r2_cpp_parity(
        model_path,
        r2_cpp_tools["runner"],
        signal.numpy(),
        irregular_blocks=(1, 7, 3, 31, 5, 64),
    )
    assert report["passed"] is True
    assert report["max_abs_error"] <= 2.0e-5
    assert report["latency_samples"] == 32


def test_common_a2_r2_binary_reports_all_blocks_but_smoke_is_not_scientific(
    tmp_path: Path, r2_cpp_tools: dict[str, Path]
) -> None:
    repository = Path(__file__).resolve().parents[2]
    a2_model = repository / "experiments/runs/m2_a2_tanh_seed0_v3/model_full.nam"
    if not a2_model.is_file():
        pytest.skip("reproduced A2 native model is not available")
    r2_model = export_r2_native_model(
        AAFSSR(core_kind="mono", aa_mode="off"), tmp_path / "benchmark-r2.json"
    )
    completed = subprocess.run(
        [
            str(r2_cpp_tools["benchmark"]),
            str(a2_model),
            str(r2_model),
            "12145",
            "48580",
            "100",
            "100",
            "0.00001",
            "1",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["format"] == "fssr-r2-interleaved-benchmark-v1"
    assert result["same_binary"] is True
    assert result["compiler_optimization"] == "-Ofast"
    assert result["lto_ipo"] is True
    assert result["scientific_eligible"] is False
    assert set(result["models"]["a2"]["blocks"]) == {"1", "16", "64", "128"}
    assert set(result["models"]["candidate"]["blocks"]) == {
        "1",
        "16",
        "64",
        "128",
    }


def test_aa_nam_adaa_cpp_matches_python_with_padded_reset(
    tmp_path: Path, r2_cpp_tools: dict[str, Path]
) -> None:
    torch.manual_seed(20260828)
    model_path = export_r2_native_model(
        AANAM(aa_mode="adaa1"), tmp_path / "aa-nam-adaa1.json"
    )
    signal = torch.linspace(-0.2, 0.3, 33)
    report = verify_r2_cpp_parity(
        model_path,
        r2_cpp_tools["runner"],
        signal.numpy(),
        irregular_blocks=(1, 7, 3, 13),
    )
    assert report["passed"] is True
    assert report["max_abs_error"] <= 2.0e-5


def test_ambitious_xl_candidate_keeps_cpp_parity(
    tmp_path: Path, r2_cpp_tools: dict[str, Path]
) -> None:
    torch.manual_seed(20260828)
    model = AAFSSRXL(aa_mode="full_island_x2")
    _make_nontrivial(model)
    signal = torch.linspace(-0.55, 0.61, 113) + 0.03 * torch.sin(
        torch.arange(113) * 0.17
    )
    model_path = export_r2_native_model(model, tmp_path / "aa-fssr-xl.json")
    report = verify_r2_cpp_parity(
        model_path,
        r2_cpp_tools["runner"],
        signal.numpy(),
        irregular_blocks=(1, 11, 3, 29, 7),
    )
    assert report["python_cpp_parity"] is True
    assert report["max_abs_error"] <= 2.0e-5
    assert report["reset_verified"] is True
