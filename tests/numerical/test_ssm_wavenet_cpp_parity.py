from __future__ import annotations

import copy
import importlib.util
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from fssr_nam.models.ssm_wavenet import SSMWaveNet, zoh_to_foh

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "product_ssm_export", ROOT / "scripts/product_ssm_export.py"
)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


@pytest.fixture(scope="session")
def runner(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("cmake") is None or shutil.which("c++") is None:
        pytest.skip("CMake and a C++ compiler are required for native parity")
    build = tmp_path_factory.mktemp("ssm-cpp-build")
    subprocess.run(
        [
            "cmake",
            "-S",
            str(ROOT / "cpp"),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DNAM_TOOLS_FAST_MATH=OFF",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--target", "ssm_wavenet_block_runner", "-j2"],
        check=True,
        capture_output=True,
        text=True,
    )
    return build / "ssm_wavenet_block_runner"


@pytest.mark.parametrize("output_act", ["tanh", "softsign", "none"])
@pytest.mark.parametrize("rate", [48000, 96000])
def test_fft_parity_blocks_reset(
    tmp_path: Path, runner: Path, rate: int, output_act: str
) -> None:
    torch.manual_seed(925)
    model = SSMWaveNet(
        num_blocks=2,
        channels=4,
        state_dim=3,
        discretization="zoh",
        output_act=output_act,
    ).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Conv1d):
                module.weight.uniform_(-0.6, 0.6)
                module.bias.uniform_(-0.2, 0.2)
        # Include slow poles and both phi2 branches, with observable modal residues.
        for block in model.blocks:
            block.ssm.log_dt.copy_(torch.tensor([-9.5, -7.0, -4.0, -2.5]))
    zoh_to_foh(model)
    path = EXPORT.export_model(model, tmp_path / "model.json")
    reference = copy.deepcopy(model).double()
    with torch.no_grad():
        for block in reference.blocks:
            block.ssm.log_dt.add_(math.log(48000 / rate))
    rng = np.random.default_rng(925)
    t = np.arange(2 * rate) / rate
    x = (
        0.2 * rng.standard_normal(len(t))
        + 0.4 * np.sin(2 * np.pi * (20 * t + (10000 - 20) * t**2 / 4))
    ).astype(np.float32)
    x.tofile(tmp_path / "in.f32")
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with torch.no_grad():
            expected = (
                reference(torch.from_numpy(x).double()[None, None]).numpy().ravel()
            )
    finally:
        torch.set_num_threads(threads)

    def run(blocks: str, name: str) -> tuple[np.ndarray, np.ndarray]:
        out = tmp_path / name
        subprocess.run(
            [
                str(runner),
                str(path),
                str(tmp_path / "in.f32"),
                str(out),
                blocks,
                str(rate),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return np.fromfile(out, dtype=np.float32), np.fromfile(
            str(out) + ".reset.f32", dtype=np.float32
        )

    one, reset = run(str(len(x)), "one.f32")
    irregular, _ = run("1,7,64,333", "irregular.f32")
    assert one.shape == expected.shape
    assert np.isfinite(one).all()
    error = float(np.max(np.abs(one - expected)))
    block_error = float(np.max(np.abs(irregular - one)))
    reset_error = float(np.max(np.abs(reset - one)))
    print(
        f"{rate=} {output_act=} max_abs={error:.9g} "
        f"blocks={block_error} reset={reset_error}"
    )
    assert error <= 1e-5
    assert block_error <= 1e-6
    assert reset_error <= 1e-6
    # A separately rendered prefix has no access to the future samples.
    x[:1024].tofile(tmp_path / "in.f32")
    prefix, _ = run("7,64", "prefix.f32")
    np.testing.assert_array_equal(prefix, one[:1024])


def test_export_converts_copy_and_refuses_unsupported(tmp_path: Path) -> None:
    model = SSMWaveNet(num_blocks=1, channels=2, state_dim=2, discretization="zoh")
    EXPORT.export_model(model, tmp_path / "model.json")
    assert model.blocks[0].ssm.discretization == "zoh"
    for kwargs in (
        {"discretization": "free"},
        {"num_controls": 1, "cond": "poles", "discretization": "zoh"},
        {"act_type": "sine"},
    ):
        with pytest.raises(ValueError):
            EXPORT.export_model(SSMWaveNet(**kwargs), tmp_path / "bad.json")
