from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from fssr_nam.campaign.amp_quality_arch_v1 import CANDIDATES
from fssr_nam.models import DEPLOYMENT_PROFILES, build_arch_v1_candidate


@pytest.fixture(scope="session")
def arch_skeleton_runner(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("cmake") is None or shutil.which("c++") is None:
        pytest.skip("CMake and a C++ compiler are required for native checks")
    root = Path(__file__).resolve().parents[2]
    build = tmp_path_factory.mktemp("amp-arch-skeleton-cpp")
    subprocess.run(
        [
            "cmake",
            "-S",
            str(root / "cpp"),
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
            "amp_arch_skeleton_runner",
            "--parallel",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return build / "amp_arch_skeleton_runner"


@pytest.mark.parametrize("family", CANDIDATES)
@pytest.mark.parametrize("profile", DEPLOYMENT_PROFILES)
def test_native_skeleton_is_block_invariant_allocation_free_and_size_matched(
    arch_skeleton_runner: Path, family: str, profile: str
) -> None:
    completed = subprocess.run(
        [str(arch_skeleton_runner), family, profile],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    model = build_arch_v1_candidate(family, profile=profile)
    assert report["format"] == "fssr-amp-arch-skeleton-check-v1"
    assert report["finite"] is True
    assert report["audio_loop_allocations"] == 0
    assert report["block_max_abs_error"] <= 2.0e-6
    assert report["reset_max_abs_error"] <= 2.0e-6
    assert report["latency_samples"] == 32
    assert report["parameters"] == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert report["weight_bytes"] == 4 * report["parameters"]
    assert report["persistent_state_bytes"] > 0
    assert report["scratch_bytes"] > 0


def test_native_skeleton_rejects_unregistered_family_and_profile(
    arch_skeleton_runner: Path,
) -> None:
    invalid_family = subprocess.run(
        [str(arch_skeleton_runner), "result_dependent_family", "slim"],
        check=False,
        capture_output=True,
        text=True,
    )
    invalid_profile = subprocess.run(
        [str(arch_skeleton_runner), "micro_tcn_x2", "result_dependent_profile"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert invalid_family.returncode != 0
    assert "unknown AMP-QUALITY-ARCH-v1 family" in invalid_family.stderr
    assert invalid_profile.returncode != 0
    assert "unknown AMP-QUALITY-ARCH-v1 profile" in invalid_profile.stderr
