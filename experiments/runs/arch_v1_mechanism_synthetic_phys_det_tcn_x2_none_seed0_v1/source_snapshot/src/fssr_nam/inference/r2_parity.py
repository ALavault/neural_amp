"""Python/C++ block, irregular-block, and reset parity for R2 exports."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .r2_native import R2NativeReference, load_r2_native_payload


def _comparison(
    expected: np.ndarray, actual: np.ndarray, atol: float
) -> dict[str, Any]:
    if expected.shape != actual.shape or not np.isfinite(actual).all():
        return {"passed": False, "max_abs_error": None}
    maximum = float(np.max(np.abs(expected.astype(np.float64) - actual)))
    return {"passed": maximum <= atol, "max_abs_error": maximum}


def verify_r2_cpp_parity(
    model_path: str | Path,
    runner_path: str | Path,
    samples: Any,
    *,
    atol: float = 2.0e-5,
    regular_block: int = 64,
    irregular_blocks: tuple[int, ...] = (1, 7, 31, 3, 128, 5),
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the frozen maximum-absolute-error parity contract."""
    if atol <= 0.0 or regular_block < 1 or any(block < 1 for block in irregular_blocks):
        raise ValueError("parity tolerance and block sizes must be positive")
    payload = load_r2_native_payload(model_path)
    signal = np.asarray(samples, dtype=np.float32).reshape(-1)
    if signal.size < 1 or not np.isfinite(signal).all():
        raise ValueError("R2 parity input must be finite and non-empty")
    expected = R2NativeReference(payload).process(signal)
    runner = Path(runner_path)
    if not runner.is_file():
        raise FileNotFoundError(runner)
    with tempfile.TemporaryDirectory(prefix="fssr-r2-parity-") as directory:
        temporary = Path(directory)
        input_path = temporary / "input.f32"
        regular_path = temporary / "regular.f32"
        irregular_path = temporary / "irregular.f32"
        reset_path = temporary / "reset.f32"
        signal.tofile(input_path)
        subprocess.run(
            [
                str(runner),
                str(model_path),
                str(input_path),
                str(regular_path),
                str(regular_block),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                str(runner),
                str(model_path),
                str(input_path),
                str(irregular_path),
                ",".join(str(block) for block in irregular_blocks),
                str(reset_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        regular = np.fromfile(regular_path, dtype=np.float32)
        irregular = np.fromfile(irregular_path, dtype=np.float32)
        reset = np.fromfile(reset_path, dtype=np.float32)
    comparisons = {
        "python_vs_cpp_regular": _comparison(expected, regular, atol),
        "python_vs_cpp_irregular": _comparison(expected, irregular, atol),
        "reset_repeat": _comparison(irregular, reset, 0.0),
        "regular_vs_irregular": _comparison(regular, irregular, 0.0),
    }
    passed = all(comparison["passed"] for comparison in comparisons.values())
    errors = [
        comparison["max_abs_error"]
        for comparison in comparisons.values()
        if comparison["max_abs_error"] is not None
    ]
    report = {
        "format": "fssr-r2-parity-v1",
        "python_cpp_parity": passed,
        "passed": passed,
        "max_abs_error": max(errors) if len(errors) == len(comparisons) else None,
        "atol": atol,
        "sample_count": int(signal.size),
        "regular_block": regular_block,
        "irregular_blocks": list(irregular_blocks),
        "reset_verified": comparisons["reset_repeat"]["passed"],
        "aa_mode": payload["aa_mode"],
        "internal_sample_rate": payload["internal_sample_rate"],
        "dilation_scale": payload["dilation_scale"],
        "latency_samples": payload["latency_samples"],
        "sizes": payload["sizes"],
        "comparisons": comparisons,
    }
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, allow_nan=False, sort_keys=True, indent=2)
            stream.write("\n")
    return report
