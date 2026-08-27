"""Machine-readable Python/C++ parity verification for R1 native inference."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .r1_native import R1NativeReference, load_r1_native_payload


def _comparison(
    expected: np.ndarray, actual: np.ndarray, *, atol: float, rtol: float
) -> tuple[bool, float | None]:
    if expected.shape != actual.shape or not np.isfinite(actual).all():
        return False, None
    errors = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    maximum = float(errors.max(initial=0.0))
    limits = atol + rtol * np.abs(expected.astype(np.float64))
    return bool(np.all(errors <= limits)), maximum


def verify_r1_cpp_parity(
    model_path: str | Path,
    runner_path: str | Path,
    samples: Any,
    *,
    irregular_blocks: Sequence[int] = (1, 7, 64, 3, 128, 17),
    regular_block: int = 64,
    atol: float = 2.0e-5,
    rtol: float = 2.0e-5,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run both block modes plus reset and return the parity evidence record."""
    if (
        regular_block < 1
        or not irregular_blocks
        or any(size < 1 for size in irregular_blocks)
    ):
        raise ValueError("parity block sizes must be positive")
    payload = load_r1_native_payload(model_path)
    signal = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not np.isfinite(signal).all():
        raise ValueError("parity input must be finite")
    reference = R1NativeReference(payload)
    expected = reference.process(signal)
    runner = Path(runner_path)
    if not runner.is_file():
        raise FileNotFoundError(runner)
    with tempfile.TemporaryDirectory(prefix="fssr-r1-parity-") as directory:
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
                ",".join(str(size) for size in irregular_blocks),
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
        "python_vs_cpp_regular": _comparison(expected, regular, atol=atol, rtol=rtol),
        "python_vs_cpp_irregular": _comparison(
            expected, irregular, atol=atol, rtol=rtol
        ),
        "reset_repeat": _comparison(irregular, reset, atol=0.0, rtol=0.0),
        "regular_vs_irregular": _comparison(regular, irregular, atol=0.0, rtol=0.0),
    }
    passed = all(result[0] for result in comparisons.values())
    finite_errors = [
        result[1] for result in comparisons.values() if result[1] is not None
    ]
    maximum = max(finite_errors) if len(finite_errors) == len(comparisons) else None
    residual = payload.get("residual")
    slow = payload["slow_controller"]
    coverage = {
        "core": payload["core"]["kind"],
        "residual": "none" if residual is None else residual["kind"],
        "receptive_field": None if residual is None else residual["receptive_field"],
        "precision": "float32",
        "slow_controller": "none" if slow == "none" else slow["kind"],
    }
    report: dict[str, Any] = {
        "format": "fssr-r1-parity-v1",
        "python_cpp_parity": passed,
        "passed": passed,
        "max_abs_error": maximum,
        "atol": atol,
        "rtol": rtol,
        "sample_count": int(signal.size),
        "regular_block": regular_block,
        "irregular_blocks": list(irregular_blocks),
        "reset_verified": comparisons["reset_repeat"][0],
        "covered_compositions": [coverage],
        "comparisons": {
            name: {"passed": result[0], "max_abs_error": result[1]}
            for name, result in comparisons.items()
        },
    }
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, allow_nan=False, sort_keys=True, indent=2)
            stream.write("\n")
    return report


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("runner", type=Path)
    parser.add_argument("input_f32", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    parser.add_argument("--rtol", type=float, default=2.0e-5)
    arguments = parser.parse_args()
    signal = np.fromfile(arguments.input_f32, dtype=np.float32)
    report = verify_r1_cpp_parity(
        arguments.model,
        arguments.runner,
        signal,
        atol=arguments.atol,
        rtol=arguments.rtol,
        report_path=arguments.report,
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
