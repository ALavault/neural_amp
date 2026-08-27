"""Generate, persist, reload, and verify the M0 identity fixture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from fssr_nam.data.synthetic import generate_identity_fixture
from fssr_nam.metrics.time import error_to_signal_ratio


def main() -> None:
    fixture = generate_identity_fixture(seed=0)
    output_dir = Path("experiments/summaries/m0_identity")
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "input.wav"
    output_path = output_dir / "output.wav"
    wavfile.write(input_path, fixture.sample_rate, fixture.input)
    wavfile.write(output_path, fixture.sample_rate, fixture.output)

    input_rate, reloaded_input = wavfile.read(input_path)
    output_rate, reloaded_output = wavfile.read(output_path)
    if input_rate != fixture.sample_rate or output_rate != fixture.sample_rate:
        raise RuntimeError("sample rate changed during round trip")
    if not np.array_equal(reloaded_input, reloaded_output):
        raise RuntimeError("identity samples changed during round trip")

    summary = {
        "schema_version": 1,
        "system": "identity",
        "sample_rate": fixture.sample_rate,
        "sample_count": int(fixture.input.size),
        "seed": fixture.seed,
        "dtype": str(reloaded_input.dtype),
        "esr": error_to_signal_ratio(reloaded_output, reloaded_input),
        "status": "passed",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
