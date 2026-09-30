"""The bench's --target-advance at the loading level (FIX_ffuzz.md, section 4).

The bench is imported in a subprocess: importing it patches torchaudio.load, the
DataLoader and TMPDIR for the whole interpreter. Like every bench run, it needs the
packages the bench's docstring lists outside uv.lock (lightning, wandb, nablafx).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]

PROBE = """
import json, sys
sys.path.insert(0, "scripts")
import product_nablafx_bench as bench
bench.TARGET_ADVANCE = 4
out = {}
for name in sys.argv[1:]:
    audio, rate = bench._load(name)
    out[name] = {"rate": rate, "samples": audio[0].tolist()}
print(json.dumps(out))
"""


def test_load_advances_targets_only(tmp_path: Path) -> None:
    signal = (np.arange(1, 65, dtype=np.float32) / 128).reshape(-1, 1)
    target = tmp_path / "take.target.wav"
    source = tmp_path / "take.input.wav"
    for path in (target, source):
        sf.write(path, signal, 48_000, subtype="FLOAT")
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(target), str(source)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    moved = np.array(loaded[str(target)]["samples"], dtype=np.float32)
    kept = np.array(loaded[str(source)]["samples"], dtype=np.float32)
    assert loaded[str(target)]["rate"] == 48_000
    assert moved.shape == kept.shape == (64,)
    np.testing.assert_array_equal(moved[:60], signal[4:, 0])
    np.testing.assert_array_equal(moved[60:], 0)
    np.testing.assert_array_equal(kept, signal[:, 0])
