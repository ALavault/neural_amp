"""Check that the plugin's own processBlock matches the native runner.

The plugin adds float/double casts, denormal flushing and gain staging on top of
the engine measured in the fact sheet; this renders the same test input through
both paths and compares them sample by sample.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from product_report import MANIFEST, ROOT, RUNS, build_tools, run_native

from fssr_nam.product.data import device_pairs

TOLERANCE = 1e-5
SAMPLES = 480_000
RENDER = (
    ROOT
    / "build/demo_plugin/FssrAmpOfflineRender_artefacts/Release"
    / "FssrAmpOfflineRender"
)


def main() -> int:
    _, runner = build_tools()
    work = pathlib.Path(__file__).parent.parent / "build/demo_plugin/parity"
    work.mkdir(parents=True, exist_ok=True)
    worst = 0.0
    for device, run_id in RUNS.items():
        pairs = device_pairs(MANIFEST, device, root=ROOT)
        signal, _ = sf.read(pairs["test"][0], dtype="float32")
        signal = np.ascontiguousarray(signal[:SAMPLES])
        (work / "in.f32").write_bytes(signal.astype(np.float32).tobytes())
        for label in ("lite", "full"):
            model = ROOT / "demo/runs" / run_id / f"model_{label}.nam"
            # 2 channels: the renderer also asserts both outputs are identical,
            # which is what the standalone needs to feed both speakers.
            subprocess.run(
                [
                    str(RENDER),
                    str(model),
                    str(work / "in.f32"),
                    str(work / "out.f32"),
                    "64",
                    "2",
                ],
                check=True,
            )
            plugin = np.fromfile(work / "out.f32", dtype=np.float32)
            native, _ = run_native(runner, model, signal, "64")
            count = min(len(plugin), len(native))
            difference = float(np.max(np.abs(plugin[:count] - native[:count])))
            worst = max(worst, difference)
            print(f"{device} {label}: max abs diff {difference:.3e}")
            if not np.all(np.isfinite(plugin)):
                print(f"non-finite plugin output for {device} {label}", file=sys.stderr)
                return 1
    if worst > TOLERANCE:
        print(f"plugin parity {worst:.3e} above {TOLERANCE:.0e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
