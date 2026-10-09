# SSM-WaveNet native tools

Implementation of [v1 and v2 contracts](SSM_WAVENET_NATIVE_FORMAT.md).
V1 remains unconditioned; v2 adds joint FiLM and ReLU/LeakyReLU heads.
Product engineering only; no campaign selection or scientific claims.

```sh
uv run python scripts/product_ssm_export.py \
  --run-id paired_ssm_wavenet_s925 --checkpoint last --out /tmp/ssm.json
cmake -S cpp -B /tmp/ssm-build -DCMAKE_BUILD_TYPE=Release -DNAM_TOOLS_FAST_MATH=OFF
cmake --build /tmp/ssm-build --target ssm_wavenet_block_runner ssm_wavenet_benchmark -j2
/tmp/ssm-build/ssm_wavenet_block_runner /tmp/ssm.json input.f32 output.f32 1,7,64,333 96000
/tmp/ssm-build/ssm_wavenet_benchmark /tmp/ssm.json
```

The runner accepts comma-separated positive block sizes and an optional sample
rate (default 48000). It cycles the block sizes, truncating the final block. It
also resets and reprocesses the same input into `OUTPUT.reset.f32`. Files contain
raw native-endian float32 mono samples. A single block may span the whole file.

`fssr::ssm::Model` loads and resets at 48 kHz. `set_sample_rate` retains state;
call `reset` explicitly for a new stream. `process` supports in-place buffers,
allocates no memory, and uses float32 state and coefficients. Continuous SSM
parameters and FOH coefficient calculations use float64. The exporter converts
a copy, leaving an in-memory source model unchanged. CLI loading uses the same
CPU checkpoint loader as `product_sr_transfer.py`.

The benchmark emits one JSON record per block size (32, 64, 128, 256), using
48 kHz, three two-second warmup passes and five two-second timed passes. Its
stimulus, per-block clock timing and percentile calculation match
`nam_benchmark MODEL BLOCK 2 5`. Compare median ns/sample; realtime factor is
`1e9 / (48000 * ns_per_sample)` (higher is faster). Build NAM with
`NAM_TOOLS_FAST_MATH=OFF` for the same arithmetic policy. The SSM target always
uses `-O3`, independently of that option.

```sh
uv run pytest -q -s tests/numerical/test_ssm_wavenet_cpp_parity.py
```

The test compares two-second noise-and-sweep inputs against PyTorch float64 FFT
at 48 and 96 kHz, including slow modes, all output activations, irregular blocks,
reset and causality. `-s` displays the measured maximum errors.

## Measured (2026-09-24, Xeon E5-2630 v3, machine shared)

Parity (`test_ssm_wavenet_cpp_parity.py`, float32 state, IEEE `-O3`): max absolute error
1.2e-6 to 1.5e-6 at 48 kHz and 2.0e-6 to 2.7e-6 at 96 kHz, for a 1e-5 contract; irregular
blocks and reset exact (0).

Cost, like for like (`scripts/product_ssm_cost.sh`, raw lines with build and load average
in `demo/plugin/ssm_cost_2026-09-24.jsonl`): the three engines back to back, each build
applied to all of them. SSM-WaveNet is `paired_ssm_wavenet_s925` (8 blocks, 16 channels,
4 modes) exported to FOH; NAM A2 lite and full are the Big Muff models of
`scripts/product_report.py`. Median ns per sample, block sizes 32 / 64 / 128 / 256:

| build | A2 lite | A2 full | SSM-WaveNet |
|---|---|---|---|
| IEEE `-O3` (`NAM_TOOLS_FAST_MATH=OFF`) | 583 / 555 / 535 / 541 | 3 244 / 2 965 / 2 900 / 2 828 | 8 438 / 8 431 / 8 414 / 8 422 |
| fast-math (`ON`, SSM with `-ffast-math`) | 540 / 492 / 739 / 434 | 3 438 / 3 304 / 3 118 / 3 077 | 6 792 / 6 839 / 6 855 / 6 869 |

SSM-WaveNet costs about 3 times A2 full in IEEE and about 2 times in fast-math, 12 to 15
times A2 lite; 2.5 to 3 times real time at 48 kHz on one core of this machine. A2 keeps
its approximate activations (`NAM_ENABLE_A2_FAST`) in both builds; SSM-WaveNet uses libm.
Load average during the runs: 10 to 15 on 32 threads.

History: Codex's first version measured 10.8 to 18.9 us per sample. A profile put 47 % in
the SSM recurrence (`std::complex` in arrays of structures), 29 % in the 1x1 layers and
22 % in libm `tanhf`/`expf`; storing the 1x1 weights input-major and the SSM state as a
structure of arrays brought it to 8.4 us. Next lever, not taken: approximate tanh and
sigmoid as A2 does, which changes parity.

## VS100R v2 validation

Use `CUDA_VISIBLE_DEVICES=""` and `TMPDIR=/fastdata/lavaulta/tmp` for all Python
commands. The export command and plugin semantics are documented in the
[v2 format section](SSM_WAVENET_NATIVE_FORMAT.md#version-2-film-controls).

```sh
.venv/bin/pytest -q -s tests/numerical/test_ssm_wavenet_cpp_parity.py \
  tests/numerical/test_ssm_wavenet_cpp_v2.py
FSSR_VS100R_NATIVE_CAPTURE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/pytest -q -s tests/numerical/test_ssm_wavenet_cpp_v2.py \
  -k vs100r_checkpoint
cmake -S demo/plugin -B build/demo_plugin -DCMAKE_BUILD_TYPE=Release
cmake --build build/demo_plugin -j2
```

The opt-in test reads `vs100r48fast_film_s42` and the first held-out VS100R take
returned by `product_vs100r_eval.takes()`. It decimates the complete `di_test`
loopback section by 1/4, quantizes input once to float32 for both engines, and
compares with float64 PyTorch after the same float32 ZOH-to-FOH conversion as
export. ESR is `sum((native-reference)^2) / sum(reference^2)`, over the entire
section from zero state, with no cropping or alignment. This tests implementation
parity, not fidelity to the amp. Captured audio and checkpoints remain local.

The offline harness checks finite output and identical mono/stereo output. For a
v2 JSON it additionally checks silence plus an explanatory warning after the host
changes to 96 kHz, then restores 48 kHz before rendering. Optional Gain/Contour
arguments are applied through JUCE's automatable parameter API.

### Local check, 2026-10-09

V1 checkpoint `paired_ssm_wavenet_s925`, `last`: the current exporter and the
pre-change exporter produced identical 403,592-byte JSON files. VS100R's v2
export is local at `build/vs100r_native/vs100r48fast_film_s42.json` (860,938
bytes; ignored by Git). No run registry was modified, per task constraints.

Synthetic v2 tests: 11 passed (private-capture test opt-in), maximum absolute
error 8.30497303e-7 (ReLU), 8.44501613e-7 (leaky); reset and irregular blocks
bitwise equal. The v1 cases passed at 48/96 kHz (maximum 2.6794619e-6).
The first synthetic fixture had a closed untrained ReLU head, so the test's
control-effect assertion failed; the fixture now deliberately exposes positive
and negative branches. No production tolerance was relaxed.

Held-control benchmark, IEEE `-O3`, GCC 15.2, Xeon E5-2630 v3, shared host,
original warmup and
five timed passes; v2 controls `(0.5, 0.5)`. Raw records remain in
`build/vs100r_native/cost_v1.jsonl` and `cost_v2.jsonl`:

| Block | v1 median ns/sample | v2 median ns/sample |
|---|---:|---:|
| 32 | 8574.328125 | 9379.406250 |
| 64 | 8305.546875 | 9375.382813 |
| 128 | 8458.132813 | 9604.976563 |
| 256 | 8422.585938 | 9475.480469 |

These compare the unconditioned paired checkpoint with the VS100R checkpoint,
whose weights and output-head activation differ. They do not isolate the cost
of FiLM alone, and do not measure continuously automated controls. The host was
not isolated from other jobs. No scientific claim is made.

Validation commands in this checkout used `.venv/bin/python`, `.venv/bin/pytest`
and `.venv/bin/ruff`: `make lint` exited 2 because the installed Snap `uv`
launcher refused to start (`snap-confine` / AppArmor). The equivalent commands
`.venv/bin/ruff check src tests scripts diagnosis/seeds diagnosis/butterfly` and
`.venv/bin/ruff format --check src tests scripts` passed (374 files formatted).
`cmake --build build/fssr_cpp -j2` completed successfully.

The checkpoint's synthetic noise test did not meet a strict 3e-6 threshold at
all controls. Exploratory double accumulation in the control MLP and then in
audio affine layers did not resolve the changing-control case and was reverted.
The checkpoint test uses the v1 contractual 1e-5 limit; the smaller synthetic
fixtures retain 3e-6. This is an explicit relaxation of the initial 3e-6 target,
not a claim of increased model fidelity.

Final native parity command (CPU, two Torch threads, capture opt-in):

```sh
CUDA_VISIBLE_DEVICES="" TMPDIR=/fastdata/lavaulta/tmp OMP_NUM_THREADS=2 \
MKL_NUM_THREADS=2 FSSR_VS100R_NATIVE_CAPTURE=1 \
MPLCONFIGDIR=/fastdata/lavaulta/tmp/mpl-vs100r-native \
.venv/bin/pytest tests/numerical/test_ssm_wavenet_cpp_v2.py \
  tests/numerical/test_ssm_wavenet_cpp_parity.py -q -s
```

Result: **19 passed in 256.09 s**. VS100R maximum absolute error on one second
of seeded noise: 9.1802795e-7 at `(0,1)`, 4.12496923e-6 at `(0.5,0.25)`,
7.8657407e-7 at `(1,0)`; 6.69378679e-6 when switching at samples 16007 and 32011.
The 30 s real `di_test` section (1,440,000 samples), take order 2,
`G075_C025_B075_M025_T050`, normalized controls `(0.75,0.25)`, gives maximum
error **5.18048048e-6**, ESR **1.51911859e-12** against float64 PyTorch.
Reset and irregular-block comparisons are bitwise equal. Output shape and
finite values pass. Full command output is local in `build/vs100r_native/parity.txt`.

Offline renderer checks (one second of seed-42 noise, amplitude 0.12, stereo):

```sh
build/demo_plugin/FssrAmpOfflineRender_artefacts/Release/FssrAmpOfflineRender \
  build/vs100r_native/vs100r48fast_film_s42.json \
  build/vs100r_native/plugin_input.f32 build/vs100r_native/plugin_low.f32 \
  256 2 2.5 2.5 48000
```

Exit 0 for `(2.5,2.5)` and `(7.5,7.5)`; finite outputs, max setting difference
**0.6087586283683777**. Blocks 17/256 and stereo channels are bitwise identical.
The `(2.5,2.5)` render is bitwise identical to `ssm_wavenet_block_runner` with
normalized `(0.25,0.25)`. Rates 44100 and 96000 return exit 2 with
`modele v2 refuse: l'hote doit fonctionner a 48000 Hz`. The harness's internal
post-load rate-switch check also passes. NAM A2
`product_a2_electro_harmonix_big_muff_seed0_v2/model_lite.nam` renders successfully
and matches `nam_block_runner` bitwise (max error 0). The local script
`build/vs100r_native/verify_plugin.py` records exact invocations and assertions;
its output is `plugin_verification.txt` in the same directory.

Initial build errors (runner JSON include path and an ambiguous JUCE string
comparison) were corrected. No manual DAW/GUI session, listening evaluation or
continuous-automation cost measurement was performed. All Python ran with CUDA
hidden. No checkpoint, raw audio or exported model is tracked.

Review fix (Codex read-only review, 2026-10-09): float32 `SmoothedValue` ramps can
overshoot [0, 1] by one ulp, and `set_controls` throws outside that range, i.e. in
the audio callback. The plugin now clamps the smoothed controls before the call.
After the fix, `build/vs100r_native/verify_plugin.py` passes unchanged (same
max setting difference, plugin/native bitwise identical), and the full suite
`CUDA_VISIBLE_DEVICES="" make test` gives 796 passed, 1 skipped (the opt-in
real-capture test, run separately above).

Final plugin build: `cmake -S demo/plugin -B build/demo_plugin
-DCMAKE_BUILD_TYPE=Release` and `cmake --build build/demo_plugin -j2` both exited
0. VST3, Standalone and the offline renderer were built; compiler warnings remain
in JUCE/NAM and for intentional exact sample-rate comparisons. Build output is
local in `build/vs100r_native/plugin_build_retry.txt`.

Exact held-control cost commands (both exited 0):

```sh
build/fssr_cpp/ssm_wavenet_benchmark \
  build/vs100r_native/paired_ssm_wavenet_s925.json
build/fssr_cpp/ssm_wavenet_benchmark \
  build/vs100r_native/vs100r48fast_film_s42.json 0.5 0.5
```
