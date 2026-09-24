# SSM-WaveNet native tools

Implementation of the unchanged [v1 contract](SSM_WAVENET_NATIVE_FORMAT.md).
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

## Measured (2026-09-24, Xeon E5-2630 v3, machine shared, load average 13 to 26)

Parity (`test_ssm_wavenet_cpp_parity.py`, float32 state, IEEE `-O3`): max absolute error
1.2e-6 to 1.5e-6 at 48 kHz and 2.0e-6 to 2.7e-6 at 96 kHz, for a 1e-5 contract; irregular
blocks and reset exact (0).

Cost, `paired_ssm_wavenet_s925` (8 blocks, 16 channels, 4 modes), median ns per sample:

| engine | block 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|
| SSM-WaveNet, first version (Codex) | 18 899 | 17 191 | 10 792 | 11 302 |
| SSM-WaveNet, 1x1 layers input-major and SSM state as structure of arrays, IEEE | 8 459 | 8 489 | 8 434 | 8 481 |
| same, `-ffast-math` (not the default: parity is only guaranteed for IEEE) | 7 041 | 6 818 | 6 891 | 6 921 |
| NAM A2 lite (Codex, same session) | 966 | 869 | 900 | 612 |
| NAM A2 full (Codex, same session) | 5 256 | 3 806 | 4 125 | 4 213 |

SSM-WaveNet is about twice the cost of A2 full and ten times A2 lite here: 2.5 times
real time at 48 kHz on one core of this machine. A profile of the first version put 47 %
in the SSM recurrence (`std::complex` in arrays of structures), 29 % in the 1x1 layers and
22 % in libm `tanhf`/`expf`. The two rewrites above removed about a fifth of the cost.
Next lever, not taken: approximate tanh/sigmoid as NAM A2 does, which changes parity.
