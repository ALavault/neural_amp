# SSM-WaveNet native model format v1

`ssm-wavenet-native-v1` is the exchange format between a trained SSM-WaveNet
(`src/fssr_nam/models/ssm_wavenet.py`) and a native, causal, stateful, sample-by-sample
engine. Product lane (D-PRODUCT-001), fixed on 2026-09-24 before the engine was written.

The engine implements **only the first-order-hold (FOH) discretization**. A ZOH checkpoint
is exported through `zoh_to_foh`, which keeps its response at the training rate exactly
and makes it rate-portable by rescaling dt alone (`demo/sr_transfer/FIX_foh.md`).
Version 1 refuses knob-conditioned models (`num_controls > 0`): FiLM and pole heads are
version 2 material.

## Root

```json
{
  "format": "ssm-wavenet-native-v1",
  "version": 1,
  "precision": "float32",
  "trained_sample_rate_hz": 48000,
  "latency_samples": 0,
  "source": {"run_id": "paired_ssm_wavenet_s925", "checkpoint": "last", "commit": "..."},
  "architecture": {
    "num_blocks": 8, "channels": 16, "state_dim": 4,
    "act_type": "gated", "output_act": "tanh", "discretization": "foh"
  },
  "input_conv": {"weight": [16 floats], "bias": [16 floats]},
  "blocks": [ { ... } ],
  "output_net": {
    "conv1": {"weight": [[16 floats] x 16], "bias": [16 floats]},
    "conv2": {"weight": [[16 floats] x 1], "bias": [1 float]}
  }
}
```

Weights are PyTorch 1x1 `Conv1d` weights with the kernel axis dropped: `weight[out][in]`.
`input_conv` has one input channel, so its weight is a flat list of `channels` values.
`output_act` is `tanh`, `softsign` or `none`. JSON numbers are read once as float64 and
stored as float32, except the SSM parameters, which stay float64 until the coefficients
are computed (below).

## Block

```json
{
  "ssm": {
    "log_dt": [C],
    "lambda_re": [[H] x C], "lambda_im": [[H] x C],
    "c_re": [[H] x C], "c_im": [[H] x C],
    "d": [C]
  },
  "pre_gate": {"weight": [[C] x 2C], "bias": [2C]},
  "res_conv": {"weight": [[C] x C], "bias": [C]},
  "skip_conv": {"weight": [[C] x C], "bias": [C]}
}
```

`lambda_re = -exp(log_A_real)` and `lambda_im = A_imag` are exported already combined;
`c` and `d` are the FOH values after `zoh_to_foh` (d absorbs the m = 0 difference).

## Engine semantics

`set_sample_rate(fs)` (and at load, fs = 48 kHz): for every block, channel `k` and mode `j`,
in float64,

- `dt = exp(log_dt[k]) * trained_sample_rate_hz / fs`
- `z = lambda[k][j] * dt`, `a = exp(z)`
- `phi2(z) = (exp(z) - 1 - z) / z^2`, by the series `1/2 + z/6 + z^2/24` when `|z| < 1e-4`
- `beta1 = dt * phi2(z)`, `beta0 = dt * a * phi2(-z)`

then stored as complex float32 (`a`, `c * beta1`, `c * beta0`). Changing the rate keeps
the state.

Per sample, with `x` the input sample:

1. `h = input_conv.weight * x + input_conv.bias` (C values); `skip = 0` (C values).
2. For each block, with `u` its input (C values):
   - SSM, per channel k: `s[k][j] = a[k][j] * s[k][j] + cb1[k][j] * u[k] + cb0[k][j] * u_prev[k]`
     for every mode j (`cb1 = c * beta1`, `cb0 = c * beta0`), then
     `y[k] = 2 * Re(sum_j s[k][j]) + d[k] * u[k]`, and `u_prev[k] = u[k]`.
     (Equivalently state `h = a h + beta1 u + beta0 u_prev` and output `2 Re(c . h)`;
     folding c into the betas saves one complex multiply per mode.)
   - `v = pre_gate.weight @ y + pre_gate.bias` (2C values);
     `g[k] = tanh(v[k]) * sigmoid(v[C + k])`.
   - `skip += skip_conv.weight @ g + skip_conv.bias`;
     next block input `= res_conv.weight @ g + res_conv.bias + u`.
3. `out = output_act(conv2.weight @ relu(conv1.weight @ relu(skip) + conv1.bias) + conv2.bias)`.

`reset()` zeroes every `s` and `u_prev`. There is no lookahead: latency is 0 samples.

## Parity contract

Against the PyTorch model exported from (`zoh_to_foh` applied, evaluated in float64 with
its FFT forward from a zero state), on a 2 s noise-and-sweep input:

- max absolute error `<= 1e-5` at 48 kHz;
- the same at 96 kHz, the PyTorch model having `log_dt += log(48000 / 96000)`;
- processing the input in irregular block sizes (e.g. 1, 7, 64, 333, rest) gives the same
  output as one block, to `<= 1e-6`;
- after `reset()`, reprocessing the input gives the first output again, to `<= 1e-6`.

If float32 state cannot meet `1e-5` on slow modes, the engine may keep `s` in float64;
both costs are then reported.

## Version 2: FiLM controls

`ssm-wavenet-native-v2` sets `version: 2`. The v1 layout and contract above remain
unchanged, including byte-for-byte export for unconditioned ReLU models. V2 uses
all the same audio weights, ZOH-to-FOH export and FOH recurrence. It requires:

```json
"architecture": {
  "num_blocks": 8, "channels": 16, "state_dim": 4,
  "act_type": "gated", "output_act": "tanh", "discretization": "foh",
  "num_controls": 2, "cond": "film", "cond_head": "joint",
  "cond_hidden": [32], "head_act": "leaky"
}
```

The additional root object `film_net` contains `linear1` and `linear2`, each
with `weight[out][in]` and `bias[out]`. Dimensions are respectively
`[32, num_controls]` and `[num_blocks * 4 * channels, 32]`. Its evaluation is
`linear2(relu(linear1(controls)))`. Only this joint MLP is supported; additive
heads, pole conditioning and `act_type: odd` are rejected explicitly.

Controls are finite normalized values in `[0,1]`; `set_controls(span)` rejects
invalid dimensions or values before changing anything. At load they default to
`0.5`. VS100R's order is **Gain, Contour**, with physical settings divided by 10.
The native engine does not infer knob names from weights; consumers must know
this order. For each block, the corresponding `4C` values are `scale[2C]` then
`shift[2C]`. Immediately after `pre_gate`, before tanh/sigmoid:

```
v = (1 + scale) * v + shift
```

The engine computes the MLP only when `set_controls` receives changed values,
never within `process`. This call allocates no memory and preserves the SSM
state and previous input. `reset` clears audio history but retains controls.
Calls on one instance must be serialized by the caller. Audio blocks can be
split at any control event without resetting; events affect that sample onward.

Both output-head activations use `head_act`: `relu` or `leaky`, where
`leaky(x) = x` for `x >= 0`, otherwise `0.01 * x`, exactly the slope in
`head_activation`. The MLP hidden activation always remains ReLU. Other output
head activations are rejected. The final `output_act` is unchanged from v1.

The numerical v2 tests use float64 PyTorch FFT SSMs and apply FiLM at its event
sample between layers, preserving history across events. Acceptance is maximum
absolute error `<= 3e-6` for the small synthetic fixtures and `<= 1e-5`
(the v1 contract) for the trained VS100R checkpoint. This accounts for float32
native weights, coefficients, state and accumulated arithmetic versus float64
reference arithmetic and amplification of intermediate rounding by trained
weights. The VS100R noise/control-change case exceeds 3e-6, so that stricter
threshold is not claimed for this checkpoint. It is a tested tolerance, not a bound for arbitrary
weights or arbitrarily long signals. Reset and irregular-block parity are
bitwise; prefix rendering verifies causality.

### Product plugin

The JUCE plugin accepts `.nam` via the existing NAM A2 path, or v2 `.json` with
two controls and a 48 kHz training rate. Gain and Contour are automatable 0–10
parameters, normalized by `/10`. A 20 ms linear ramp runs at a 1 kHz control
rate (48 audio samples per tick), with tick phase retained across host blocks.
This avoids an MLP evaluation per audio sample even during automation; held
controls do not reevaluate the MLP. The engine itself accepts instantaneous
events; smoothing belongs to the plugin.

The v2 plugin **requires exactly 48 kHz**. Loading at another host rate returns
an error. If the host changes rate after loading, model processing produces
silence and the editor reports suspension until 48 kHz is restored. There is
no plugin resampler. NAM's existing rate behavior is unchanged.

Export (CPU only; output is ignored by Git):

```sh
export CUDA_VISIBLE_DEVICES="" TMPDIR=/fastdata/lavaulta/tmp
mkdir -p build/vs100r_native
uv run python scripts/product_ssm_export.py --run-id vs100r48fast_film_s42 \
  --out build/vs100r_native/vs100r48fast_film_s42.json
```

`ssm_wavenet_block_runner MODEL INPUT.f32 OUTPUT.f32 BLOCKS 48000 CONTROLS.json`
accepts a strictly increasing event list beginning at sample zero, e.g.
`[{"sample":0,"controls":[0.5,0.5]},{"sample":2107,"controls":[1,0]}]`.
`ssm_wavenet_benchmark MODEL [GAIN CONTOUR]` takes normalized values; its
existing stimulus, warmup and timings are unchanged and measure held controls.
The offline plugin renderer takes optional `[channels [gain contour [sample_rate]]]`
after the block size; its knobs use the physical 0–10 scale.
