# FSSR-R1 native model format v1

`fssr-r1-native-v1` is the exchange format for the diagnostic R1 structured
candidate and its immediate ablation.  JSON numbers are converted once to IEEE
754 `float32`; native inference is mono, causal, stateful, and has zero added
latency.

This version supports either `"slow_controller":"none"` for focused ablations,
or exactly the existing `SlowStateController`: one PyTorch-compatible GRUCell
fed every completed causal decimation interval.  No other controller or
modulation route is accepted, so an unknown slow model cannot accidentally
receive a parity claim.

## Root and core

The required root fields are:

```json
{
  "format": "fssr-r1-native-v1",
  "version": 1,
  "campaign_version": "FSSR-R1",
  "precision": "float32",
  "sample_rate_hz": 48000,
  "latency_samples": 0,
  "slow_controller": "none",
  "core": {},
  "residual": null
}
```

The core is either `mono` (one spline, two FIRs) or `cascade` (two splines,
three FIRs):

```json
{
  "kind": "cascade",
  "filters": [
    {"coefficients": [0.0, 1.0]},
    {"coefficients": [0.0, 1.0]},
    {"coefficients": [0.0, 1.0]}
  ],
  "shapers": [
    {
      "knots": [-2.0, -0.5, 0.5, 2.0],
      "values": [-2.0, -0.5, 0.5, 2.0],
      "slopes": [1.0, 1.0, 1.0, 1.0],
      "drive": 1.0,
      "offset": 0.0
    },
    {
      "knots": [-2.0, -0.5, 0.5, 2.0],
      "values": [-2.0, -0.5, 0.5, 2.0],
      "slopes": [1.0, 1.0, 1.0, 1.0],
      "drive": 1.0,
      "offset": 0.0
    }
  ],
  "output_gain": 1.0
}
```

For FIR coefficients, index zero multiplies the oldest sample and the last
coefficient multiplies the current sample.  Spline knots must form the uniform
grid used by `SmoothHermiteSpline`.  A core executes
`H0 -> (drive*spline input + offset) -> spline1 -> H1`, optionally followed by
`spline2 -> H2`, then `output_gain`.  The Hermite formula and linear endpoint
extrapolation match `SmoothHermiteSpline`.

## Slow controller

The controller object has `kind: "fssr-slow-gru-v1"`, positive `hidden_size`
and `decimation`, and the exact PyTorch GRUCell arrays `weight_ih`, `weight_hh`,
`bias_ih`, and `bias_hh`.  Gate order is `[reset,update,new]`.  Its three input
features are interval means of `[abs(x),x*x,x]`; modulation is emitted from the
old hidden state before the current sample is accumulated.  Projection arrays
produce drive, offset, and gain through
`[exp(.25*tanh(.)), .1*tanh(.), exp(.25*tanh(.))]`.

For mono, these modulate the only shaper drive/offset and final output gain.  For
cascade, they modulate only the first shaper drive/offset and final output gain;
the second shaper is not modulated.  The `application`, `feature_order`,
`gate_order`, and range constants are literal validated fields.

## Residual

The optional residual has exactly 8 channels, kernel size 3, and one of two
registered dilation schedules:

- RF31: `[1,2,4,8]`;
- RF2047: `[1,2,4,8,16,32,64,128,256,512]`.

Its object contains:

```json
{
  "kind": "causal-depthwise-separable-tcn",
  "input_features": ["input", "core"],
  "channels": 8,
  "kernel_size": 3,
  "dilations": [1, 2, 4, 8],
  "receptive_field": 31,
  "negative_slope": 0.01,
  "scale": 0.05,
  "input_projection": {"weight": [[0.0, 0.0]], "bias": [0.0]},
  "layers": [{
    "depthwise_weight": [[0.0, 0.0, 0.0]],
    "depthwise_bias": [0.0],
    "pointwise_weight": [[0.0]],
    "pointwise_bias": [0.0]
  }],
  "output_projection": {"weight": [0.0], "bias": 0.0}
}
```

The abbreviated arrays above illustrate orientation only; every channel
dimension must be length 8.  Matrices are `[output][input]`.  Depthwise kernel
index zero is the oldest tap and index two is current.  Each layer computes
`hidden += leaky_relu(pointwise(depthwise(hidden)))`.  The residual output is
`scale*tanh(linear(hidden))`, added to the completed core.  `input_features`
fixes the projection order and is validated literally.

For scientific continuity, the S3/RF31 ablation may instead use
`"kind":"causal-full-convolution-tcn"`.  Its layers replace the four separable
arrays by `convolution_weight` with shape `[output][input][tap]` and
`convolution_bias` with shape `[output]`.  Full convolution is registered only
for RF31.  RF2047 is always the ten-layer depthwise-separable candidate.  The
exporter recognizes the historical `FastResidualTCN.layers[i].conv` layout and
never silently relabels it as separable.

`R1NativeReference` in `src/fssr_nam/inference/r1_native.py` is the executable
Python definition.  `r1_block_runner` accepts regular or comma-separated
irregular blocks and an optional second output after reset.  `r1_benchmark`
measures block size 64 and reports median and p95 timings as JSON.

`verify_r1_cpp_parity` runs regular block-64 inference, irregular blocks, and a
second pass after reset.  Its `fssr-r1-parity-v1` JSON report always includes
`python_cpp_parity`, `passed`, `max_abs_error`, per-comparison errors, and the
exact core/slow/residual/RF composition covered.

## Cost-only LSTM width sweep

`r1_benchmark --lstm-sweep SECONDS ITERATIONS` benchmarks the preregistered
widths `16,32,48,64,96` with the same block-64 clock harness.  A single width is
available through `--lstm-width WIDTH SECONDS ITERATIONS`.  The kernel is the
one-layer Wright topology (four dense LSTM gates, linear head, direct input
skip) with fixed deterministic non-trained weights.  Reports declare
`blind:true`, `trained_weights:false`, and `selection_uses_audio_or_esr:false`;
the probe is synthetic and no dataset or fidelity result is read.  These modes
measure architecture cost only and do not make a Python/C++ fidelity claim for
trained LSTM weights.
