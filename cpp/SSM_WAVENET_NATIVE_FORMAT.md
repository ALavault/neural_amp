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
