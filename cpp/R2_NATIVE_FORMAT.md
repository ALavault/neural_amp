# FSSR-R2 native format v1

`fssr-r2-native-v1` is the prospective float32 deployment format for both
candidate families. The top-level `family` discriminator is exactly `aa-fssr`
or `aa-nam`; loaders reject any other value and reject fields from the other
topology.

Every file declares `aa_mode`, `internal_sample_rate`, `dilation_scale`, and
`latency_samples`. The supported modes are `off`, `adaa1`,
`full_island_x2`, and `teacher_x4`, with rate factors 1, 1, 2, and 4. The two
oversampled modes contain one interpolation/decimation FIR pair and a
16-sample causal base-rate delay. ADAA uses the analytic cubic-Hermite
primitive and the registered `1e-4` divided-difference limit.

`aa-fssr` stores its mono/cascade FIR–Hermite core, GRU16 controller, and
16-channel RF2047 residual. `aa-nam` stores the pinned 23-layer, 8-channel A2
Full WaveNet with one independent Hermite activation per channel and layer.
Its convolution dilations and output-head FIR are scaled at ×2/×4 so that the
physical RF stays equal to 6347 base-rate samples. A native AA-NAM reset
prewarms the biased WaveNet with its declared zero-padding history; ADAA adds
one history sample per activation layer.

The `sizes` object is mandatory and reports trainable parameters, weight
bytes, persistent-state bytes, and scratch bytes. The C++ loader independently
validates topology dimensions and reports its actual state allocation in the
interleaved benchmark. Scientific parity is maximum absolute error `2e-5` or
less over regular blocks, irregular blocks, and reset repetition.
