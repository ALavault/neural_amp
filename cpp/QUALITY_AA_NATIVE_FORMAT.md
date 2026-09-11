# QUALITY-AA native format v1

`fssr-quality-aa-native-v1` extends the R2 float32 payload only for the
prospective `FSSR-QUALITY-AA-v2` backend selection. Oversampled x2/x4 payloads
declare an integer `latency_samples` from 1 through 48; the FIR length must be
`factor * latency_samples + 1`, and `resampling.linear_delay_samples` must equal
the top-level latency. All R2 topology, state, reset, block and size contracts
remain unchanged. Historical `fssr-r2-native-v1` payloads still require exactly
16 samples.
