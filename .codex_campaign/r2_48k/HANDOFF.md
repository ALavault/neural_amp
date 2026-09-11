# FSSR-R2-48K handoff

`FSSR-R2-48K-v1` is terminal with verdict `INVALID` at the synthetic mechanism
stage. Do not run `r2-48k-screen` or any later stage, and do not relaunch the
mechanism matrix under this version.

The authoritative records are:

- `.codex_campaign/r2_48k/VERDICT.json`
- `.codex_campaign/r2_48k/MATURITY.json`
- `.codex_campaign/r2_48k/GATE_LEDGER.jsonl`
- `experiments/runs/r2_48k_mechanism_synthetic_analytic_matrix_seed0_v1/failure.json`

The run rendered all 216 synthetic conditions but produced no valid immutable
summary because exact zero periodicity error became negative infinity and the
strict JSON writer rejected it. The x2 gate was not evaluated; therefore this
is neither a synthetic AA pass nor a valid failure.

Any continuation requires a new campaign version frozen before measurements,
with a JSON representation for exact-zero ratios specified in advance. Preserve
the old partial result and failure record. Blackstar/UA tests and
`EXTERNAL_REPORT_ONLY` remain locked; no physical waveform was read by the
mechanism run.
