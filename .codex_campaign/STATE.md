# Campaign State

- Updated: 2026-08-27T13:05:50+02:00
- Maturity: M3 — FSSR-NAM implementation
- Status: in progress
- Current task: implement and numerically validate FSSR-NAM variants S0 through S4 in order, beginning with the structured causal FIR/spline core.
- External results accessed: no
- External retest authorized: false

## Completed

- Inspected the initially empty workspace and host hardware.
- Initialized a local Git repository on `main`.
- Pinned official NAM trainer `v0.13.0` and NeuralAmpModelerCore `v0.5.4` as submodules.
- Created the M0 repository skeleton, campaign controls, minimal synthetic identity generator, ESR metric, and tests.
- Recreated the Python 3.12 environment from locked dependencies and validated NAM imports plus a CUDA tensor operation.
- Audited public dataset metadata and licenses without downloading audio.
- Passed lint, eight M0 tests, the catalog audit, and the persisted float32 identity round trip.
- Created the first local commit `f51b511`.
- Completed the 192/96/48 kHz synthetic corpus, including eight prescribed systems and a licensed real-DI ingestion diagnostic.
- Validated alignment and all prescribed metric perturbations; generated inspected diagnostic figures.
- Demonstrated a controlled synthetic parasite reduction from `-19.56 dB` to `-71.68 dB` under x2 processing, without promoting it to a hardware claim.
- Inspected the exact official A2 packed topology, reproduced Lite and Full training on two seeds, and validated official exports.
- Validated complete-file Python/C++ parity, regular and irregular block processing, exact reset behavior, and a pinned-core CPU reference.

## M0 gate

M0 passed on 2026-08-27. Evidence is summarized in `reports/M0_BOOTSTRAP.md`.

## M1 gate

M1 passed on 2026-08-27. Evidence is summarized in `reports/M1_METRICS.md`.

## M2 gate

M2 passed on 2026-08-27. Evidence is summarized in `reports/M2_BASELINES.md`.

## M3 gate remaining

- Implement S0 structured causal FIR and smooth learnable spline.
- Add S1 causal slow state and S2 constrained fast residual, then compose S3.
- Add and delay-validate S4 local x2 antialiasing.
- Pass identity, tanh, slow-state, short-overfit, finite-gradient, causality, reset, block-parity, and export tests.
- Verify that the residual does not carry all output energy and that estimated cost remains compatible with A2.
