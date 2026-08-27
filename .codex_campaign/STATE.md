# Campaign State

- Updated: 2026-08-27T12:15:00+02:00
- Maturity: M1 — Metrics and synthetic systems
- Status: in progress
- Current task: implement the deterministic 192 kHz synthetic systems, controlled decimation, alignment, and metric perturbation suite.
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

## M0 gate

M0 passed on 2026-08-27. Evidence is summarized in `reports/M0_BOOTSTRAP.md`.

## M1 gate remaining

- Implement all eight prescribed synthetic nonlinear systems at 192 kHz.
- Validate controlled 192→96→48 kHz decimation.
- Implement integer/fractional alignment and the prescribed time, spectral, harmonic, transient, and parasite diagnostics.
- Demonstrate metric responses to all twelve controlled perturbations and produce `reports/M1_METRICS.md`.
