# Campaign State

- Updated: 2026-08-27T12:39:19+02:00
- Maturity: M2 — NAM A2 reproduction
- Status: in progress
- Current task: inspect the pinned A2 architecture/configuration, reproduce minimal Full/Lite training and export, validate block causality, and establish stable CPU benchmarks.
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

## M0 gate

M0 passed on 2026-08-27. Evidence is summarized in `reports/M0_BOOTSTRAP.md`.

## M1 gate

M1 passed on 2026-08-27. Evidence is summarized in `reports/M1_METRICS.md`.

## M2 gate remaining

- Document the exact official A2 Full/Lite topology and training/export configs.
- Train and export minimal A2 models without undocumented source changes.
- Validate full-file, sample, regular-block, and irregular-block parity and reset behavior.
- Reproduce at least two seeds with plausible learning behavior.
- Establish stable Python and official C++ CPU benchmarks, recording any divergence.
