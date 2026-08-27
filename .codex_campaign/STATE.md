# Campaign State

- Updated: 2026-08-27T13:47:27+02:00
- Maturity: M4 — principal smoke test
- Status: in progress
- Current task: execute the four-path M4 preflight, audit failures, then launch the 24-run B0/B2/S3/S4 matrix only if every path completes.
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
- Implemented S0 through S4 and passed the synthetic implementation gate with trained export/block checks, slow/residual ablations, and controlled local-x2 validation.
- Verified the downloaded ToneTwist archives against official checksums and licenses, permanently assigned them to `INTERNAL_DEV`, and selected Fulltone Full Drive 2 plus Big Muff for M4.
- Prepared six bounded 48 kHz pairs without level normalization: source-disjoint Fulltone files and the published Big Muff train/validation/test files. All are finite and unclipped.
- Implemented the 12,153-parameter B2 GRU and a common M4 training/inference path with official A2 overlap-block parity below `2e-6`.

## M0 gate

M0 passed on 2026-08-27. Evidence is summarized in `reports/M0_BOOTSTRAP.md`.

## M1 gate

M1 passed on 2026-08-27. Evidence is summarized in `reports/M1_METRICS.md`.

## M2 gate

M2 passed on 2026-08-27. Evidence is summarized in `reports/M2_BASELINES.md`.

## M3 gate

M3 passed on 2026-08-27. Evidence is summarized in `reports/M3_MODEL.md`.

## M4 gate remaining

- Implement a cost-matched recurrent B2 baseline.
- Validate one end-to-end physical run per model before launching the matrix.
- Run B0, B2, S3, and S4 for seeds 0, 1, and 2 on both devices.
- Measure fidelity, diagnostic spectra, residual energy, training time, and inference cost for every valid run.
- Apply the preregistered quality, efficiency, or antialiasing smoke condition before entering MATURATION.
