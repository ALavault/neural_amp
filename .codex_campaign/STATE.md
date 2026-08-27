# Campaign State

- Updated: 2026-08-27T15:07:55+02:00
- Maturity: M6 — negative-result audit and publication dossier
- Status: completed under pivot P4 — NO-GO
- Current task: none; regenerate validation or the safe archive with Make targets as needed.
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
- Completed four immutable v1 physical preflights; all paths were finite and block-consistent, and the runs exposed a checkpoint-selection defect before any counted training.
- Completed corrected v2 preflights for all four model paths; step-0 eligibility and block/reset behavior now pass.
- Completed and audited the first four counted Fulltone seed-0 runs; A2 leads this single condition, while all model paths satisfy the operational run checks.
- Completed all 24 counted M4 runs with no failed seed or missing metric. Preliminary medians favor A2 on both devices, so the M4 gate awaits cost audit and formal autopsy.
- Generated `reports/M4_CORE.md`, the machine-readable aggregate, the ordered autopsy, and corrected Python streaming benchmark. The fixed M4 gate failed.
- Completed the equal-parameter recovery; its stop rule failed on both devices. Selected pivot P4 and skipped the gated MATURATION phase.
- Froze the negative protocol record, generated local licensed listening examples, completed the final audit and paper dossier, and prepared the safe-archive manifest.

## M0 gate

M0 passed on 2026-08-27. Evidence is summarized in `reports/M0_BOOTSTRAP.md`.

## M1 gate

M1 passed on 2026-08-27. Evidence is summarized in `reports/M1_METRICS.md`.

## M2 gate

M2 passed on 2026-08-27. Evidence is summarized in `reports/M2_BASELINES.md`.

## M3 gate

M3 passed on 2026-08-27. Evidence is summarized in `reports/M3_MODEL.md`.

## M4 gate remaining

M4 completed as a failed gate on 2026-08-27. All 24 runs and the bounded recovery are retained. MATURATION was not entered.

## M6 gate

M6 completed as `NO-GO` on 2026-08-27. Full validation and archive verification are the final reproducibility checks; they do not change the scientific verdict.
