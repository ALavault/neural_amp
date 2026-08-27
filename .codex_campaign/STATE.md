# Campaign State

- Updated: 2026-08-27T12:05:15+02:00
- Maturity: M0 — Bootstrap
- Status: in progress
- Current task: lock the Python environment, validate the identity smoke test, and complete the public-data inventory.
- External results accessed: no
- External retest authorized: false

## Completed

- Inspected the initially empty workspace and host hardware.
- Initialized a local Git repository on `main`.
- Pinned official NAM trainer `v0.13.0` and NeuralAmpModelerCore `v0.5.4` as submodules.
- Created the M0 repository skeleton, campaign controls, minimal synthetic identity generator, ESR metric, and tests.

## M0 gate remaining

- Resolve and recreate the Python 3.12 environment.
- Complete the license-aware public dataset inventory.
- Run lint, tests, data audit, and the persisted identity round trip.
- Register the smoke artifact, finish `reports/M0_BOOTSTRAP.md`, and create the first local commit.

