# FSSR-R1 validation appendix

Validation was run after the terminal `NO-GO-R1` decision without launching a
new trajectory.

## Passing checks

- Full Pytest suite: `154 passed` in 46.83 seconds.
- Ruff semantic checks: `ruff check src tests scripts` passed.
- R1 prepared-audio audit: 18 files verified; leakage check passed.
- Historical data catalog audit passed.
- Direct R1 protocol preflight passed: exact 3/8/4/4 and 76 counts, active
  protocol digest, 715 frozen-root files, two append-only registries, data
  presence, external freeze, and reference pins.
- Dry-run reservation for factorial is rejected by the failed competence gate.
- Dry-run confirmation is rejected by the pending confirmatory lock.

## Failing checks retained as evidence

- `make lint` fails only its format-check phase because the frozen
  `src/fssr_nam/campaign/r1.py` would reflow one existing call. Applying that
  cosmetic edit after the counted failure would drift from the active
  implementation lock, so it was not changed. Ruff rule checks still pass.
- `make r1-preflight` stops in its first `r1-data` preparation command. The
  committed `r1_physical.json` records configuration SHA-256
  `aa0ce48a…b218283`, while the locked configuration bytes hash to
  `756b145f…6d009`. Prepared audio itself passes all 18 checksum/metadata checks.
  The direct protocol preflight does not currently validate this binding, which
  is a latent audit gap. Neither manifest nor locked configuration was rewritten
  after the terminal run.

These failures do not alter the verdict. They are additional reasons to start a
new preregistered lineage instead of repairing or resuming FSSR-R1.
