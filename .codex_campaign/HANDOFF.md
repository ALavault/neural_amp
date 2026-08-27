# Handoff

## Current position

M0 through M3 passed. M4 data, the cost-targeted B2 GRU, and the common four-model training path are implemented. The fixed full-run budget is 3,276,800 output samples, and preflight uses two steps without counting toward the 24-run matrix. External evaluation remains locked.

## Resume action

Run `make m4-preflight`. Preserve and diagnose any failed immutable run; launch the counted matrix only after B0, B2, S3, and S4 all complete the physical preflight.

## Absolute blockers

No absolute blocker. Public ToneTwist archives are locally available under the recorded CC-BY-NC-4.0 terms. Their noncommercial restriction must remain visible in any checkpoint or archive redistribution decision.
