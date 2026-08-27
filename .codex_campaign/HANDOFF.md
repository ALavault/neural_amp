# Handoff

## Current position

M0, M1, and M2 passed. The exact official A2 packed topology is documented, two minimal synthetic seeds converged, exports pass complete-file Python/C++ block and reset checks, and the host-specific Full/Lite CPU baselines are recorded. External evaluation remains locked. M3 is in progress.

## Resume action

Implement S0 first: causal FIR pre/post filters plus a finite, continuously differentiable learnable spline initialized near identity. Add numerical tests before introducing the slow GRU or residual TCN. Do not modify either NAM submodule.

## Absolute blockers

No absolute blocker for M3 synthetic implementation. Physical paired device archives are not yet downloaded or assigned (except EGFxSet INTERNAL_DEV), so M4 physical-device training still depends on later license/content verification.
