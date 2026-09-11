# FSSR-R2 campaign state

- Campaign: `FSSR-R2-v1`
- Lineage: prospective and architecture-agnostic
- Status: repository preflight passed; no capture or scientific run observed
- Current gate: external synchronous capture and instrumentation pending
- R1: administratively `superseded_by_R2`; its runs and conclusions are unchanged
- Counted candidate: none (the frozen screen has not run)
- Exploratory candidate: `aa-fssr-xl` is implemented and validated only on
  synthetic/serialization paths; it is deliberately excluded from selection
  until a versioned protocol amendment admits it.
- Final verdict: none
- `EXTERNAL_REPORT_ONLY`: locked and unaccessed

The repository-side protocol is frozen before capture. Hardware capture, GPU
training, native timing, sealed-test evaluation, and MUSHRA evidence must be
recorded by their stage commands; absent evidence cannot be replaced by a
synthetic or manually written result.

The immutable repository preflight is `.codex_campaign/r2/PREFLIGHT.json`.
`r2-capture-audit` currently returns `PENDING_EXTERNAL` because
`datasets/manifests/r2_capture.json` is absent; it launched no scientific run
and read no capture or sealed-test samples.

The exploratory `aa-fssr-xl` branch is intentionally outside the frozen R2
matrix: it uses a cascade core with 65-tap FIRs and 129-knot Hermite splines,
versus the R1 17-tap/17-knot controls, while retaining the causal export and
parity path. Its passing numerical checks are engineering evidence only, not
an ESR/ASR or deployment claim.

Validation record (2026-08-27): `make lint`, `make test` (242 passed), and
`make r2-preflight` (75 passed) are green. The XL export/parity check passed at
the registered `2e-5` tolerance. `r2-capture-audit` remains
`PENDING_EXTERNAL` because `datasets/manifests/r2_capture.json` is absent, and
`r2-mechanism` is correctly rejected fail-closed (`capture=passed` is missing).

Clarification (2026-08-28): only 48 kHz physical archives are available;
derived upsampling is not a valid substitute for the 192 kHz hardware
reference. No Fractal FM9 capture is added as a proxy, so R2-v1 remains frozen
and blocked at capture rather than silently changing its device matrix or ASR
claim.
