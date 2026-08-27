# FSSR-R1 handoff

R1 is terminal with `NO-GO-R1`. The steering documents are committed separately
at `ca10058`; the initial implementation, quarantine correction, native sweep,
and versioned lock amendment remain in the subsequent local commits.

The terminal evidence and adversarial assessment are consolidated in
`reports/R1_FINAL_AUDIT.md`; post-verdict checks, including two retained
validation failures, are in `reports/R1_VALIDATION_APPENDIX.md`. The R1 audit
gate is complete.

Do not invoke `r1-competence`, `r1-factorial`, `r1-horizon`, `r1-cascade`,
`r1-lock`, or `r1-confirm`. The canonical seed-0 competence directory is a
counted immutable failure and may not be resumed, replaced, or excluded. The
failure audit identifies a backend sequence-length boundary: on this stack,
65,535 samples pass while 65,536 fail, and the locked validation chunk was
100,000.

A future R2 may preregister a recurrent evaluation chunk at or below 32,768 and
exercise that exact GPU path before reserving any trajectory. It must use new
campaign/run identifiers and must not reinterpret the R1 result.

Never use `EXTERNAL_REPORT_ONLY`, reuse a run identifier, overwrite a run
directory, or edit the M0–M6 campaign files.

## Superseding handoff — FSSR-R1-v1-a2

R1-D-009 supersedes the terminal instructions above. Preserve every old file and
ledger row, but treat the canonical cuDNN failure as scientifically invalid. Once
`DIAGNOSTIC_LOCK_AMENDMENT_2` is frozen, execute the exact CUDA preflight and the
single replacement ID `r1_competence_bigmuff_lstm64-retry1_wright_seed0_v1`.
It counts as seed 0 and as one scientific trajectory. Continue to seeds 1/2 only
if its sealed-test ESR is at most 0.15. No second replacement is authorized.
