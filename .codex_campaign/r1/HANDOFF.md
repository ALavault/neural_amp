# FSSR-R1 handoff

R1 is terminal with `NO-GO-R1`. The steering documents are committed separately
at `ca10058`; the initial implementation, quarantine correction, native sweep,
and versioned lock amendment remain in the subsequent local commits.

The terminal evidence and adversarial assessment are consolidated in
`reports/R1_FINAL_AUDIT.md`; the R1 audit gate is complete.

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
