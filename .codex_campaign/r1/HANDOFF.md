# FSSR-R1 handoff

R1 is in setup. The steering documents are committed separately at `ca10058`.
The UA archive is present locally and verified, but no diagnostic trajectory
has been launched. Run `make r1-preflight` after implementation validation.
Then invoke `make r1-competence`; later stage commands must reject execution
until their predecessor gate is recorded as passed.

Never use `EXTERNAL_REPORT_ONLY`, reuse a run identifier, overwrite a run
directory, or edit the M0–M6 campaign files.
