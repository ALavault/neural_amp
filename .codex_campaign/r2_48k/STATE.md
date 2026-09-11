# FSSR-R2-48K campaign state

- Campaign: `FSSR-R2-48K-v1`
- Parent: `FSSR-R2-v1`, preserved and still blocked before capture
- Status: terminal instrumentation invalidity during the first scientific run
- Physical data: existing pairs prepared at 48 kHz; no physical 192 kHz reference
- Development: Fulltone and Big Muff, explicitly historical and previously observed
- Prospective primary devices: Blackstar and UA1176; outputs remain locked
- Candidate pool: `aa-nam`, `aa-fssr`, and newly admitted `aa-fssr-xl`
- Physical ASR claim: forbidden; ASR mechanism evidence is synthetic only
- FM9 proxy: forbidden
- Current gate: closed invalid before a valid mechanism gate result
- Final verdict: `INVALID`
- `EXTERNAL_REPORT_ONLY`: locked and unaccessed

This lineage replaces neither the protocol nor the conclusions of R2-v1. It
narrows the physical claim to 48 kHz fidelity, native cost, and listening, and
keeps the 192 kHz reference only for known synthetic fixtures. The internal
192 kHz teacher is a model-side regularizer; it is not a hardware reference and
cannot restore information absent from a 48 kHz target.

The primary ESR confidence interval and primary listening interval use only
Blackstar and UA1176. Fulltone and Big Muff remain useful for model selection
and descriptive replication, but are excluded from primary inferential
intervals because their outputs and test metrics were used historically.

No training, native benchmark, sealed-test evaluation, or listening session was
authorized or launched. Blackstar/UA outputs and `EXTERNAL_REPORT_ONLY` remain
locked.

Execution record (2026-08-28): preflight and the metadata-only audit passed. The
single registered mechanism run rendered six fixtures and 216 synthetic probe
conditions, but strict JSON serialization rejected a negative-infinite value
caused by an exactly zero periodicity residual. Consequently no valid
`mechanism.json` exists and the x2 gate was never evaluated. The partial result,
explicit failure record, global run-ledger entry, gate event, and terminal
verdict are preserved. Under the frozen fail-closed rule this is `INVALID`, not
a candidate failure and not `NO-GO-R2-48K`.

Post-verdict repository validation: `make lint` passes, all 270 tests pass, and
`make campaign-status` reports `maturity=terminal status=terminal_invalid` with
the internal-validation outputs still locked.
