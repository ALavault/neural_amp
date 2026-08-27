# FSSR-R1 campaign state

- Campaign: `FSSR-R1-v1`
- Lineage: prospective; the M0–M6 `NO-GO` remains immutable
- State: terminal audit after competence execution failure
- Diagnostic trajectories used: 1 / 19
- Confirmatory runs used: 0 / 76
- Current gate: competence failed; every later stage is locked
- Candidate: none
- Confirmatory hypothesis: none
- Verdict: `NO-GO-R1`
- `EXTERNAL_REPORT_ONLY`: locked and unaccessed

The canonical seed-0 run failed during its first scheduled validation because
cuDNN rejects recurrent sequences of 65,536 samples or more on the recorded
GPU/runtime; the configured validation chunk was 100,000. The run is immutable
and consumes one competence trajectory. It produced no test ESR, so seeds 1/2
are unauthorized and the required three-seed gate is now unreachable within
FSSR-R1. No factorial, horizon, cascade, lock, confirmation, or external run is
authorized.
