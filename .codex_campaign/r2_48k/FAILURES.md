# FSSR-R2-48K failure log

## Predecessor constraint, not a scientific candidate failure

`FSSR-R2-v1` cannot execute its preregistered physical-aliasing campaign
because no synchronous 192 kHz hardware dataset exists. Upsampling 48 kHz
archives would not repair that evidential gap, and an FM9 capture would change
the measured system. No R2-v1 scientific run failed and no candidate was
rejected.

## Known R2-48K validity limits

- Fulltone and Big Muff are historical development, not fresh holdouts.
- Blackstar has one test source and UA1176 has two; bootstrap intervals cannot
  imply broad source-population coverage.
- Big Muff provides file-level split provenance without proof of independent
  performances or sessions.
- Synthetic ASR evidence cannot establish physical hardware-alias reduction.

These are disclosed scope limits. An instrumentation or protocol breach remains
`INVALID`; a valid failure of any simultaneous decision gate is
`NO-GO-R2-48K`.

## Repository preflight implementation correction — 2026-08-28

The first repository-only preflight invocation stopped before writing evidence
because its report requested a nonexistent derived `parameters` field from the
native payload. No audio, training, benchmark, or scientific result was read or
launched. The report now counts parameters directly from the instantiated model;
the failed invocation is not a scientific trajectory.

## Mechanism launcher serialization correction — 2026-08-28

The first formal launcher invocation stopped while serializing the resolved
protocol because YAML had materialized `frozen_at` as a datetime object. The
qualification function had not been called: zero fixtures and zero probe
conditions were rendered, no mechanism summary or global ledger event was
written, and no physical waveform was read. The incomplete directory is
preserved under
`experiments/quarantine/r2_48k_mechanism_launcher_datetime_pre_measurement_v1`.
The launcher now validates a JSON-safe resolved protocol before reserving the
immutable scientific run ID. This is a pre-measurement instrumentation
correction, not a failed AA trajectory or a scientific retry.

## Terminal mechanism instrumentation invalidity — 2026-08-28

The single registered mechanism matrix rendered all six fixtures and 216 probe
conditions, then failed while serializing its immutable result because an exact
zero periodicity residual was represented numerically as negative infinity,
which the strict JSON writer rejects. A partial `result.json` and the explicit
`failure.json` are preserved in
`experiments/runs/r2_48k_mechanism_synthetic_analytic_matrix_seed0_v1`.
No valid mechanism summary exists, the x2 gate was not evaluated, no physical
audio was read, and no result-dependent retry is authorized. Under the frozen
rule that instrumentation failure remains invalid, the terminal campaign
verdict is `INVALID`, not `NO-GO-R2-48K`.
