# FSSR-R1 failures

## 2026-08-27 — quarantined competence implementation attempt

- Original identifier: `r1_competence_bigmuff_wright_lstm64_wright_seed0_v1`
- Timing: unauthorized launch after the initial diagnostic lock, but before any valid
  counted R1 trajectory; it requires a versioned diagnostic-lock amendment.
- Disposition: invalid protocol attempt, stopped after epoch 24 and excluded from the
  diagnostic trajectory count.
- Causes: non-canonical identifier, direct ledger path instead of `R1Executor`,
  premature loading of sealed test audio, and a forbidden `--resume` interface.
- Evidence: `experiments/quarantine/r1_competence_bigmuff_wright_lstm64_wright_seed0_v1/`.
- No test ESR was computed. The legacy runner appended an immutable failed entry and
  index row using the invalid identifier before it exited; their original result path
  now resolves through this quarantine record. The canonical seed 0 trajectory remains
  unattempted.

All valid R1 failures and gate-stopped trajectories must be appended here and to
the global run ledger; they may not be deleted or relabelled as successful.

## 2026-08-27 — canonical competence seed 0 infrastructure failure

- Run: `r1_competence_bigmuff_lstm64_wright_seed0_v1`.
- Disposition: immutable failed counted launch; no validation or test ESR exists.
- Failure point: first validation, after epoch 1 and 308 optimizer updates.
- Cause: cuDNN rejects deterministic LSTM inference chunks of 65,536–100,000
  samples on the RTX PRO 4000 Blackwell; 32,768 and 16,384 samples reproduce
  successfully with identical block-stream semantics.
- Scientific information observed: none beyond the training loss at epoch 1.
- Required action: bind any replacement attempt prospectively to an exact new run
  identifier and lock amendment; never reuse or rewrite this directory or ledger row.
