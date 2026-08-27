# FSSR-R1 reopening audit

## Disposition

The earlier `NO-GO-R1` report remains immutable but is superseded by PI decision
R1-D-009. The canonical competence execution failed in cuDNN before a checkpoint
or sealed-test ESR existed. It is therefore an invalid infrastructure execution,
not evidence for or against Wright competence, H1, or H2.

Revision `FSSR-R1-v1-a2` openly changes post-observation capacity accounting. The
failed ledger row remains `failed`, while scientific validity becomes
`invalid_infrastructure` and capacity contribution becomes zero. Exactly one new
identifier replaces the logical seed-0 condition. The replacement still consumes
one of the original three competence slots; the total valid scientific design
remains 3/8/4/4 and 19 trajectories.

## Frozen repair

- replacement: `r1_competence_bigmuff_lstm64-retry1_wright_seed0_v1`;
- recurrent evaluation chunk: 32,768 samples;
- exact CUDA chunk/parity preflight before reservation;
- no change to model, initialization seed, data, loss, optimizer, scheduler,
  early stopping, checkpoint selection, test seal, ESR gates, or external lock;
- one replacement maximum; no further replacement authorized.

The prepared physical manifest configuration digest is also corrected to the
already committed configuration bytes. Prepared audio, splits and checksums are
unchanged, and the direct preflight now verifies this binding.

## Remaining decision

There is currently no R1 verdict. `GO-A` or `GO-B` remains available only if H1
or H2 passes the original confirmatory gates. Any other scientifically valid
terminal outcome will be `NO-GO-R1`.
