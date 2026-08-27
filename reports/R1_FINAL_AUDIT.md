# FSSR-R1 final audit

## Verdict

`NO-GO-R1` is terminal. The historical M0–M6 `NO-GO` is unchanged. No H1 or
H2 result exists, so neither `GO-A` nor `GO-B` is available.

The first canonical competence trajectory,
`r1_competence_bigmuff_lstm64_wright_seed0_v1`, was reserved under the active
diagnostic lock and failed at its first scheduled validation. It durably records
one epoch and 308 optimizer updates, no selected checkpoint, and no test ESR.
It therefore consumes one of three competence trajectories. The two remaining
slots cannot produce both a replacement seed-0 continuation result and the
three seeds required by the final competence gate. No retry or later R1 stage
is authorized.

## Failure autopsy

The locked recurrent validation chunk is 100,000 samples. On the recorded RTX
PRO 4000 Blackwell, PyTorch 2.13.0, CUDA 13.0, and cuDNN 9.2 stack, a controlled
float32 probe with contiguous inputs gave:

| Sequence samples | Result |
| ---: | --- |
| 1,000; 4,093; 32,768; 65,535 | pass |
| 65,536; 100,000 | `CUDNN_STATUS_NOT_SUPPORTED` |

The cuDNN diagnostic text mentions non-contiguous input, but the probe tensors
were explicitly contiguous and the boundary is exactly the recurrent sequence
length. Loss numerics, TBPTT, data alignment, published-weight conversion, and
test sealing pass their focused checks; the observed trace occurs in validation
inference. The sealed test was never opened.

A prospective repair is straightforward—freeze a chunk such as 32,768 and run
the exact CUDA validation path before reserving a trajectory—but applying it to
R1 would require a replacement identifier/cap exception after observing a valid
failed trajectory. That repair belongs to a new preregistered lineage.

## Evidence and invariants

- The canonical failure and provenance are under
  `experiments/runs/r1_competence_bigmuff_lstm64_wright_seed0_v1/` and in the
  append-only global ledger.
- The earlier unauthorized, non-canonical implementation attempt is preserved
  under `experiments/quarantine/`; it is not a declared trajectory. Its original
  ledger/index path is deliberately left immutable and is corrected in
  `.codex_campaign/r1/FAILURES.md`.
- The active lock amendment binds implementation commit `3686ed9` without
  changing the scientific protocol digest
  `5c85e6aadace532462340828eef7493b2ad177fa12a6aa0ae86845b6888d7ae0`.
- The historical-byte audit still checks 715 exact M0–M6 files and two
  append-only registries.
- `INTERNAL_VALIDATION` outputs were never opened. `EXTERNAL_REPORT_ONLY`
  remains locked and unaccessed. Confirmation used 0/76 runs.
- The UA archive provenance remains recorded; CC-BY-NC-4.0 is not a scientific
  blocker for this internal non-commercial campaign.

## Implementation coverage at termination

Implemented and tested before the failure: Wright loss/model/conversion and
training controls; source-disjoint data preparation; strict IDs, caps, ledgers,
locks, and gate arithmetic; RF31/RF2047/cascade models; Python/C++ float32 block
parity; blind LSTM width cost sweep; paired hierarchical confirmation
statistics; and validation-only diagnostic training helpers.

The following latent paths were not eligible for scientific use and remain
incomplete or insufficiently fail-closed. They do not affect the terminal R1
verdict because competence blocks them:

1. Diagnostic model construction occurs before its seed is applied, so future
   factorial weights would not be seed-controlled without repair.
2. Later diagnostic authorization can merge editable maturity/gate documents
   without cryptographically binding the evaluator evidence.
3. The confirmatory lock validator checks field shapes but does not independently
   reconstruct H1/H2 seed-0 evidence, the candidate/predecessor relationship, or
   every frozen data/code/config digest.
4. The statistics CLI accepts arbitrary JSON and does not itself require the
   exact 76-run ledger, seed matrix, all five families, and immutable run digests.
5. Seven of the nine requested Make interfaces and the actual 76-run
   A2/LSTM/NablAFx/ablation/final executor were not completed before the stop.

These omissions must be resolved and preregistered for a future campaign; they
must not be retrofitted into FSSR-R1 or described as validated R1 capabilities.
