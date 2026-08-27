# FSSR-R1 decision log

Entries are append-only. Corrections must name the superseded entry.

## 2026-08-27 — R1-D-001 — Prospective lineage

Create FSSR-R1 as a new prospective lineage. Do not amend the M0–M6 verdict,
protocol lock, run directories, summaries, or evidence documents. Only H1 or H2
may yield `GO-A` or `GO-B`; every other terminal outcome is `NO-GO-R1`.

## 2026-08-27 — R1-D-002 — Data roles

Use Fulltone and Big Muff for `INTERNAL_DEV`. Within R1, withhold Blackstar
`G050_V100` and UA 1176LN `A100_R100_I040_O070_R100` as
`INTERNAL_VALIDATION`. Exclude Marshall from the primary confirmation because
the available archive lacks a fixed source-disjoint split appropriate to the
one-model-per-device comparison. Keep `EXTERNAL_REPORT_ONLY` locked.

## 2026-08-27 — R1-D-003 — UA archive correction

Zenodo file metadata, the published MD5, ZIP integrity, and all audio headers
were checked before use. The archive is PCM16 at 44.1 kHz. Its directory uses
`RAll` while filenames use `R100`; R1 records both spellings and interprets the
setting as all ratio buttons, without changing the old M0 inventory.

## 2026-08-27 — R1-D-004 — Bounded architecture search

Limit diagnostic training to competence (3), loss×budget (8), horizon (4), and
cascade (4). A failed branch releases no budget to a new idea. TFiLM, ADAA,
local ×4 oversampling, and a new slow controller are outside R1.

## 2026-08-27 — R1-D-005 — Wright weight provenance boundary

The pinned Wright training script defines the R1 competence recipe, including
`0.75 ESRPre + 0.25 DC`, but the published `muff-muff2/model_best.json` stores
only architecture metadata and tensors. It does not encode loss, sample rate,
normalization, scheduler, or training history. R1 therefore tests conversion
of those published tensors separately and trains competence seeds from scratch
under the locked recipe; it does not claim that the JSON itself proves which
historical loss produced the weights.

## 2026-08-27 — R1-D-006 — Prospective correction after invalid launch

Quarantine the non-canonical competence attempt without deleting its failed global
ledger row. Exclude it from diagnostic capacity because it is outside the frozen
matrix, and require every counted competence seed to use the canonical identifier,
fresh `R1Executor` reservation, sealed test boundary, and uninterrupted execution.
This correction changes no scientific threshold, dataset, loss, seed, or model.

## 2026-08-27 — R1-D-007 — Technical retry policy trigger

The canonical seed 0 run failed before validation because the preregistered runner
used a cuDNN-unsupported 100,000-sample inference chunk. A controlled probe reproduced
the failure at 100,000 and 65,536 samples and succeeded at 32,768 and 16,384. Preserve
the failed run. Before any retry, freeze a single exact replacement identifier and
the 32,768-sample implementation fix; do not alter data, initialization, optimizer,
loss, checkpoint selection, gates, seeds, or external locks.

## 2026-08-27 — R1-D-008 — Supersede retry and stop R1

This entry supersedes R1-D-007 after the preregistered capacity and gate logic
were audited together. The canonical seed-0 trajectory was reserved by
`R1Executor`, executed 308 optimizer updates in its first durably recorded
epoch, and failed at the first scheduled validation before checkpoint or test
evaluation. Unlike R1-D-006, it is a valid counted trajectory. R1-D-004 states
that a failed branch releases no budget; the remaining two competence slots
cannot supply both a replacement seed-0 result and the required three final
seeds.

A retry, `v2`/replacement identifier, fourth competence slot, or
post-observation exclusion would therefore change the frozen matrix. Seeds 1/2
and every later stage remain unauthorized, and the terminal verdict is
`NO-GO-R1`. The failure is operational rather than a measured rejection of
Wright LSTM-64; its prospective repair belongs to a newly preregistered lineage.

## 2026-08-27 — R1-D-009 — PI override: infrastructure-invalid replacement

This entry supersedes the disposition in R1-D-008 without deleting it. The PI
rules that a cuDNN sequence-length limitation is not scientific evidence against
Wright competence, H1, or H2. Reclassify the canonical seed-0 execution as
`invalid_infrastructure`: its global ledger status remains `failed`, but it no
longer consumes scientific diagnostic capacity.

This is an acknowledged post-observation protocol deviation. Revision
`FSSR-R1-v1-a2` changes capacity accounting and authorizes exactly one replacement,
`r1_competence_bigmuff_lstm64-retry1_wright_seed0_v1`. It changes no model,
weights initialization, seed, data, split, loss, optimizer, checkpoint rule,
metric, threshold, or external-data lock. Evaluation chunks are fixed at 32,768
and the exact CUDA path must pass before reservation. The replacement consumes
one competence trajectory; seeds 1/2 retain their original conditional gate.
No further replacement is authorized.
