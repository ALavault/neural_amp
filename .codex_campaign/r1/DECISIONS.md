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
