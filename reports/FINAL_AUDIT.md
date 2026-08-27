# Final Audit

## Classification

**NO-GO.** FSSR-NAM does not validate H1, H2, or H3 in the completed physical
smoke campaign. M0-M3 passed their infrastructure and synthetic gates; M4
failed, its bounded recovery failed, and MATURATION was correctly skipped.

## What works

- The environment, official NAM pins, manifests, source-file splits, run ledger,
  and immutable run layout are reproducible.
- A2 Full trains and exports through the pinned official code. Python/C++ output,
  irregular blocks, and reset behavior were validated.
- S0-S4 are causal and block-consistent. Synthetic tests verify smooth spline
  primitives, slow-state utility, fast-residual utility, and local x2 behavior.
- All 24 counted M4 runs completed with three seeds and no non-finite result.
- The equal-parameter recovery activated more residual capacity on Big Muff and
  improved seed-0 ESR, but not enough to meet its fixed stop rule.

## What fails

- Median physical test ESR favors A2 on both devices. Fulltone medians are
  `0.0639` for A2, `0.2112` for S3, and `0.2158` for S4. Big Muff medians are
  `0.5969`, `0.9670`, and `0.9595`, respectively.
- B2, S3, and S4 collapse toward low-energy outputs on Big Muff; their median
  gain error is approximately `-0.93`.
- S4 does not clearly improve physical fidelity over S3, and the campaign lacks
  a physical high-rate reference that could establish H3 parasite reduction.
- Current Python S3/S4 streaming is not real-time at block 64. No FSSR C++ path
  exists, so a final same-engine H2 cost comparison is unavailable.

## What remains uncertain

- Big Muff performer/session provenance and exact latency are not published.
- The current fast receptive field may be marginal for that pair.
- A different loss, longer context, or architecture might avoid output collapse,
  but exploring them after the failed recovery would violate the stop rule.
- Synthetic antialiasing gains may or may not transfer to physical hardware.

## Strongest results

The strongest positive contribution is methodological: exact official A2
reproduction, validated metric responsibilities, immutable provenance, and a
multi-seed negative comparison that resists favorable cherry-picking. The
strongest model finding is negative: parameter matching alone closes only 0.4%
of the Fulltone gap and 25.2% of the Big Muff gap.

## Threats to validity

- Only two physical devices were used before the gate stopped expansion.
- Big Muff split provenance is known only at the released-file level.
- M4 uses bounded 120/30/30-second excerpts and 200 optimizer steps.
- The formal protocol lock is post-campaign and cannot support confirmatory
  inference; committed pre-run configurations provide the effective provenance.
- CPU results mix official A2 C++ with diagnostic FSSR Python and must not be
  interpreted as final deployment efficiency evidence.
- No listening-study participants or perceptual results are claimed.

## Deviations and exclusions

- The external evaluation remained locked because no model passed M4.
- MATURATION, its 76 runs, mandatory physical ablations, final C++ FSSR engine,
  and external retest were skipped by gate, not silently omitted.
- M4 v1 preflights are excluded because they exposed a checkpoint-selection
  defect. Corrected v2 preflights are wiring evidence only.
- Recovery runs are excluded from the primary 24-run aggregate. No completed
  counted seed was excluded.

## Reproduction entry points

Run `make test`, `make data-audit`, `make m4-summary`, and
`make m4-recovery-summary`. Raw licensed audio and ignored checkpoints must be
available locally for training or audio regeneration. See the run ledger and
`experiments/summaries/` for exact evidence.
