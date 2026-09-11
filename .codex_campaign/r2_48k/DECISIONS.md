# FSSR-R2-48K decision log

Entries are append-only. Corrections must name the superseded entry.

## 2026-08-28 — R2-48K-D-001 — New version, no retrospective amendment

Create `FSSR-R2-48K-v1` instead of changing frozen `FSSR-R2-v1`. Preserve the
old capture gate, runs, evidence, and lack of verdict. Make `r2_48k` the active
administrative lineage.

## 2026-08-28 — R2-48K-D-002 — Honest 48 kHz claim boundary

Use the existing physical pairs at their frozen prepared rate of 48 kHz. The
Fulltone source is 48 kHz; the other devices were jointly resampled from
44.1 kHz by the R1 preparation chain. Perform no additional resampling. Forbid a physical
hardware-aliasing claim because no above-Nyquist hardware reference exists.
Permit ASR selection only on known synthetic fixtures evaluated against a
synthetic 192 kHz reference. Derived upsampling is never hardware ground truth.

## 2026-08-28 — R2-48K-D-003 — No FM9 proxy

Do not add a Fractal FM9 or another digital modeller as a substitute device.
Such a capture would measure that modeller and its internal algorithms, not the
analogue devices or missing high-rate target required by R2-v1.

## 2026-08-28 — R2-48K-D-004 — Evidence roles

Classify Fulltone and Big Muff as historical development because earlier M4
work used their outputs and test metrics. Keep Blackstar and UA1176 as the two
prospective primary devices. Exclude development devices from the primary ESR
and MUSHRA confidence intervals while retaining the original three-of-four win
requirement with both prospective devices mandatory.

## 2026-08-28 — R2-48K-D-005 — Ambitious candidate admission

Admit `aa-fssr-xl` alongside AA-NAM and AA-FSSR. Impose no a-priori parameter
cap: validation fidelity ranks candidates and the exact native block-64 cost
gate decides deployment. The old R2-v1 exclusion of XL remains unchanged.

## 2026-08-28 — R2-48K-D-006 — Narrow GO label

Reserve `GO-R2-48K` for the scoped 48 kHz result. It is not `GO-R2`, not a
physical ASR verdict, and not an unrestricted global state-of-the-art claim.

## 2026-08-28 — R2-48K-D-007 — Pre-observation mechanism execution binding

Before examining any mechanism result, bind the already frozen fixtures, modes,
reference rate, probe grid, and thresholds to the executable recipe in
`MECHANISM_IMPLEMENTATION_LOCK.yaml`. Scale fixture delays and FIR taps by zero
insertion inside x2/x4 so their physical horizons remain fixed; generate the
synthetic reference directly at 192 kHz; aggregate the exact 3x3 grid by median
ASR and maximum fundamental error. This resolves orchestration details without
changing a candidate, fixture, gate, threshold, or claim boundary. Physical
audio remains unread and cannot be used as the high-rate reference.

## 2026-08-28 — R2-48K-D-008 — Terminal INVALID at mechanism instrumentation

Record `INVALID` after the one formal mechanism matrix completed its numeric
rendering but could not serialize immutable JSON evidence: exact periodicity
produced negative infinity and the strict writer rejected it. Do not evaluate
the partial file, reconstruct a favorable summary, or retry under v1. The x2
mechanism gate has no valid result, so screening remains locked. This verdict
does not reject an AA candidate and says nothing about physical aliasing.
