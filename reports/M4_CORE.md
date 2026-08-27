# M4 - Principal Smoke Test

## Outcome

The fixed 24-run matrix completed with three seeds, two physical devices,
four models, and no failed run. **M4 did not pass.** A2 Full has the lowest
median test ESR on both devices; neither FSSR variant is non-inferior, and
S4 does not show a clear advantage over S3.

| Device | Model | Median ESR | Relative vs B0 | Median gain error |
|---|---:|---:|---:|---:|
| Fulltone Full Drive 2 | B0 | 0.063883 | +0.0% | -0.063 |
| Fulltone Full Drive 2 | B2 | 0.267918 | +319.4% | -0.366 |
| Fulltone Full Drive 2 | S3 | 0.211185 | +230.6% | -0.233 |
| Fulltone Full Drive 2 | S4 | 0.215809 | +237.8% | -0.219 |
| Electro-Harmonix Big Muff | B0 | 0.596926 | +0.0% | -0.450 |
| Electro-Harmonix Big Muff | B2 | 0.930852 | +55.9% | -0.930 |
| Electro-Harmonix Big Muff | S3 | 0.966966 | +62.0% | -0.930 |
| Electro-Harmonix Big Muff | S4 | 0.959459 | +60.7% | -0.928 |

These are bounded `INTERNAL_DEV` smoke results, not final H1-H3 evidence.

## Gate evaluation

- Quality: failed. FSSR does not beat A2 on either device.
- Efficiency: failed on fidelity before cost can establish non-inferiority.
- Antialiasing: not established. S4 and S3 are nearly tied, and no physical
  high-rate reference identifies parasite energy.

## Cost diagnostic

At block 64 and one CPU thread, the Python streaming paths measure
26.61 µs/sample for B2, 21.49 for S3,
and 24.37 for S4. The pinned official A2 Core C++
reference is 2.86 µs/sample. This cross-engine comparison
is diagnostic only; FSSR C++ inference does not yet exist.

## Ordered autopsy

The machine-readable autopsy is
`experiments/summaries/m4_smoke/autopsy.json`. The strongest findings are
output-energy collapse on Big Muff, a tenfold parameter-budget mismatch versus
A2 Full, and a possibly marginal fast receptive field. Alignment uncertainty
cannot explain the Fulltone failure; clipping, non-finite state, leakage across
known source files, and block inconsistency were not observed.

## Bounded recovery decision

Before any maturation matrix, run one seed on both devices with S3 residual
width 31. This matches A2's parameter budget while leaving data, loss, context,
and training budget fixed. Stop and retain the negative-result/benchmark pivot
if it does not materially close the ESR gap.
