# Paper Outline - Reproducible Negative Benchmark

## Working title

When a Structured Fast-Slow Residual Model Does Not Beat a WaveNet Amplifier
Baseline: A Reproducible Two-Device Study

## Contribution boundary

This is a negative-result and methods paper. It does not claim that FSSR-NAM is
superior, non-inferior, or physically antialiased. Its defensible contributions
are the controlled comparison, failure analysis, validated metric suite, and
released reproducibility machinery.

## Abstract plan

State the structured hypothesis, official A2 reproduction, two-device/three-seed
protocol, negative median ESR result, failed equal-parameter recovery, and the
practical lesson that synthetic mechanism validity does not guarantee physical
device fidelity.

## Sections

1. Introduction: structured priors, efficiency motivation, and publication bias.
2. Related work: recurrent black-box models, differentiable gray-box systems,
   neural aliasing metrics, and NAM.
3. Methods: S0-S4, causal slow controller, constrained residual, and local x2.
4. Validation: metric perturbations, synthetic systems, block parity, and A2 pin.
5. Physical protocol: rights, source-level splits, seeds, budgets, and metrics.
6. Results: M4 medians, per-seed variation, gain collapse, and CPU diagnostics.
7. Autopsy: alignment, context, state, residual energy, capacity, and recovery.
8. Limitations: two devices, post-campaign lock, no FSSR C++, and no listening test.
9. Conclusion: NO-GO result and concrete requirements for a future study.

## Required figures and tables

- Architecture diagram clearly marked as the tested hypothesis.
- Metric-validation perturbation grid.
- Per-device seed distributions and median ESR.
- Gain-error and correlation comparison.
- S3/S4 residual-energy diagnostic.
- Python block-cost plot plus separately labeled A2 C++ reference.
- Equal-parameter recovery table and ordered autopsy table.

## Submission readiness

The repository can support a workshop, reproducibility, or negative-results
submission. A main DAFx claim would require broader physical data, an FSSR C++
path, a preregistered confirmatory phase, and preferably perceptual evaluation.
