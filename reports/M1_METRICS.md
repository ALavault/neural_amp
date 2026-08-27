# M1 Metrics and Synthetic Systems

Status: **completed — gate passed 2026-08-27**

## Scope and provenance

M1 validates the data-generation, multirate, alignment, and metric machinery before any model comparison. Final artifacts were regenerated from commit `a9af90d` with `dirty=false`:

- `experiments/summaries/m1_synthetic/manifest.json`;
- `experiments/summaries/m1_synthetic/decimation_response.png`;
- `experiments/summaries/m1_metric_validation/metrics.json`; and
- `experiments/summaries/m1_metric_validation/detection_matrix.png`.

No `EXTERNAL_REPORT_ONLY` result was accessed.

## Synthetic corpus

Nine deterministic excitations exercise eight systems: polynomial, `tanh`, asymmetric waveshaper, Wiener, Hammerstein, Wiener–Hammerstein, memory clipper, and slow sag. This produces 72 generated pairs at a 192 kHz master rate with derived 96 and 48 kHz references. Every stored summary is finite and includes a canonical float32 SHA-256.

A real DI diagnostic is also processed through all eight systems. It is EGFxSet `Clean/Neck/6-22.wav`, permanently assigned to `INTERNAL_DEV`, CC-BY-4.0, mono 48 kHz PCM24. Its verified SHA-256 is `7aa3c7a4ed1ebdbd647a9456c8c93422d972ba8d033a0b95e0fb36468a2da61a`. The raw file is not tracked.

References use two explicit x2 stages. Each stage applies a 255-tap Kaiser FIR (beta 8.6), cutoff 0.9 of the target Nyquist, removes the known 127-input-sample group delay, then strides by two. Boundaries use zero extension.

## Alignment and perturbation validation

The delay convention is positive when target lags reference. Controlled tests recover a 7-sample integer delay exactly, a 3.25-sample fractional delay within 0.12 samples, and retain the sign of polarity-inverted correlation.

| Perturbation | Intended diagnostic | Observed response |
| --- | --- | --- |
| Gain 0.8 | gain error | `-0.2000` |
| Integer delay 8 | delay estimator | `7.9997` samples |
| Fractional delay 0.35 | phase-slope delay | `0.3500` samples |
| Polarity inversion | correlation | `-1.0` |
| DC offset 0.05 | DC error | `0.05` |
| Low-pass | magnitude error | `0.1723` |
| Harmonic removal | complex harmonic error | `0.1115` |
| Inharmonic tone | inharmonic ratio | `0.00828` |
| Ringing | MR-STFT | `0.1805` |
| Phase-only change | phase / magnitude error | `0.1301` rad / `1.15e-6` |

The controlled 9 kHz `tanh` case measures parasite energy at `-19.56 dB` for naive 48 kHz processing and `-71.68 dB` for x2 local processing against the decimated 192 kHz reference: a 52.12 dB reduction in this synthetic case. This is metric validation, not evidence for H3 on hardware.

## Known limitations

- The causal envelope metric does not reliably detect short narrow-band ringing; MR-STFT does in the selected test.
- Windowed STFT magnitude is not exactly invariant at frame boundaries under a fractional circular phase ramp; the frozen tolerance is `1e-5`.
- “Aliasing” is used only for the controlled known-reference pair. Elsewhere the measure is called parasite or residual spectral energy.
- EGFxSet contains isolated notes and cannot alone establish generalization to musical passages.
- Fractional alignment is validated on controlled signals; nonlinear physical pairs require a separate M2/M4 audit.

## Validation and gate decision

`make lint`, `make data-audit`, `make m1-audit`, and the complete Pytest suite pass. The suite contains 41 tests. Seven M1 implementation or protocol failures remain visible in `.codex_campaign/FAILURES.md`; none was deleted or bypassed.

M1 passes: identity error is zero, perturbations are discriminated with documented responsibility boundaries, x2 processing reduces the expected controlled parasite metric, multirate references are finite and deterministic, and no external blindness rule was violated. Work may proceed to M2 reproduction of NAM A2 Full/Lite.
