# Decision Log

Entries are append-only. Later corrections must reference the superseded entry.

## 2026-08-27 — D-M0-001 — Python runtime

Use Python 3.12 managed by `uv`. The system Python 3.14 is not selected because compatibility across PyTorch, PyTorch Lightning, and NAM is less established.

## 2026-08-27 — D-M0-002 — Official NAM pins

Pin the official trainer at `sdatkinson/neural-amp-modeler` `v0.13.0` (`f26112906de06ec6b796ad6d1982e29eed83144e`) and the official inference Core at `sdatkinson/NeuralAmpModelerCore` `v0.5.4` (`1f42f88535884450104b8711d7595019afa0495b`). The trainer release contains the packed A2 configuration with 3-channel Lite and 8-channel Full submodels.

## 2026-08-27 — D-M0-003 — External blindness

Initialize all external evaluation access controls to false. Inventory metadata may be researched, but no `EXTERNAL_REPORT_ONLY` model outputs may be generated or inspected before formal authorization.

## 2026-08-27 — D-M0-004 — Repository license

Do not presume authority to grant an open-source license. The root `LICENSE` reserves rights provisionally; third-party and dataset licenses remain independent. Revisit before public release.

## 2026-08-27 — D-M0-005 — Initial physical-data candidates

Retain four license-compatible ToneTwist candidates for later local audit: Fulltone Full Drive 2, Blackstar HT1 Overdrive, Electro-Harmonix Big Muff, and UA 6176/1176LN. They cover the requested moderate saturation, high gain, fuzz, and slow dynamics behaviors. Their tiers remain `UNASSIGNED` until archives, source boundaries, and alignment are inspected. Open-Amp and NAM’s Google Drive excitation remain metadata-only because no explicit asset license was found.

## 2026-08-27 — D-M1-001 — Metric responsibility boundaries

Use MR-STFT, not the causal envelope metric, as the primary detector for short narrow-band ringing. Retain the envelope metric for level transitions, attack, release, and slow dynamics. Treat phase-only magnitude equality with a `1e-5` windowed-STFT tolerance because frame boundaries are not invariant to a fractional circular phase ramp. These assignments are frozen for the remainder of M1 and will be documented as metric limitations.

## 2026-08-27 — D-M1-002 — EGFxSet is internal development data

Assign EGFxSet permanently to `INTERNAL_DEV` before downloading any file. Use only its CC-BY-4.0 `Clean.zip` archive to validate ingestion of a real rights-compatible DI excerpt during M1. Its isolated-note structure is diagnostic and cannot establish general amplifier fidelity; it is excluded from future external report-only evaluation.

## 2026-08-27 — D-M1-003 — M1 gate passed

Advance to M2 because the 41-test suite, data audit, synthetic audit, and all controlled metric checks pass from commit `a9af90d` with clean provenance. Preserve the seven M1 failure entries and the metric limitations in the M1 report. Do not treat the controlled x2 result as validation of H3.

## 2026-08-27 — D-M2-001 — Warn-only determinism for official MRSTFT

Use Lightning `deterministic="warn"` for A2 training because PyTorch 2.13 has no deterministic CUDA backward for the reflection pad used by the official MRSTFT loss. Keep all explicit seeds and deterministic cuDNN settings. Do not change or remove the official loss to obtain strict mode.

## 2026-08-27 — D-M2-002 — Float32 block-equivalence tolerance

Before protocol freeze, set the C++ block/reset maximum-absolute tolerance to `5e-7`. The original `1e-7` threshold was below one float32 ulp near unit amplitude and rejected block-order differences no larger than `3.5763e-7`. Preserve the failed v1 benchmark as evidence of the change.

## 2026-08-27 — D-M2-003 — M2 gate passed

Advance to M3. The official packed A2 recipe converged for two seeds on the preregistered synthetic task, exports agree across Python and C++ paths, reset and block schedules pass, Core tests pass, and independent block-64 CPU loops agree within 1%. This establishes a toolchain and host-specific cost reference, not physical-device fidelity.

## 2026-08-27 — D-M3-001 — S0 spline and FIR foundation

Implement S0 with one 17-tap causal FIR before and after a 17-knot cubic Hermite spline. Share one learned slope at each knot to guarantee C1 continuity, use linear endpoint extrapolation, initialize exactly to identity, and expose analytic first derivatives and primitives for later ADAA work. Do not impose monotonicity; retain curvature regularization as an explicit loss term.

## 2026-08-27 — D-M3-002 — Initial slow and residual branches

Use a one-layer GRU with eight states updated after each completed 64-sample interval; zero-order-hold modulation is therefore strictly causal and independent of caller block boundaries. Use a four-layer 8-channel TCN with kernel 3 and dilations `[1,2,4,8]` for the fast residual. Its 31-sample receptive field is below 1 ms, its output layer starts at zero, and a sigmoid-controlled scale is capped at 0.5. Keep normalized residual-energy regularization mandatory in training configurations.

## 2026-08-27 — D-M3-003 — S4 causal local x2 filter

Oversample only the nonlinear residual `phi(x)-x` by two using 33-tap Kaiser-windowed sinc interpolation and decimation filters. Add it to an exactly delayed linear path, align slow gain and fast-residual inputs to the same declared 16-sample causal latency, and never compensate by looking ahead. On the controlled 9 kHz M1 case this implementation must reduce known-reference parasite energy by at least 3 dB while keeping complex fundamental error below `1e-5`.

## 2026-08-27 — D-M3-004 — M3 gate passed

Advance to M4 because all ten aggregate implementation checks pass: S3 learns two synthetic systems, S1 and S2 improve their discriminating cases, the residual remains below 0.11% of output energy, trained exports preserve causal block behavior, S4's declared delay is aligned, and estimated linear MACs remain below A2 Full. Treat every quality and antialiasing result as synthetic-only until multi-device evidence exists.

## 2026-08-27 — D-M4-001 — Physical smoke-test devices and alignment

Assign the checksum-verified Fulltone Full Drive 2 and Electro-Harmonix Big Muff archives permanently to `INTERNAL_DEV` before model-output inspection. Use Fulltone setting `V100_T050_O050_B000` for moderate saturation and Big Muff setting `S050_V100` for hard clipping. Fulltone marker peaks occupy the same sample grid, so apply no delay correction and remove two seconds at each boundary. For Big Muff, retain the published paired split without extra delay correction: windowed correlations are inconsistent under its strong nonlinearity, so selecting one inferred delay would be unjustified. Resample both members jointly from 44.1 to 48 kHz without level normalization. Keep each complete upstream source file in one split and disclose that Big Muff performer/session provenance is unavailable.

## 2026-08-27 — D-M4-002 — Cost-targeted recurrent baseline

Use a one-layer causal mono GRU with 62 hidden states for B2 Full. Its 12,153 trainable parameters differ from official A2 Full's 12,145 by 0.066%, and its estimated 11,780 multiply-accumulates per sample closely match A2 Full's 11,777. Treat those estimates only as sizing criteria; retain or reject cost comparability using the measured block CPU benchmark.

## 2026-08-27 — D-M4-003 — Common bounded smoke protocol

Give B0, B2, S3, and S4 the same 6,346-sample causal context, two 8,192-sample output regions per optimizer step, 200 Adam steps, float32 precision, and MSE plus the official A2 MR-STFT weight. This fixes 3,276,800 output samples seen per run. Select checkpoints only by ESR on the first 240,000 validation samples. Add the preregistered FSSR curvature and residual penalties without changing the shared data budget. Compare S4 to its target delayed by its declared 16-sample algorithmic latency. Require a two-step end-to-end preflight for all four paths before launching the 24 counted runs.

## 2026-08-27 — D-M4-004 — Step zero is an eligible checkpoint

Include the initialized model as checkpoint step 0 in validation-ESR selection. This prevents a fixed training budget from forcing selection of a degraded update and applies identically to every family. The four v1 preflights remain wiring diagnostics produced by the superseded selection implementation; require v2 preflights before counted runs.

## 2026-08-27 — D-M4-005 — Corrected preflight passed

Permit counted M4 training after all four v2 paths completed with finite predictions and alternate-block differences no larger than `4.47e-8`. B0 and B2 selected trained steps; S3 and S4 correctly retained step 0 after their two diagnostic updates degraded validation ESR. Run the four Fulltone seed-0 conditions first, audit them, and only then continue the other 20 matrix cells.

## 2026-08-27 — D-M4-006 — First counted wave is operationally valid

Continue the remaining M4 matrix after the four Fulltone seed-0 runs each consumed exactly 3,276,800 output samples, selected a non-final checkpoint, produced finite metrics, and passed alternate-block comparison within `1.2e-7`. Do not change the protocol in response to A2's initial Fulltone lead (`0.0668` test ESR versus approximately `0.2318` for S3/S4); one seed on one device is insufficient for a gate decision.

## 2026-08-27 — D-M4-007 — Complete matrix enters autopsy

Do not enter MATURATION after all 24 counted runs completed. Preliminary median test ESR favors A2 on both Fulltone (`0.0639` versus `0.2112`/`0.2158` for S3/S4) and Big Muff (`0.5969` versus `0.9670`/`0.9595`). The antialiased variant does not show a clear fidelity advantage over S3. Complete the required cost measurement and ordered autopsy before selecting a permitted pivot or declaring the campaign scientifically blocked.

## 2026-08-27 — D-M4-008 — Equal-parameter recovery stop rule

Run only S3 with 31 residual channels, seed 0, on the two M4 devices. Its 12,192 parameters are within 0.39% of A2 Full. Keep every other M4 setting fixed. Continue beyond this diagnostic only if it closes at least 50% of the seed-0 ESR gap between width-8 S3 and A2 on both devices; otherwise stop architecture expansion and select the negative-result/benchmark pivot.
