# Failure Log

Entries are append-only and include failed experiments as well as infrastructure failures.

## 2026-08-27 — F-M0-001 — Direct baseline reference rejected

`make bootstrap` failed while building the editable root package because Hatchling rejects direct dependency references unless explicitly enabled. The failing dependency was the intentionally commit-pinned official NAM trainer. No partial experiment was run. Resolution: enable `tool.hatch.metadata.allow-direct-references` and rerun the complete bootstrap.

## 2026-08-27 — F-M0-002 — Initial format check failed

The first `make lint` passed all Ruff lint rules but failed `ruff format --check` because ten newly created Python files required canonical end-of-file formatting. Tests and the dataset audit passed independently. Resolution: apply the configured Ruff formatter and rerun lint and tests; no rule was disabled.

## 2026-08-27 — F-M0-003 — Whitespace check did not stop first commit

`git diff --cached --check` reported trailing blank lines and one Markdown trailing-space line, but command sequencing allowed the first commit to continue. No source behavior or experimental result was affected. Resolution: preserve the original commit, normalize tracked text mechanically, require a clean `git diff --check`, and commit the correction separately instead of rewriting history.

## 2026-08-27 — F-M1-001 — Initial multirate lint failure

All 22 targeted excitation/system/multirate tests passed, but Ruff rejected a `DecimationConfig()` call in a default argument and an ambiguous multiplication glyph in a docstring. Resolution: use an immutable module-level default and ASCII `x2`, then rerun lint and tests. Numerical code and thresholds were unchanged.

## 2026-08-27 — F-M1-002 — Parabolic fractional-delay bias

The first fractional-delay test estimated `3.043` samples for a controlled `3.25`-sample delay, outside the preregistered test tolerance of 0.12 samples. Integer delay and polarity tests passed. Resolution: replace parabolic peak interpolation with a magnitude-weighted cross-spectrum phase-slope estimate after integer-delay compensation; retain the original test target and tolerance.

## 2026-08-27 — F-M1-003 — Two perturbation checks misassigned

The first full perturbation audit failed `ringing_detected` and `phase_without_magnitude_detected`. For ringing, causal envelope error was only `0.000477`, while MR-STFT was `0.1805`; this demonstrates that the envelope metric is not a general ringing detector. For the exact FFT phase transform, windowed-STFT magnitude error was `1.15e-6`, narrowly above the initial `1e-6` tolerance. Resolution: assign ringing detection to MR-STFT (`>0.1`) and set a `1e-5` numerical tolerance for windowed magnitude while retaining the phase threshold. Perturbation signals and measured values were not changed.

## 2026-08-27 — F-M1-004 — Wrong harmonic basis in first aliasing figure

Visual inspection of the first passing detection matrix showed that the controlled 9 kHz aliasing case was evaluated with the generic 250 Hz harmonic basis, producing an uninterpretable large complex-harmonic value. The alias detection check itself used the separate known-reference parasite metric and was unaffected. Resolution: make each perturbation carry its own fundamental and permitted frequencies, assign 9 kHz to the controlled case, and regenerate all artifacts.

## 2026-08-27 — F-M1-005 — Corpus command continued after lint failure

The first complete corpus command reported two Ruff line-length errors in manifest limitation strings, but semicolon sequencing still ran the passing integration test and generated the 72-pair summary. No scientific value was inferred from that intermediate artifact. Resolution: wrap the strings, rerun formatting/lint/tests with fail-fast `&&`, and regenerate the artifact only from a validated tree.

## 2026-08-27 — F-M1-006 — Selected DI checksum transcription error

The first configuration entry for the selected EGFxSet WAV mistyped the final portion of its SHA-256. Direct `sha256sum` and the verified extraction script agreed on `7aa3c7a4ed1ebdbd647a9456c8c93422d972ba8d033a0b95e0fb36468a2da61a`. Resolution: correct the configuration before regenerating the M1 corpus manifest; the downloaded bytes were not changed.

## 2026-08-27 — F-M1-007 — Decimator implementation label was imprecise

The M1 configuration initially called the reference decimator `polyphase_fir`, while the implementation directly convolves with the same linear-phase FIR, removes its known delay, and strides by two. These operations are mathematically equivalent for the stored output but the implementation is not organized as polyphase branches. Resolution: rename the configured method to `linear_phase_fir_then_stride` before the final M1 artifact generation.

## 2026-08-27 — F-M2-001 — Core tests launched from the wrong directory

The first invocation of `build/nam_core/tools/run_tests` aborted because the upstream test executable resolves `example_models/` relative to its working directory. Resolution: rerun the unchanged binary from `third_party/NeuralAmpModelerCore`; the complete upstream suite printed `Success!`. The root Make target now encodes the required working directory.

## 2026-08-27 — F-M2-002 — Packed A2 inspector used an absent attribute

The first architecture inspection assumed that `PackedWaveNet` exposed a `submodels` collection. The pinned v0.13.0 API instead exposes `num_submodels` and `extract_submodel(index)`. Resolution: use the public extraction method and rerun the inspection; no model or upstream source was modified.

## 2026-08-27 — F-M2-003 — M2 runner test could not import scripts

The first focused runner test failed during collection because `scripts/` lacked an `__init__.py` marker. Resolution: add the marker so configuration construction can be tested directly, then rerun the unchanged assertions. No training was started.

## 2026-08-27 — F-M2-004 — Package marker did not enter the editable import path

The attempted `scripts/__init__.py` remedy did not change the Hatch editable package import path, so the same focused test still failed during collection. Resolution: move the reusable configuration adapter into the packaged `fssr_nam.training` module and import it from both the command and test. The ineffective marker was removed; no training was started.

## 2026-08-27 — F-M2-005 — Extracted adapter test exceeded line length

The first lint pass after moving the adapter found one 92-character fixture path in the test. Resolution: split the path construction across components and rerun the full focused validation; no lint rule or behavior changed.

## 2026-08-27 — F-M2-006 — Strict CUDA determinism unsupported by official loss

Run `m2_a2_tanh_seed0_v1` failed during its first backward pass because PyTorch 2.13 reports no deterministic CUDA implementation for `reflection_pad1d_backward`, reached through the official A2 MRSTFT loss. The failed run and partial export remain registered. Resolution: preregister `deterministic_mode: warn_only`, retain all explicit seeds and deterministic cuDNN settings, record this limitation in each environment, and relaunch as a new run identifier. Architecture, loss, data, and optimizer are unchanged.

## 2026-08-27 — F-M2-007 — Failed-run environment snapshot preceded seed setup

The immutable environment snapshot in failed run `m2_a2_tanh_seed0_v1` was written before strict deterministic algorithms were enabled, even though the subsequent runtime error proves strict mode was active during training. Resolution: preserve the run unchanged and move seed/determinism setup before environment capture for all later runs. The failed run has no reported scientific metric.

## 2026-08-27 — F-M2-008 — Result index used CRLF

The first run-index row used the CSV writer's default CRLF terminator, which failed `git diff --check` in this LF-normalized repository. Resolution: preserve the row values, normalize only its terminator, and configure subsequent writes with `lineterminator="\\n"`.

## 2026-08-27 — F-M2-009 — Lightning overrode PyTorch warn-only determinism

Run `m2_a2_tanh_seed0_v2` failed on the same MRSTFT reflection-pad backward because the adapter still passed `Trainer(deterministic=True)`, causing Lightning to restore strict mode after PyTorch had been configured for warnings. Resolution: map the preregistered `warn_only` mode to Lightning's native `deterministic="warn"` value and relaunch under a new identifier. No architecture, loss, data, optimizer, or seed changed.

## 2026-08-27 — F-M2-010 — CPU audit had an unused import

The first static check of the new CPU audit rejected an unused `re` import before any benchmark was launched. Resolution: remove the import and rerun lint, formatting, compilation, and tests without disabling the rule.

## 2026-08-27 — F-M2-011 — Initial block-parity tolerance was sub-ulp

Run `m2_a2_cpu_seed0_v1` failed the preregistered `1e-7` C++ block-equivalence threshold. Direct inspection found maximum absolute differences of `3.5763e-7` for Lite irregular blocks, `2.9802e-7` for Full block size 1, and `2.3842e-7` for Full irregular blocks; regular 16/64/128 outputs and reset outputs were bit-identical. These discrepancies are about three float32 ulps near unit amplitude, not an algorithmic delay or state loss. Resolution before protocol freeze: relax the explicit float32 block/reset tolerance to `5e-7`, preserve v1 as failed, and rerun the complete audit as v2.

## 2026-08-27 — F-M2-012 — Enriched architecture inspector needed formatting

The first M2 closure command stopped at `ruff format --check` after the inspector gained operation and state-buffer derivations. Resolution: apply the configured formatter and rerun lint, all tests, data audit, architecture regeneration, campaign status, and whitespace checks.
