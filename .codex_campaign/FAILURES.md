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

## 2026-08-27 — F-M3-001 — S0 model exports were not import-sorted

The first S0 static check rejected the new model package's relative import order before numerical tests ran. Resolution: apply Ruff's prescribed lexical order and rerun lint, formatting, and all focused spline/FIR tests.

## 2026-08-27 — F-M3-002 — Slow-controller test concatenated the feature axis

The first S1–S3 focused suite failed because its test concatenated scalar-controller outputs shaped `(3, samples)` along the default feature axis instead of time. Resolution: concatenate along the last axis and detach a separate diagnostic scalar conversion; model code and expected values are unchanged.

## 2026-08-27 — F-M3-003 — M3 summary limitations exceeded line length

The first static check of the M3 aggregation script rejected two 94-character limitation strings before summary generation. Resolution: wrap the literals without changing their text or any gate threshold, then rerun focused numerical validation.

## 2026-08-27 — F-M3-004 — Short-segment overfit stopped above its target

The initial 200-step S0 short-segment test reached MSE `3.10e-6`, above its declared `2e-6` target, while all other S0 tests passed. Resolution: retain the threshold and optimizer, extend this focused convergence test to 300 steps, and remove an unrelated diagnostic tensor-conversion warning. No production training result changed.

## 2026-08-27 — F-M3-005 — First overfit extension remained above target

At 300 steps the same focused overfit reached MSE `2.35e-6`, still above the unchanged `2e-6` threshold. Resolution: use the actual 17-knot S0 configuration instead of the reduced 13-knot fixture and allow 400 steps. The threshold, data, optimizer, and production runs remain unchanged.

## 2026-08-27 — F-M4-001 — Resampling fixture included filter boundaries

The first physical-data unit test required a resampled affine pair to remain within `2e-3` through the first and last samples. Joint polyphase filtering preserved the relation in the signal interior, but zero extension produced a maximum boundary discrepancy of `6.1e-3` for the artificial DC offset. Resolution: retain the tolerance and test the interior after 64 boundary samples; keep a separate peak assertion to prove that preparation performs no gain normalization.

## 2026-08-27 — F-M4-002 — Initial M4 preparation script exceeded line length

The first static check rejected seven long provenance and limitation literals before data preparation ran. Resolution: wrap the expressions without changing their serialized text, configuration, or signal processing, then rerun the focused checks.

## 2026-08-27 — F-M4-003 — Physical-data test required configured formatting

After lint and tests passed, `ruff format --check` requested the configured compact form for one multiline assertion and stopped the command before data generation. Resolution: apply the repository formatter to that test and rerun the complete focused validation before preparation.

## 2026-08-27 — F-M4-004 — Repository-wide Ruff command entered pinned submodules

After all 67 project tests and the data audit passed, an ad hoc `ruff check .` traversed the pinned NAM and Eigen repositories and reported their upstream formatting. Those repositories are immutable campaign dependencies and must not be edited. Resolution: use the established `make lint` target, which scopes checks to `src`, `tests`, and `scripts`, and preserve the third-party pins unchanged.

## 2026-08-27 — F-M4-005 — B2 streaming fixture exceeded line length

The first B2 focused suite passed all numerical checks, then static analysis rejected one 95-character list literal in the block-equivalence test. Resolution: expand the list without changing the tested block schedule or model behavior and rerun the focused validation.

## 2026-08-27 — F-M4-006 — M4 common-path style check failed after numerical pass

All four M4 model factories and the official A2 block-overlap parity test passed, but static analysis rejected one tuple concatenation and one long budget assertion. Resolution: use tuple unpacking and a named `samples_seen` expression without changing the 3,276,800-sample budget, then rerun the focused validation.

## 2026-08-27 — F-M4-007 — Common-path files required configured formatting

After the corrected common-path tests and lint passed, the formatter requested compact forms for two expressions. Resolution: apply the configured formatter to the two new files and rerun all focused checks; no behavior or budget changed.

## 2026-08-27 — F-M4-008 — First M4 run-harness lint pass

Static analysis stopped the first run-harness check on two long function signatures before any run directory was created. Resolution: wrap the signatures and remove an unused draft regularizer helper, retaining the active explicit loss calculation, then rerun lint, formatting, and tests.

## 2026-08-27 — F-M4-009 — Run harness required configured formatting

The corrected run harness passed lint, after which the formatter requested five compact expression forms. Resolution: apply the configured formatter before any experimental run and rerun the static and numerical checks.

## 2026-08-27 — F-M4-010 — Initial model omitted from checkpoint selection

All four v1 physical preflights completed with finite outputs and block parity at or below `4.47e-8`. They exposed that S3 and S4 initial validation ESRs (`0.1265` and `0.1263`) were lower than every two-step checkpoint, but the harness initialized the best score to infinity and could only select steps 1 or 2. Resolution before any counted M4 run: save and register step 0 as the initial candidate, preserve all v1 runs unchanged, and require a new four-path v2 preflight from the corrected commit.

## 2026-08-27 — F-M4-011 — Matrix runner required configured formatting

The resumable matrix runner passed lint, then its first format check requested a one-line run identifier expression. Resolution: apply the configured formatter and rerun static validation before starting the runner.

## 2026-08-27 — F-M4-012 — First Python CPU benchmark lint pass

Static analysis rejected a long benchmark signature and an unread campaign-config payload before any benchmark run was created. Resolution: wrap the signature and retain the payload through its SHA256 in the resolved benchmark configuration, then rerun static checks.

## 2026-08-27 — F-M4-013 — CPU benchmark required configured formatting

The corrected benchmark passed lint, then the formatter requested compact generator expressions in two block-64 lookups. Resolution: apply the configured formatter and rerun validation before creating the immutable benchmark run.

## 2026-08-27 — F-M4-014 — Python benchmark omitted dynamic streaming state

Run `m4_python_cpu_seed0_v1` completed all timing measurements, but its `state_bytes` field counted registered buffers before streaming and therefore reported zero for B2 while omitting dynamic FIR, GRU, and TCN histories. Timing values are unaffected. Resolution: preserve v1, count materialized runtime state tensors by their explicit state fields after benchmarking, and rerun the complete benchmark as v2.

## 2026-08-27 — F-M4-015 — Runtime-state correction exceeded line length

The first static check of the runtime-state correction rejected one 91-character compound condition before v2 ran. Resolution: wrap the unchanged condition and rerun lint and formatting.

## 2026-08-27 — F-M4-016 — First M4 summary lint pass

Static analysis rejected long report and autopsy prose literals in the initial summary generator before aggregation ran. Resolution: wrap the source literals and generated Markdown at readable boundaries, replace an ambiguous dash, and rerun lint and formatting without changing any metric or gate calculation.

## 2026-08-27 — F-M4-017 — M4 summary required configured formatting

The corrected summary generator passed lint, then the formatter compacted one median expression and normalized quotes in two formatted report fields. Resolution: apply the configured formatter and rerun validation before generating the aggregate.

## 2026-08-27 — F-M4-018 — Fixed M4 smoke gate failed

All 24 counted runs completed, but A2 Full has lower median test ESR on both devices. Fulltone medians are `0.0639` for A2 versus `0.2112`/`0.2158` for S3/S4; Big Muff medians are `0.5969` versus `0.9670`/`0.9595`. Neither non-inferiority nor physical antialiasing evidence is present. Resolution: keep M4 in recovery, preserve the negative results, complete the ordered autopsy, and authorize only the preregistered width-31 equal-parameter diagnostic before selecting a negative-result pivot.

## 2026-08-27 — F-M4-019 — Recovery files required configured formatting

The recovery implementation passed lint, after which formatting checks requested compact forms in the runner and parameter-budget test before tests executed. Resolution: apply the configured formatter and rerun focused validation before creating any recovery run.

## 2026-08-27 — F-M4-020 — Recovery summary table exceeded line length

The first recovery-summary lint pass rejected two long formatted Markdown table rows before aggregation ran. Resolution: construct the rows from wrapped source expressions and rerun static validation without changing the values.

## 2026-08-27 — F-M4-021 — Recovery summary required configured formatting

The corrected recovery summary passed lint, then the formatter compacted one dictionary comprehension. Resolution: apply the configured formatter and rerun checks before aggregation.

## 2026-08-27 — F-M4-022 — Equal-parameter recovery stop rule failed

Width-31 S3 completed both authorized runs with 12,192 parameters and finite block-consistent outputs. It closes only `0.4%` of the seed-0 Fulltone ESR gap and `25.2%` of the Big Muff gap, below the fixed `50%` threshold required on both devices. Resolution: stop further model expansion, select pivot P4, skip the gated maturation matrix, and complete a transparent NO-GO audit and publication dossier.

## 2026-08-27 — F-M6-001 — Audio-example generator required formatting

The listening-example generator passed lint, then the formatter compacted two path/comprehension expressions before any audio was written. Resolution: apply the configured formatter and rerun static checks before generation.

## 2026-08-27 — F-M6-002 — Protocol checksum path was relative to the wrong directory

The first root-level `sha256sum -c` validation found no file because the checksum record named `PROTOCOL_LOCK.yaml` relative to the repository root instead of `.codex_campaign`. Resolution: store the repository-relative path and rerun verification; the lock content and expected digest are unchanged.

## 2026-08-27 — F-M6-003 — Final archive builder required formatting

The archive builder passed lint, then the formatter compacted one prohibited-path predicate before any archive was created. Resolution: apply the configured formatter and rerun validation before packaging.

## 2026-08-27 — F-M6-004 — Archive policy rejected synthetic M0 WAV fixtures

The first final archive build compressed the tracked tree, then rejected `experiments/summaries/m0_identity/input.wav` and `output.wav` because the verifier prohibited every WAV extension. These are deterministic synthetic identity fixtures, not raw or restricted audio. Resolution: retain path-based prohibitions for raw datasets, predictions, licensed listening examples, logs, and checkpoints while allowing tracked synthetic fixtures; rebuild from a new clean commit.

## 2026-08-27 — F-M6-005 — Corrected archive predicate required formatting

The corrected archive policy passed lint, then the formatter compacted its path-filter comprehension. Resolution: apply the configured formatter and rerun static checks before rebuilding.

## 2026-08-27 — F-M6-006 — Archive sidecar path was directory-relative

The successful safe archive wrote a conventional basename-only checksum sidecar, which requires changing into `artifacts/` before verification. Resolution: write the repository-relative archive path and record the exact source commit in the verification JSON, then rebuild from the corrected clean commit. Archive contents and exclusion policy are unchanged.
