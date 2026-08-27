# Claims Register

Claims remain `PROPOSED` until linked evidence satisfies the applicable gate. No model-quality claim exists at M0.

| ID | Date | Claim | Status | Evidence |
| --- | --- | --- | --- | --- |
| C-M0-001 | 2026-08-27 | The host provides one NVIDIA GPU with approximately 24 GiB VRAM. | VERIFIED | `environment/system-report.txt`; `nvidia-smi` capture summarized in `reports/M0_BOOTSTRAP.md` |
| C-M0-002 | 2026-08-27 | Official NAM trainer and Core sources are pinned to immutable commits. | VERIFIED | `.gitmodules`; `configs/models/baselines/nam_a2_full.yaml` |
| C-M0-003 | 2026-08-27 | The deterministic identity fixture survives a float32 WAV round trip with ESR zero. | VERIFIED | `experiments/summaries/m0_identity/metrics.json`; `make test` and `make smoke` output in `reports/M0_BOOTSTRAP.md` |
| C-M1-001 | 2026-08-27 | The configured synthetic generator deterministically produces 72 finite 192/96/48 kHz pairs across nine excitations and eight systems. | VERIFIED | `experiments/summaries/m1_synthetic/manifest.json`; `tests/integration/test_synthetic_corpus.py` |
| C-M1-002 | 2026-08-27 | The M1 metric suite passes all controlled perturbation responsibility checks. | VERIFIED | `experiments/summaries/m1_metric_validation/metrics.json`; `tests/numerical/test_metric_perturbations.py` |
| C-M1-003 | 2026-08-27 | In the controlled 9 kHz tanh diagnostic, x2 processing reduces known-reference parasite energy by 52.12 dB relative to naive 48 kHz processing. | VERIFIED_SYNTHETIC_ONLY | `experiments/summaries/m1_metric_validation/metrics.json` |
| C-M1-004 | 2026-08-27 | The M1 real-DI diagnostic is a verified mono 48 kHz PCM24 EGFxSet file licensed CC-BY-4.0 and assigned to INTERNAL_DEV. | VERIFIED | `datasets/manifests/catalog.yaml`; `datasets/manifests/EGFXSET_ATTRIBUTION.md` |
| C-M2-001 | 2026-08-27 | The pinned official A2 packed configuration contains 23-layer 3-channel Lite and 8-channel Full WaveNets with a 6,347-sample receptive field. | VERIFIED | `experiments/summaries/m2_a2_architecture/architecture.json`; pinned trainer source |
| C-M2-002 | 2026-08-27 | Minimal official A2 training converges on the synthetic tanh task for seeds 0 and 1 without modifying upstream sources. | VERIFIED_SYNTHETIC_ONLY | `experiments/runs/m2_a2_tanh_seed0_v3`; `experiments/runs/m2_a2_tanh_seed1_v1` |
| C-M2-003 | 2026-08-27 | A2 Lite and Full exports preserve complete-file causal output across regular and irregular C++ block schedules within `5e-7`, and reset exactly. | VERIFIED | `experiments/runs/m2_a2_cpu_seed0_v2/metrics.json` |
| C-M2-004 | 2026-08-27 | On the recorded host at block 64, A2 Full costs 2,862 ns/sample median and 3,521 ns/sample p95 using the official fast-path dispatcher. | VERIFIED_HOST_SPECIFIC | `experiments/runs/m2_a2_cpu_seed0_v2/metrics.json`; `official-bench-a2-fast.txt` |
| C-M3-001 | 2026-08-27 | S3 learns the synthetic tanh and slow-sag diagnostics to validation ESR below `1e-3` in one seed. | VERIFIED_SYNTHETIC_ONLY | `experiments/summaries/m3_validation/metrics.json` |
| C-M3-002 | 2026-08-27 | On the M3 diagnostics, S1 improves S0 by 18.0% for slow sag and S2 improves S0 by 32.0% for short nonlinear memory. | VERIFIED_SYNTHETIC_ONLY | `experiments/summaries/m3_validation/metrics.json` |
| C-M3-003 | 2026-08-27 | Trained FSSR variants reload and preserve causal irregular-block output within `4.77e-7`, with exact reset reproducibility. | VERIFIED_SYNTHETIC_ONLY | eight M3 run `metrics.json` files indexed by `experiments/summaries/m3_validation/metrics.json` |
| C-M3-004 | 2026-08-27 | The controlled S4 x2 diagnostic reduces known-reference parasite energy by 17.15 dB while its complex 9 kHz fundamental error remains `1.64e-7`. | VERIFIED_SYNTHETIC_ONLY | `experiments/summaries/m3_validation/metrics.json`; `tests/numerical/test_local_oversampling.py` |
| C-M4-001 | 2026-08-27 | The fixed physical smoke matrix completed all 24 B0/B2/S3/S4 runs across two devices and three seeds without a failed or missing result. | VERIFIED_INTERNAL_DEV | `experiments/summaries/m4_smoke/metrics.json`; `.codex_campaign/RUN_LEDGER.jsonl` |
| C-M4-002 | 2026-08-27 | In M4, A2 Full has lower median test ESR than S3 and S4 on both devices; no quality or non-inferiority smoke condition passes. | VERIFIED_INTERNAL_DEV_NEGATIVE | `experiments/summaries/m4_smoke/metrics.json`; `reports/M4_CORE.md` |
| C-M4-003 | 2026-08-27 | On the host at block 64, the current Python streaming paths cost 21,493 ns/sample for S3 and 24,371 ns/sample for S4; these are diagnostic and not comparable final C++ H2 evidence. | VERIFIED_HOST_SPECIFIC_DIAGNOSTIC | `experiments/runs/m4_python_cpu_seed0_v2/metrics.json` |
