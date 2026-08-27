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
