# Claims Register

Claims remain `PROPOSED` until linked evidence satisfies the applicable gate. No model-quality claim exists at M0.

| ID | Date | Claim | Status | Evidence |
| --- | --- | --- | --- | --- |
| C-M0-001 | 2026-08-27 | The host provides one NVIDIA GPU with approximately 24 GiB VRAM. | VERIFIED | `environment/system-report.txt`; `nvidia-smi` capture summarized in `reports/M0_BOOTSTRAP.md` |
| C-M0-002 | 2026-08-27 | Official NAM trainer and Core sources are pinned to immutable commits. | VERIFIED | `.gitmodules`; `configs/models/baselines/nam_a2_full.yaml` |
| C-M0-003 | 2026-08-27 | The deterministic identity fixture survives a float32 WAV round trip with ESR zero. | VERIFIED | `experiments/summaries/m0_identity/metrics.json`; `make test` and `make smoke` output in `reports/M0_BOOTSTRAP.md` |
