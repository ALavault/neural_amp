# Risk Register

| ID | Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- | --- |
| R-001 | CUDA/PyTorch wheel mismatch on the Blackwell GPU | Medium | High | Validate a CUDA tensor operation before training; record versions and fall back only with a documented environment revision. |
| R-002 | Public datasets lack compatible licenses or aligned physical pairs | High | High | Keep tier assignments provisional, use synthetic validation, and record absolute data blockers without fabricating coverage. |
| R-003 | Dual-socket Xeon timing variance and turbo effects destabilize CPU benchmarks | Medium | High | Pin thread/core where possible, warm up, repeat, report median/p95, and record frequency policy. |
| R-004 | Full campaign exceeds practical compute time | Medium | High | Enforce gates, smoke tests, successive halving, and the 24-configuration search cap. |
| R-005 | Residual network bypasses the structured core | Medium | High | Penalize and report residual energy; include a no-core comparable-budget ablation. |
| R-006 | Spectral metrics mislabel harmless harmonics as aliasing | Medium | High | Validate controlled synthetic perturbations and use qualified terminology. |
| R-007 | Run artifacts exhaust disk space | Low | High | Check space before matrices, index artifacts, and avoid duplicate checkpoints. |
