# M0 Bootstrap Report

Status: **in progress**  
Started: 2026-08-27

## Initial state

The workspace contained no files and was not a Git repository. M0 therefore initializes the complete research scaffold rather than normalizing existing code.

## Host summary

- Ubuntu 26.04 LTS, Linux 7.0.0-30-generic.
- Two Intel Xeon E5-2630 v3 sockets; 16 physical and 32 logical CPU cores total.
- 121 GiB RAM.
- NVIDIA RTX PRO 4000 Blackwell with 24,467 MiB, driver 595.84, reported CUDA 13.2 capability.
- 765 GiB free in the workspace filesystem at inspection.

The full captured summary is in `environment/system-report.txt`. CPU benchmarking controls and CUDA execution remain to be validated.

## Official baseline pins

- Trainer: `https://github.com/sdatkinson/neural-amp-modeler.git`, `v0.13.0`, commit `f26112906de06ec6b796ad6d1982e29eed83144e`.
- Inference: `https://github.com/sdatkinson/NeuralAmpModelerCore.git`, `v0.5.4`, commit `1f42f88535884450104b8711d7595019afa0495b`.

Inspection of the pinned trainer identifies the packed A2 submodels as 3 channels (Lite) and 8 channels (Full), each using the release’s fixed 23-layer configuration. Reproduction and performance claims remain deferred to M2.

## Validation pending

- Locked environment recreation.
- Lint and Pytest suites.
- Dataset catalog audit.
- Identity float32 WAV generation/reload and zero-ESR check.
- CUDA smoke operation.
- First immutable local commit.

