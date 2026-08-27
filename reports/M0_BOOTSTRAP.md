# M0 Bootstrap Report

Status: **completed — gate passed 2026-08-27**
Started: 2026-08-27

## Initial state

The workspace contained no files and was not a Git repository. M0 therefore initializes the complete research scaffold rather than normalizing existing code.

## Host summary

- Ubuntu 26.04 LTS, Linux 7.0.0-30-generic.
- Two Intel Xeon E5-2630 v3 sockets; 16 physical and 32 logical CPU cores total.
- 121 GiB RAM.
- NVIDIA RTX PRO 4000 Blackwell with 24,467 MiB, driver 595.84, reported CUDA 13.2 capability.
- 765 GiB free in the workspace filesystem at inspection.

The full captured summary is in `environment/system-report.txt`. A PyTorch tensor operation completed on the GPU using PyTorch 2.13.0+cu130 and cuDNN 92000. Final CPU timing controls remain an M2/M5 concern.

## Official baseline pins

- Trainer: `https://github.com/sdatkinson/neural-amp-modeler.git`, `v0.13.0`, commit `f26112906de06ec6b796ad6d1982e29eed83144e`.
- Inference: `https://github.com/sdatkinson/NeuralAmpModelerCore.git`, `v0.5.4`, commit `1f42f88535884450104b8711d7595019afa0495b`.

Inspection of the pinned trainer identifies the packed A2 submodels as 3 channels (Lite) and 8 channels (Full), each using the release’s fixed 23-layer configuration. Reproduction and performance claims remain deferred to M2.

## Environment and data inventory

`make bootstrap` recreated Python 3.12.13 and installed the commit-pinned NAM package from the resolved lock. Public metadata was audited without downloading audio. Individual Zenodo records establish compatible research licenses for several physical candidates; records without an explicit asset license remain blocked. All final tier assignments remain pending local archive inspection.

## Validation evidence

| Check | Result |
| --- | --- |
| `make lint` | Passed; 13 Python files canonical under Ruff |
| `make test` | Passed; 8 tests |
| `make data-audit` | Passed; required catalog fields and tiers valid |
| `make smoke` | Passed; 12,000 float32 samples at 48 kHz, ESR `0.0` after WAV round trip |
| CUDA smoke | Passed; device tensor operation returned the expected value |
| Ledger behavior | Passed; incomplete and duplicate run entries rejected |
| External freeze | Closed; `external_retest_authorized=false` |

The first local commit is `f51b511` (`Bootstrap reproducible FSSR-NAM campaign`). Three M0 infrastructure failures are retained in `.codex_campaign/FAILURES.md`; each was corrected without deleting a test or hiding the original event.

## Gate decision

M0 passes: the environment recreates, tests pass, a synthetic signal is generated and reloaded, official NAM sources are immutably pinned, and the provenance registry is tested. Work may proceed to M1; no claim about A2 or FSSR-NAM fidelity is made.
