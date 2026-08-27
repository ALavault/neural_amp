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

