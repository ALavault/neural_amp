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
