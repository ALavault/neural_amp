# FSSR-R2-48K-v2 campaign state

- Campaign: `FSSR-R2-48K-v2`
- Parent: terminal `FSSR-R2-48K-v1`, preserved with verdict `INVALID`
- Amendment: evidence serialization only
- Scientific thresholds, fixtures, modes, probes, renderer and metric formula: unchanged
- New transport: canonical tagged extended-real values in strict JSON
- Physical data: unchanged 48 kHz archive; no physical 192 kHz reference
- Blackstar/UA1176 outputs: locked
- `EXTERNAL_REPORT_ONLY`: locked and unaccessed
- FM9 proxy: forbidden
- Scientific runs launched: 1 (the single authorized synthetic mechanism matrix)
- Transport preflight: passed
- Metadata-only data audit: passed
- Mechanism measurement: complete, 24 aggregate rows and 415 tagged extended-real diagnostics
- Mechanism gate: invalid before x2 evaluation because an anti-silence guard failed
- Current gate: terminal; screening and every downstream stage remain locked
- Final verdict: `INVALID`

The v1 partial mechanism file is not imported, repaired, or used for selection.
The v2 matrix read zero physical audio samples and did not access Blackstar,
UA1176, `EXTERNAL_REPORT_ONLY`, or an FM9 proxy. The strict-JSON correction
worked, but the unchanged scientific guard prevented a valid x2 verdict.
