# FSSR-R2-48K-v2 claims

| Claim | Status | Evidence |
| --- | --- | --- |
| V1 remains terminal and immutable with verdict `INVALID`. | repository fact | `.codex_campaign/r2_48k/VERDICT.json` |
| V2 changes only evidence serialization. | frozen design fact | `configs/r2_48k_v2/protocol.yaml` |
| Extended-real evidence roundtrips through strict JSON. | supported | `.codex_campaign/r2_48k_v2/PREFLIGHT.json`; `tests/unit/test_json_evidence.py` |
| The v2 matrix completes under strict JSON. | supported | `experiments/summaries/r2_48k_v2/mechanism.json` |
| The v2 x2 synthetic mechanism passes. | not established; gate invalid | `experiments/summaries/r2_48k_v2/mechanism_invalid.json` |
| A candidate passes the physical ESR, CPU, latency, parity and MUSHRA gates. | not evaluated | — |
| V2 establishes physical hardware alias reduction or global SOTA. | unsupported and forbidden | `configs/r2_48k_v2/protocol.yaml` |
| Final v2 verdict. | `INVALID` | `.codex_campaign/r2_48k_v2/VERDICT.json` |
