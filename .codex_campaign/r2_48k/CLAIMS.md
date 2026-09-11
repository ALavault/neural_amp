# FSSR-R2-48K claims

| Claim | Status | Evidence |
| --- | --- | --- |
| The reused manifest describes 18 pairs prepared at 48 kHz. | verified metadata-only setup fact | `.codex_campaign/r2_48k/DATA_AUDIT.json` |
| R2-v1 remains frozen before capture with zero scientific runs. | repository fact | `.codex_campaign/r2/PROTOCOL_LOCK.yaml` |
| FM9 is not used as a proxy. | frozen design fact | `configs/r2_48k/protocol.yaml` |
| `aa-fssr-xl` is selection-eligible in R2-48K. | frozen design fact | `configs/models/r2_48k/aa_fssr_xl.yaml` |
| A candidate improves heldout physical ESR by at least 15%. | not evaluated | — |
| The AA mechanism reduces synthetic aliasing by the frozen thresholds. | not evaluated | — |
| A candidate reduces aliasing in the physical hardware above 24 kHz. | unsupported and forbidden | `configs/r2_48k/protocol.yaml` |
| A candidate passes native CPU, latency, and parity gates. | not evaluated | — |
| A candidate has a primary MUSHRA advantage above ten points. | not evaluated | — |
| The x2 synthetic mechanism passes its frozen gate. | not evaluated; evidence serialization invalid | `experiments/runs/r2_48k_mechanism_synthetic_analytic_matrix_seed0_v1/failure.json` |
| The terminal verdict is `INVALID`. | verified administrative verdict | `.codex_campaign/r2_48k/VERDICT.json`; `.codex_campaign/r2_48k/GATE_LEDGER.jsonl` |

No valid AA, ESR, listening, CPU, or state-of-the-art result was established.
The partial mechanism JSON is not evidence and must not be mined for selection.
