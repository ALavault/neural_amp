# FSSR-R1 claims

| Claim | Status | Evidence |
| --- | --- | --- |
| The UA archive matches the published file metadata and is readable. | verified setup fact | `datasets/manifests/r1_catalog.yaml` |
| R1 source groups are disjoint across train, validation, and sealed test. | verified setup fact | `datasets/splits/r1_physical.json` |
| The published Wright JSON is convertible without changing its tensors. | verified implementation fact | `experiments/summaries/r1_wright_reference/validation.json` |
| Wright LSTM-64 competence is reproduced. | not established; execution failed before test ESR | `experiments/runs/r1_competence_bigmuff_lstm64_wright_seed0_v1/status.json` |
| The competence failure is caused by the 100,000-sample cuDNN validation call, not a non-contiguous input. | reproduced operational finding | `experiments/summaries/r1_competence_failure_audit.json` |
| A loss is promoted by the factorial gate. | not evaluated; gate locked | `experiments/summaries/r1_competence_failure_audit.json` |
| RF2047 closes the preregistered gap. | not evaluated; gate locked | `experiments/summaries/r1_competence_failure_audit.json` |
| The cascade passes its synthetic and physical gates. | not evaluated; gate locked | `experiments/summaries/r1_competence_failure_audit.json` |
| H1 or H2 passes. | not evaluated; confirmation unauthorized | `experiments/summaries/r1_competence_failure_audit.json` |
| FSSR-R1 verdict is `NO-GO-R1`. | terminal protocol result | `.codex_campaign/r1/STATE.md` |

Mechanistic observations cannot change the verdict unless the frozen H1 or H2
gate passes.
