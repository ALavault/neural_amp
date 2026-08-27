# FSSR-R1 claims

| Claim | Status | Evidence |
| --- | --- | --- |
| The UA archive matches the published file metadata and is readable. | verified setup fact | `datasets/manifests/r1_catalog.yaml` |
| R1 source groups are disjoint across train, validation, and sealed test. | verified setup fact | `datasets/splits/r1_physical.json` |
| The published Wright JSON is convertible without changing its tensors. | implementation test pending final validation | `third_party/Automated-GuitarAmpModelling/Results/muff-muff2/model_best.json` |
| Wright LSTM-64 competence is reproduced. | untested | no run |
| A loss is promoted by the factorial gate. | untested | no run |
| RF2047 closes the preregistered gap. | untested | no run |
| The cascade passes its synthetic and physical gates. | untested | no run |
| H1 or H2 passes. | untested | no confirmation |

Mechanistic observations cannot change the verdict unless the frozen H1 or H2
gate passes.
