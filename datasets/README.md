# Dataset policy

Raw audio is stored outside Git. Only manifests, split definitions, checksums required by the protocol, and redistribution-safe derived summaries belong here.

Every source is assigned exactly one tier: `SYNTHETIC`, `INTERNAL_DEV`, `INTERNAL_VALIDATION`, or `EXTERNAL_REPORT_ONLY`. Windows from one source recording must remain in the same split. External report-only outputs cannot be inspected until the frozen protocol explicitly authorizes the retest.

`manifests/catalog.yaml` is the pre-download inventory. A resource may be downloaded only after its exact version and license permit the intended experimental use. Unknown fields remain `null`; they must never be inferred.

## Storage

Since 2026-10-07, `datasets/raw` and `datasets/amp_capture` are symlinks to `/bigdata2/lavaulta/neural_amp_data/` (copied with `rsync -aH`, checked with `rsync --checksum`). Regenerable files were removed that day (public archives whose extracted copies remain, generated audio); the list and how to get them back are in `/bigdata2/lavaulta/neural_amp_data/README.md` and `DELETED_20261007.txt`.
