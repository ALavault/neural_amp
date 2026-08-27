# Failure Log

Entries are append-only and include failed experiments as well as infrastructure failures.

## 2026-08-27 — F-M0-001 — Direct baseline reference rejected

`make bootstrap` failed while building the editable root package because Hatchling rejects direct dependency references unless explicitly enabled. The failing dependency was the intentionally commit-pinned official NAM trainer. No partial experiment was run. Resolution: enable `tool.hatch.metadata.allow-direct-references` and rerun the complete bootstrap.

## 2026-08-27 — F-M0-002 — Initial format check failed

The first `make lint` passed all Ruff lint rules but failed `ruff format --check` because ten newly created Python files required canonical end-of-file formatting. Tests and the dataset audit passed independently. Resolution: apply the configured Ruff formatter and rerun lint and tests; no rule was disabled.
