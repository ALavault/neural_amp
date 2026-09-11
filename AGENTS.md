# FSSR-NAM

Follow `CODEX_EXECUTE_A_LA_LETTRE.md` and its maturity gates. Never invent
measurements, hide failures, silently alter frozen protocols or use
`EXTERNAL_REPORT_ONLY` results for selection.

## Layout and commands

Python: `src/fssr_nam/`; native inference: `cpp/`; tests:
`tests/{unit,integration,regression,numerical}/`; configuration: `configs/`;
immutable runs: `experiments/runs/<run_id>/`; decisions/evidence: `.codex_campaign/`.

- `make bootstrap`: locked environment; `make lint`: Ruff.
- `make data-audit`: catalog validation without downloads.
- `make smoke`: deterministic identity fixture; `make campaign-status`: maturity/freeze.
- Iterate with targeted tests, e.g. `uv run pytest tests/unit/test_synthetic.py -q`.
  Run `make test` (full suite) before committing code changes. Documentation-only
  changes need `git diff --check`, not the suite.

## Implementation and evidence

Use Python 3.12, four-space indentation, public type annotations, `snake_case`
functions/files and `PascalCase` classes; follow Ruff. Pytest files:
`test_<behavior>.py`. Numerical coverage requires applicable finite-value, shape,
causality, reset and block-parity checks, deterministic seeds and explicit tolerances.

Never overwrite runs. Register every training/benchmark, including failures, in
`.codex_campaign/RUN_LEDGER.jsonl`. Preserve source-file boundaries in splits.
Do not commit raw audio, private datasets, secrets, caches or artifacts with
unclear redistribution rights.

## Product lane

`demo/`, `src/fssr_nam/product/` and `scripts/product_*.py` form an engineering
lane outside the campaign gates (see `.codex_campaign/DECISIONS.md`,
`D-PRODUCT-001`). Iteration, retry and in-place fixes are allowed there; runs go
to `demo/RUNS.jsonl`, not `RUN_LEDGER.jsonl`, and results carry no scientific
claim. Product code must not read Blackstar, UA 1176 or `EXTERNAL_REPORT_ONLY`
outputs, which stay sealed for the scientific lane.

Keep imperative commits scoped to one scientific/infrastructure step.
PRs state hypothesis/maintenance goal, commands, affected configurations, new run
IDs and protocol deviations; link generated tables/figures instead of copying results.
