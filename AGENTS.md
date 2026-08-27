# Repository Guidelines

## Scientific Scope

This repository implements the FSSR-NAM amplifier-modeling campaign. Treat it as a scientific record: never invent missing measurements, hide failed runs, alter frozen protocols silently, or use `EXTERNAL_REPORT_ONLY` results for model selection. Follow `CODEX_EXECUTE_A_LA_LETTRE.md` and advance only through the declared maturity gates.

## Project Structure

Python code lives in `src/fssr_nam/`, grouped by data, DSP, models, metrics, training, inference, benchmarking, statistics, and reporting. Native inference code belongs in `cpp/`. Tests are split into `tests/unit/`, `integration/`, `regression/`, and `numerical/`. Configuration is under `configs/`; immutable run artifacts belong in `experiments/runs/<run_id>/`. Campaign decisions and evidence indexes are maintained in `.codex_campaign/`. Raw audio must not be committed.

## Development Commands

- `make bootstrap` creates the locked Python environment.
- `make test` runs the complete test suite.
- `make lint` checks formatting and static style with Ruff.
- `make data-audit` validates dataset catalog metadata without downloading data.
- `make smoke` generates and verifies the deterministic identity fixture.
- `make campaign-status` prints the current maturity and freeze state.

Run targeted tests while iterating, for example `uv run pytest tests/unit/test_synthetic.py -q`, then run `make test` before committing.

## Style and Tests

Use Python 3.12, four-space indentation, type annotations on public interfaces, `snake_case` for functions/files, and `PascalCase` for classes. Ruff is authoritative. Tests use Pytest and should be named `test_<behavior>.py`. Every numerical component needs finite-value, shape, causality, reset, and block-parity coverage where applicable. Use deterministic seeds and explicit tolerances.

## Commits and Reviews

Use concise imperative subjects such as `Validate synthetic identity metric`. Keep commits scoped to one scientific or infrastructure step. Pull requests must state the hypothesis or maintenance goal, commands run, affected configurations, new run IDs, and any protocol deviation. Link generated tables/figures instead of transcribing results manually.

## Provenance and Safety

Do not overwrite run directories. Register every training or benchmark in `.codex_campaign/RUN_LEDGER.jsonl`, including failures. Preserve source-file boundaries in data splits. Do not commit secrets, private datasets, caches, or redistributed artifacts whose licenses are unclear.
