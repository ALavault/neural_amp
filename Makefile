.PHONY: bootstrap test lint data-audit smoke campaign-status fetch-egfxset-clean m1-audit

bootstrap:
	uv python install 3.12
	uv sync --extra baseline --extra dev --extra ml
	uv pip compile pyproject.toml --extra baseline --extra dev --extra ml --output-file environment/requirements-lock.txt

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

data-audit:
	uv run python scripts/data_audit.py datasets/manifests/catalog.yaml

smoke:
	uv run python scripts/smoke_identity.py

campaign-status:
	uv run python scripts/campaign_status.py

fetch-egfxset-clean:
	uv run python scripts/fetch_egfxset_clean.py

m1-audit:
	uv run python scripts/validate_synthetic_corpus.py
	uv run python scripts/validate_metrics.py
