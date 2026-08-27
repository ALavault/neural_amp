.PHONY: bootstrap test lint data-audit smoke campaign-status fetch-egfxset-clean m1-audit m2-data m2-inspect m2-core-test m2-cpp-build m3-audit

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

m2-data:
	uv run python scripts/generate_m2_dataset.py

m2-inspect:
	uv run python scripts/inspect_nam_a2.py

m2-core-test:
	cmake -S third_party/NeuralAmpModelerCore -B build/nam_core -DCMAKE_BUILD_TYPE=Release -DNAM_ENABLE_A2_FAST=ON
	cmake --build build/nam_core --parallel 8
	cd third_party/NeuralAmpModelerCore && ../../build/nam_core/tools/run_tests

m2-cpp-build:
	cmake -S cpp -B build/fssr_cpp -DCMAKE_BUILD_TYPE=Release
	cmake --build build/fssr_cpp --parallel 8

m3-audit:
	uv run python scripts/summarize_m3.py
