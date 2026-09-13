.PHONY: bootstrap test lint data-audit smoke campaign-status fetch-egfxset-clean m1-audit m2-data m2-inspect m2-core-test m2-cpp-build m3-audit r1-data r1-preflight r1-competence r2-preflight r2-capture-audit r2-mechanism r2-screen r2-teacher r2-distill r2-lock r2-confirm r2-benchmark r2-listen r2-audit r2-48k-preflight r2-48k-data-audit r2-48k-mechanism r2-48k-screen r2-48k-teacher r2-48k-distill r2-48k-lock r2-48k-confirm r2-48k-benchmark r2-48k-listen r2-48k-audit r2-48k-v2-preflight r2-48k-v2-data-audit r2-48k-v2-mechanism quality-aa-preflight quality-aa-mechanism quality-aa-native quality-aa-audit arch-preflight arch-native-skeleton arch-training-feasibility arch-mechanism arch-v2-preflight arch-v2-competence arch-v2-compare arch-v2-audit arch-v3-preflight arch-v3-training-feasibility arch-v3-round-1 demo demo-plugin demo-report demo-capture-selftest sota-bench

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

m4-data:
	uv run python scripts/prepare_m4_data.py

m4-preflight:
	uv run python scripts/run_m4_preflight.py

m4-matrix:
	uv run python scripts/run_m4_matrix.py

m4-benchmark:
	uv run python scripts/benchmark_m4_python.py --run-id m4_python_cpu_seed0_v2

m4-summary:
	uv run python scripts/summarize_m4.py

m4-recovery:
	uv run python scripts/run_m4_recovery.py

m4-recovery-summary:
	uv run python scripts/summarize_m4_recovery.py

audio-examples:
	uv run python scripts/generate_m4_audio_examples.py

final-archive:
	uv run python scripts/build_final_archive.py --output artifacts/fssr_nam_final_audit_2026-08-27.tar.zst

smoke:
	uv run python scripts/smoke_identity.py

campaign-status:
	uv run python scripts/campaign_status.py

amp-sota-preflight:
	uv run python scripts/run_sota_preflight.py

amp-sota-mechanism:
	uv run python scripts/run_sota_mechanism.py

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

r1-data:
	uv run python scripts/prepare_r1_data.py
	uv run python scripts/prepare_r1_data.py --audit-only
	uv run python scripts/prepare_r1_wright_data.py

r1-preflight: r1-data
	uv run pytest tests/unit/test_wright_compatibility.py tests/unit/test_wright_training.py -q
	uv run python scripts/validate_r1_wright.py
	uv run python scripts/run_r1_competence.py --preflight
	uv run python scripts/run_r1_preflight.py

r1-competence:
	uv run python scripts/run_r1_competence_gate.py

r2-preflight:
	uv run pytest tests/unit/test_r2_*.py tests/numerical/test_r2_aa_nam.py tests/numerical/test_r2_antialias.py tests/numerical/test_r2_asr.py tests/numerical/test_r2_fixtures.py tests/numerical/test_r2_native.py tests/numerical/test_r2_resampling.py -q
	uv run python scripts/run_r2_preflight.py

r2-capture-audit:
	uv run python scripts/audit_r2_capture.py

r2-mechanism:
	uv run python scripts/run_r2_stage.py mechanism

r2-screen:
	uv run python scripts/run_r2_stage.py screen

r2-teacher:
	uv run python scripts/run_r2_stage.py teacher

r2-distill:
	uv run python scripts/run_r2_stage.py distill

r2-lock:
	uv run python scripts/run_r2_stage.py lock

r2-confirm:
	uv run python scripts/run_r2_stage.py confirm

r2-benchmark:
	cmake -S cpp -B build/fssr_cpp -DCMAKE_BUILD_TYPE=Release
	cmake --build build/fssr_cpp --target r2_block_runner r2_benchmark --parallel 8
	uv run python scripts/run_r2_stage.py benchmark

r2-listen:
	uv run python scripts/run_r2_stage.py listen

r2-audit:
	uv run python scripts/run_r2_stage.py audit

r2-48k-preflight:
	uv run pytest tests/unit/test_r2_48k_*.py tests/numerical/test_r2_mechanism.py -q
	uv run python scripts/run_r2_48k_preflight.py

r2-48k-data-audit:
	uv run python scripts/audit_r2_48k_data.py

r2-48k-mechanism:
	uv run python scripts/run_r2_48k_mechanism.py
	uv run python scripts/run_r2_48k_stage.py mechanism

r2-48k-screen:
	uv run python scripts/run_r2_48k_stage.py screen

r2-48k-teacher:
	uv run python scripts/run_r2_48k_stage.py teacher

r2-48k-distill:
	uv run python scripts/run_r2_48k_stage.py distill

r2-48k-lock:
	uv run python scripts/run_r2_48k_stage.py lock

r2-48k-confirm:
	uv run python scripts/run_r2_48k_stage.py confirm

r2-48k-benchmark:
	cmake -S cpp -B build/fssr_cpp -DCMAKE_BUILD_TYPE=Release
	cmake --build build/fssr_cpp --target r2_block_runner r2_benchmark --parallel 8
	uv run python scripts/run_r2_48k_stage.py benchmark

r2-48k-listen:
	uv run python scripts/run_r2_48k_stage.py listen

r2-48k-audit:
	uv run python scripts/run_r2_48k_stage.py audit

r2-48k-v2-preflight:
	uv run pytest tests/unit/test_json_evidence.py tests/unit/test_r2_48k_v2.py tests/numerical/test_r2_mechanism.py -q
	uv run python scripts/run_r2_48k_v2_preflight.py

r2-48k-v2-data-audit:
	uv run python scripts/audit_r2_48k_v2_data.py

r2-48k-v2-mechanism:
	uv run python scripts/run_r2_48k_v2_mechanism.py
	uv run python scripts/run_r2_48k_v2_stage.py mechanism

quality-aa-preflight:
	uv run pytest tests/unit/test_quality_aa_protocol.py tests/numerical/test_quality_aliasing.py tests/numerical/test_quality_aa_mechanism.py -q
	uv run python scripts/run_quality_aa_preflight.py

quality-aa-mechanism:
	uv run pytest tests/unit/test_quality_aa_gates.py tests/numerical/test_quality_aliasing.py tests/numerical/test_quality_aa_mechanism.py -q
	uv run python scripts/run_quality_aa_mechanism.py

quality-aa-native:
	uv run pytest tests/numerical/test_r2_native.py tests/numerical/test_r2_cpp_parity.py -q
	uv run python scripts/run_quality_aa_native.py

quality-aa-audit:
	uv run pytest tests/unit/test_quality_aa_protocol.py tests/unit/test_quality_aa_gates.py tests/numerical/test_quality_aliasing.py tests/numerical/test_quality_aa_mechanism.py tests/numerical/test_r2_native.py tests/numerical/test_r2_cpp_parity.py -q
	uv run python scripts/run_quality_aa_audit.py

arch-preflight:
	uv run pytest tests/unit/test_amp_quality_arch_protocol.py tests/unit/test_nablafx_loss.py tests/numerical/test_amp_quality_arch_models.py tests/numerical/test_sota_comparators.py -q
	uv run python scripts/run_arch_preflight.py

arch-native-skeleton:
	uv run pytest tests/numerical/test_amp_arch_native_skeleton.py -q
	uv run python scripts/run_arch_native_skeleton.py

arch-training-feasibility:
	uv run pytest tests/unit/test_nablafx_loss.py tests/numerical/test_sota_comparators.py -q
	uv run python scripts/run_arch_physical_training_feasibility.py

arch-mechanism:
	uv run pytest tests/unit/test_amp_arch_gates.py tests/unit/test_amp_arch_registry.py tests/unit/test_amp_arch_training.py tests/numerical/test_amp_arch_fixtures.py tests/numerical/test_amp_quality_arch_models.py -q
	uv run python scripts/run_arch_mechanism.py

arch-v2-preflight:
	uv run pytest tests/unit/test_amp_competence_arch_v2_protocol.py tests/unit/test_amp_arch_v2_gates.py tests/unit/test_amp_arch_v2_training.py tests/numerical/test_amp_arch_v2_fixtures.py -q
	uv run python scripts/run_arch_v2_preflight.py

arch-v2-competence:
	uv run pytest tests/unit/test_amp_competence_arch_v2_protocol.py tests/unit/test_amp_arch_v2_gates.py tests/unit/test_amp_arch_v2_training.py tests/numerical/test_amp_arch_v2_fixtures.py -q
	uv run python scripts/run_arch_v2_competence.py

arch-v2-compare:
	uv run pytest tests/unit/test_amp_competence_arch_v2_protocol.py tests/unit/test_amp_arch_v2_gates.py tests/unit/test_amp_arch_v2_training.py tests/numerical/test_amp_arch_v2_fixtures.py -q
	uv run python scripts/run_arch_v2_compare.py

arch-v2-audit:
	uv run pytest tests/unit/test_amp_competence_arch_v2_protocol.py tests/unit/test_amp_arch_v2_gates.py tests/unit/test_amp_arch_v2_training.py tests/unit/test_amp_arch_v2_terminal_evidence.py tests/numerical/test_amp_arch_v2_fixtures.py -q
	uv run python scripts/run_arch_v2_audit.py

arch-v3-preflight:
	uv run pytest tests/unit/test_amp_quality_arch_v3_protocol.py tests/unit/test_amp_arch_v3_gates.py tests/numerical/test_amp_arch_v3_fixtures.py tests/numerical/test_amp_quality_arch_v3_models.py -q
	uv run python scripts/run_arch_v3_preflight.py

arch-v3-training-feasibility:
	uv run pytest tests/unit/test_amp_arch_v3_training.py tests/numerical/test_amp_quality_arch_v3_models.py -q
	uv run python scripts/run_arch_v3_training_feasibility.py

arch-v3-round-1:
	uv run pytest tests/unit/test_amp_quality_arch_v3_protocol.py tests/unit/test_amp_arch_v3_gates.py tests/unit/test_amp_arch_v3_training.py tests/numerical/test_amp_arch_v3_fixtures.py tests/numerical/test_amp_quality_arch_v3_models.py -q
	uv run python scripts/run_arch_v3_round_1.py

demo-plugin:
	cmake -S demo/plugin -B build/demo_plugin -DCMAKE_BUILD_TYPE=Release
	cmake --build build/demo_plugin --parallel 8

demo-report:
	uv run python scripts/product_plugin_parity.py
	uv run python scripts/product_report.py
	uv run python scripts/product_listening.py
	uv run python scripts/product_deck.py

# Rebuild the demo from the trained runs: plugin, parity check, fact sheet.
sota-bench:
	uv run python scripts/sota_bench.py

demo-capture-selftest:
	uv run python scripts/product_capture_selftest.py

demo: demo-plugin demo-report
