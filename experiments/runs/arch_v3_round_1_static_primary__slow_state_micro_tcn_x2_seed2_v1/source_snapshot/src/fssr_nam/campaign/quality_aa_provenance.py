"""Exact source/config/environment capture for dirty-worktree scientific runs."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

SOURCE_FILES = (
    "configs/quality_aa_v1/protocol.yaml",
    ".codex_campaign/quality_aa_v1/PROTOCOL_LOCK.yaml",
    "configs/quality_aa_v2/protocol.yaml",
    ".codex_campaign/quality_aa_v2/PROTOCOL_LOCK.yaml",
    "src/fssr_nam/campaign/quality_aa_v1.py",
    "src/fssr_nam/campaign/quality_aa_v2.py",
    "src/fssr_nam/campaign/quality_aa_gates.py",
    "src/fssr_nam/campaign/quality_aa_v2_gates.py",
    "src/fssr_nam/campaign/quality_aa_registry.py",
    "src/fssr_nam/campaign/quality_aa_v2_registry.py",
    "src/fssr_nam/campaign/quality_aa_provenance.py",
    "src/fssr_nam/data/r2_fixtures.py",
    "src/fssr_nam/metrics/quality_aliasing.py",
    "src/fssr_nam/metrics/quality_aa_mechanism.py",
    "src/fssr_nam/models/oversampling.py",
    "src/fssr_nam/models/r2.py",
    "src/fssr_nam/inference/r2_native.py",
    "src/fssr_nam/inference/r2_parity.py",
    "cpp/src/fssr_r1_native.cpp",
    "cpp/include/fssr_r1_native.hpp",
    "cpp/benchmarks/r2_interleaved_benchmark.cpp",
    "cpp/CMakeLists.txt",
    "scripts/run_quality_aa_preflight.py",
    "scripts/run_quality_aa_mechanism.py",
    "scripts/run_quality_aa_native.py",
    "scripts/run_quality_aa_audit.py",
    "configs/amp_quality_arch_v1/protocol.yaml",
    ".codex_campaign/amp_quality_arch_v1/PROTOCOL_LOCK.yaml",
    "src/fssr_nam/campaign/amp_quality_arch_v1.py",
    "src/fssr_nam/campaign/amp_arch_gates.py",
    "src/fssr_nam/campaign/amp_arch_registry.py",
    "src/fssr_nam/data/arch_fixtures.py",
    "src/fssr_nam/losses/nablafx.py",
    "src/fssr_nam/models/arch_v1.py",
    "src/fssr_nam/models/sota_comparators.py",
    "src/fssr_nam/training/arch_v1.py",
    "scripts/run_arch_mechanism.py",
    "configs/amp_competence_arch_v2/protocol.yaml",
    ".codex_campaign/amp_competence_arch_v2/PROTOCOL_LOCK.yaml",
    "src/fssr_nam/campaign/amp_competence_arch_v2.py",
    "src/fssr_nam/campaign/amp_arch_v2_gates.py",
    "src/fssr_nam/campaign/amp_arch_v2_registry.py",
    "src/fssr_nam/data/arch_v2_fixtures.py",
    "src/fssr_nam/training/arch_v2.py",
    "scripts/run_arch_v2_competence.py",
    "scripts/run_arch_v2_compare.py",
    "scripts/run_arch_v2_audit.py",
    "configs/amp_quality_arch_v3/protocol.yaml",
    "configs/amp_quality_arch_v3/round_1.yaml",
    ".codex_campaign/amp_quality_arch_v3/PROTOCOL_LOCK.yaml",
    ".codex_campaign/amp_quality_arch_v3/ROUND_1_LOCK.yaml",
    ".codex_campaign/amp_quality_arch_v3/TRAINING_FEASIBILITY.json",
    ".codex_campaign/amp_quality_arch_v3/TRAINING_FEASIBILITY_V2.json",
    "experiments/summaries/amp_quality_arch_v3/preflight.json",
    "src/fssr_nam/campaign/amp_quality_arch_v3.py",
    "src/fssr_nam/campaign/amp_arch_v3_gates.py",
    "src/fssr_nam/campaign/amp_arch_v3_registry.py",
    "src/fssr_nam/data/arch_v3_fixtures.py",
    "src/fssr_nam/models/arch_v3.py",
    "src/fssr_nam/training/arch_v3.py",
    "scripts/run_arch_v3_preflight.py",
    "scripts/run_arch_v3_training_feasibility.py",
    "scripts/run_arch_v3_round_1.py",
)


def _require_string_keys(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                location = "/".join(path) or "<root>"
                raise TypeError(
                    f"strict JSON mapping key at {location} must be text, got {key!r}"
                )
            _require_string_keys(item, (*path, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_string_keys(item, (*path, str(index)))


def strict_json(value: Any) -> str:
    _require_string_keys(value)
    return (
        json.dumps(
            value, allow_nan=False, indent=2, sort_keys=True, separators=(",", ": ")
        )
        + "\n"
    )


def write_new_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def write_new_json(path: Path, value: Any) -> None:
    write_new_text(path, strict_json(value))


def replace_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(strict_json(value), encoding="utf-8")
    temporary.replace(path)


def digest_text(text: str) -> str:
    """Return the SHA-256 required by the repository's run-ledger schema."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_state(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    head = run("rev-parse", "HEAD").strip()
    status = run("status", "--short")
    diff = run("diff", "--binary", "--no-ext-diff")
    return {"head": head, "status": status, "diff": diff, "dirty": bool(status)}


def capture_provenance(root: Path, run_dir: Path, command: list[str]) -> dict[str, Any]:
    """Capture exact decision-critical sources, including untracked files."""
    state = git_state(root)
    snapshot = run_dir / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=False)
    copied = []
    for relative in SOURCE_FILES:
        source = root / relative
        if not source.is_file():
            continue
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(relative)
    write_new_text(run_dir / "git-head.txt", state["head"] + "\n")
    write_new_text(run_dir / "git-status.txt", state["status"])
    write_new_text(run_dir / "git-diff.patch", state["diff"])
    write_new_json(
        run_dir / "command.json",
        {"argv": command, "cwd": str(root)},
    )
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "torch": version("torch"),
        "hypothesis": version("hypothesis"),
    }
    write_new_json(run_dir / "environment.json", environment)
    write_new_json(
        run_dir / "source_snapshot_manifest.json",
        {
            "method": "exact-file-copy-v1",
            "files": copied,
            "git_head": state["head"],
            "worktree_dirty": state["dirty"],
            "untracked_sources_captured_by_copy": True,
        },
    )
    return {
        "git_head": state["head"],
        "worktree_dirty": state["dirty"],
        "source_snapshot_files": len(copied),
        "environment": environment,
    }
