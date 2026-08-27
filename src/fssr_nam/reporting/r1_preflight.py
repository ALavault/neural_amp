"""Non-destructive historical and protocol checks for FSSR-R1."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

CAMPAIGN_VERSION = "FSSR-R1-v1"
HISTORICAL_BASELINE = "b38d16f"
FROZEN_ROOTS = (
    ".codex_campaign",
    "reports",
    "experiments/runs",
    "experiments/summaries",
    "paper",
    "configs",
    "datasets/manifests",
    "datasets/splits",
)
APPEND_ONLY_PATHS = {
    ".codex_campaign/RUN_LEDGER.jsonl",
    ".codex_campaign/RESULT_INDEX.csv",
}
R1_PREFIXES = (
    ".codex_campaign/r1/",
    "configs/r1/",
    "configs/training/r1_",
    "configs/data/r1_",
    "datasets/manifests/r1_",
    "datasets/manifests/R1_",
    "datasets/splits/r1_",
    "experiments/runs/r1_",
    "experiments/summaries/r1_",
    "reports/R1_",
)
REQUIRED_RESOLVED_FIELDS = {
    "loss_mode",
    "checkpoint_steps",
    "phase_schedule",
    "receptive_field",
    "promotion_gate",
    "campaign_version",
}
STAGE_CONFIGS = (
    "configs/training/r1_competence.yaml",
    "configs/training/r1_factorial.yaml",
    "configs/training/r1_horizon.yaml",
    "configs/training/r1_cascade.yaml",
    "configs/training/r1_confirm.yaml",
)
DIAGNOSTIC_PROTOCOL_PATHS = (
    "configs/r1/protocol.yaml",
    "configs/training/r1_competence.yaml",
    "configs/training/r1_factorial.yaml",
    "configs/training/r1_horizon.yaml",
    "configs/training/r1_cascade.yaml",
    "configs/models/r1/wright_lstm64.yaml",
    "configs/models/r1/rf31.yaml",
    "configs/models/r1/rf2047.yaml",
    "configs/models/r1/cascade_rf2047.yaml",
    "configs/data/r1_physical.yaml",
    "configs/data/r1_wright_bigmuff_native.yaml",
)
DIAGNOSTIC_IMPLEMENTATION_PATHS = (
    "Makefile",
    "cpp/CMakeLists.txt",
    "cpp/R1_NATIVE_FORMAT.md",
    "cpp/benchmarks/r1_benchmark.cpp",
    "cpp/include/fssr_r1_native.hpp",
    "cpp/src/fssr_r1_native.cpp",
    "cpp/tests/r1_block_runner.cpp",
    "scripts/freeze_r1_diagnostic.py",
    "scripts/prepare_r1_data.py",
    "scripts/prepare_r1_wright_data.py",
    "scripts/run_r1_competence.py",
    "scripts/run_r1_competence_gate.py",
    "scripts/run_r1_diagnostic.py",
    "scripts/run_r1_preflight.py",
    "scripts/run_r1_stage.py",
    "scripts/summarize_r1.py",
    "scripts/validate_r1_wright.py",
    "src/fssr_nam/campaign",
    "src/fssr_nam/inference",
    "src/fssr_nam/losses",
    "src/fssr_nam/models/r1.py",
    "src/fssr_nam/models/wright.py",
    "src/fssr_nam/reporting/r1_preflight.py",
    "src/fssr_nam/statistics/r1.py",
    "src/fssr_nam/training/r1.py",
    "src/fssr_nam/training/r1_competence.py",
    "src/fssr_nam/training/r1_diagnostic.py",
    "src/fssr_nam/training/wright.py",
)
DIAGNOSTIC_ACTIVE_PATH = ".codex_campaign/r1/DIAGNOSTIC_LOCK_ACTIVE"
DIAGNOSTIC_AMENDMENT_1_PATH = ".codex_campaign/r1/DIAGNOSTIC_LOCK_AMENDMENT_1.yaml"
DIAGNOSTIC_AMENDMENT_2_PATH = ".codex_campaign/r1/DIAGNOSTIC_LOCK_AMENDMENT_2.yaml"


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _historical_paths(root: Path, baseline: str) -> list[str]:
    output = _git(root, "ls-tree", "-r", "--name-only", baseline, "--", *FROZEN_ROOTS)
    return [
        path
        for path in output.splitlines()
        if path and not any(path.startswith(prefix) for prefix in R1_PREFIXES)
    ]


def validate_historical_bytes(
    root: Path, baseline: str = HISTORICAL_BASELINE
) -> dict[str, int]:
    """Require every tracked M0-M6 evidence byte to match the baseline.

    The two global tabular registries may only gain a byte-for-byte suffix.
    New R1 paths are outside the historical comparison.
    """
    exact = 0
    append_only = 0
    for relative in _historical_paths(root, baseline):
        expected = subprocess.run(
            ["git", "show", f"{baseline}:{relative}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"historical file is absent: {relative}")
        actual = path.read_bytes()
        if relative in APPEND_ONLY_PATHS:
            if not actual.startswith(expected):
                raise RuntimeError(f"append-only history was rewritten: {relative}")
            append_only += 1
        else:
            if actual != expected:
                raise RuntimeError(f"historical evidence changed: {relative}")
            exact += 1
    return {"exact_files": exact, "append_only_files": append_only}


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def validate_stage_configs(root: Path) -> dict[str, str]:
    stages: dict[str, str] = {}
    for relative in STAGE_CONFIGS:
        config = load_yaml(root / relative)
        missing = REQUIRED_RESOLVED_FIELDS - set(config)
        if missing:
            raise ValueError(f"{relative} lacks fields: {sorted(missing)}")
        if config["campaign_version"] != CAMPAIGN_VERSION:
            raise ValueError(f"wrong campaign version in {relative}")
        stage = str(config["stage"])
        if stage in stages:
            raise ValueError(f"duplicate R1 stage config: {stage}")
        stages[stage] = relative
    if set(stages) != {"competence", "factorial", "horizon", "cascade", "confirm"}:
        raise ValueError("R1 stage config set is incomplete")
    return stages


def validate_protocol_counts(root: Path) -> dict[str, int]:
    protocol = load_yaml(root / "configs/r1/protocol.yaml")
    diagnostic = protocol["diagnostic"]
    caps = {key: int(value) for key, value in diagnostic["stages"].items()}
    if caps != {"competence": 3, "factorial": 8, "horizon": 4, "cascade": 4}:
        raise ValueError(f"unexpected diagnostic caps: {caps}")
    if (
        sum(caps.values()) != int(diagnostic["maximum_trajectories"])
        or sum(caps.values()) != 19
    ):
        raise ValueError("diagnostic trajectory maximum is not exactly 19")
    confirmation = protocol["confirmation"]
    primary = (
        len(confirmation["devices"])
        * len(confirmation["families"])
        * len(confirmation["primary_seeds"])
    )
    extension = (
        len(confirmation["devices"])
        * len(confirmation["extension_families"])
        * len(confirmation["extension_seeds"])
    )
    if primary != 60 or extension != 16 or primary + extension != 76:
        raise ValueError("confirmatory design does not expand to exactly 76 runs")
    return {**caps, "diagnostic_total": 19, "confirmatory_total": 76}


def validate_external_freeze(root: Path) -> dict[str, bool]:
    path = root / ".codex_campaign/r1/EXTERNAL_FREEZE.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "external_retest_authorized": False,
        "external_results_accessed": False,
        "internal_validation_outputs_locked": True,
    }
    for key, expected in required.items():
        if payload.get(key) is not expected:
            raise RuntimeError(f"R1 freeze control failed: {key}")
    return required


def validate_submodule_pins(root: Path) -> dict[str, str]:
    expected = {
        "third_party/Automated-GuitarAmpModelling": (
            "e3146386b0fd0b562bc393231be3a5938cf9feac"
        ),
        "third_party/Automated-GuitarAmpModelling/CoreAudioML": (
            "bad9469f94a2fa63a50d70ff75f5eff2208ba03f"
        ),
        "third_party/nablafx": "045db6e7d6087151c7e3a264844bd8c4eafc885c",
    }
    actual: dict[str, str] = {}
    for relative, commit in expected.items():
        path = root / relative
        if not path.is_dir():
            raise RuntimeError(f"required reference checkout is absent: {relative}")
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != commit:
            raise RuntimeError(f"reference pin mismatch for {relative}: {head}")
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if dirty:
            raise RuntimeError(f"reference checkout is dirty: {relative}")
        actual[relative] = head
    return actual


def validate_prepared_manifests(root: Path) -> dict[str, int]:
    physical_path = root / "datasets/manifests/r1_physical.json"
    wright_path = root / "datasets/manifests/r1_wright_bigmuff_native.json"
    physical = json.loads(physical_path.read_text(encoding="utf-8"))
    wright = json.loads(wright_path.read_text(encoding="utf-8"))
    if physical.get("campaign_version") != CAMPAIGN_VERSION:
        raise RuntimeError("physical R1 manifest has wrong campaign version")
    if physical.get("external_report_only_accessed") is not False:
        raise RuntimeError("physical R1 manifest accessed external report-only data")
    physical_config = root / str(physical.get("configuration", ""))
    if (
        not physical_config.is_file()
        or physical.get("configuration_sha256")
        != hashlib.sha256(physical_config.read_bytes()).hexdigest()
    ):
        raise RuntimeError("physical R1 manifest configuration binding mismatch")
    groups: dict[str, dict[str, set[str]]] = {}
    for entry in physical["files"]:
        groups.setdefault(entry["device"], {}).setdefault(entry["split"], set()).add(
            entry["source_id"]
        )
    for device, splits in groups.items():
        if set(splits) != {"train", "validation", "test"}:
            raise RuntimeError(f"incomplete prepared splits for {device}")
        if any(
            splits[left] & splits[right]
            for left, right in (
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            )
        ):
            raise RuntimeError(f"prepared source leakage for {device}")
    if {entry["split"] for entry in wright["files"]} != {
        "train",
        "validation",
        "test",
    }:
        raise RuntimeError("Wright native manifest is incomplete")
    if wright.get("dataset_license") != "CC-BY-NC-4.0":
        raise RuntimeError("Wright audio license is not recorded correctly")
    return {"physical_files": len(physical["files"]), "wright_files": 3}


def protocol_digest(
    root: Path, paths: Iterable[str] = DIAGNOSTIC_PROTOCOL_PATHS
) -> str:
    digest = hashlib.sha256()
    for relative in paths:
        encoded = relative.encode("utf-8")
        payload = (root / relative).read_bytes()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _validated_sha256_file(path: Path) -> str:
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    recorded = path.with_suffix(".sha256").read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", recorded) or recorded != expected:
        raise RuntimeError(f"lock amendment digest mismatch: {path.name}")
    return expected


def load_active_infrastructure_amendment(root: Path) -> dict[str, Any] | None:
    """Load amendment 2 only when the tracked active pointer selects it."""
    active_path = root / DIAGNOSTIC_ACTIVE_PATH
    if not active_path.is_file():
        return None
    active_relative = active_path.read_text(encoding="utf-8").strip()
    if active_relative != DIAGNOSTIC_AMENDMENT_2_PATH:
        return None
    validate_lock_digest(root)
    amendment = load_yaml(root / active_relative)
    from fssr_nam.campaign.r1 import validate_infrastructure_amendment

    validate_infrastructure_amendment(amendment)
    return amendment


def validate_lock_digest(root: Path) -> str:
    lock_path = root / ".codex_campaign/r1/DIAGNOSTIC_LOCK.yaml"
    checksum_path = root / ".codex_campaign/r1/DIAGNOSTIC_LOCK.sha256"
    lock = load_yaml(lock_path)
    if lock.get("status") != "frozen":
        raise RuntimeError("diagnostic protocol is not frozen")
    expected_protocol = protocol_digest(root)
    if lock.get("protocol_sha256") != expected_protocol:
        raise RuntimeError("diagnostic protocol digest mismatch")
    expected_lock = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    recorded = checksum_path.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", recorded) or recorded != expected_lock:
        raise RuntimeError("diagnostic lock digest mismatch")
    implementation_commit = lock.get("implementation_commit")

    amendment_1_path = root / DIAGNOSTIC_AMENDMENT_1_PATH
    amendment_1 = load_yaml(amendment_1_path)
    amendment_1_sha256 = _validated_sha256_file(amendment_1_path)
    if (
        amendment_1.get("status") != "frozen"
        or amendment_1.get("campaign_version") != CAMPAIGN_VERSION
        or amendment_1.get("amendment_number") != 1
        or amendment_1.get("base_lock_sha256") != expected_lock
        or amendment_1.get("protocol_sha256") != expected_protocol
        or amendment_1.get("scientific_protocol_changed") is not False
        or amendment_1.get("external_report_only_locked") is not True
    ):
        raise RuntimeError("diagnostic lock amendment 1 is inconsistent")

    active_path = root / DIAGNOSTIC_ACTIVE_PATH
    if active_path.exists():
        active_relative = active_path.read_text(encoding="utf-8").strip()
        if active_relative == DIAGNOSTIC_AMENDMENT_1_PATH:
            implementation_commit = amendment_1.get("effective_implementation_commit")
        elif active_relative == DIAGNOSTIC_AMENDMENT_2_PATH:
            amendment_path = root / active_relative
            amendment = load_yaml(amendment_path)
            _validated_sha256_file(amendment_path)
            from fssr_nam.campaign.r1 import validate_infrastructure_amendment

            validate_infrastructure_amendment(amendment)
            if (
                amendment.get("status") != "frozen"
                or amendment.get("campaign_version") != CAMPAIGN_VERSION
                or amendment.get("base_amendment_path") != DIAGNOSTIC_AMENDMENT_1_PATH
                or amendment.get("base_amendment_sha256") != amendment_1_sha256
                or amendment.get("protocol_sha256") != expected_protocol
            ):
                raise RuntimeError("diagnostic lock amendment 2 is inconsistent")
            implementation_commit = amendment.get("effective_implementation_commit")
        else:
            raise RuntimeError("unexpected active diagnostic lock amendment")

    if not isinstance(implementation_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", implementation_commit
    ):
        raise RuntimeError("diagnostic lock has no valid implementation commit")
    subprocess.run(
        ["git", "cat-file", "-e", f"{implementation_commit}^{{commit}}"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    drift = _git(
        root,
        "diff",
        "--name-only",
        implementation_commit,
        "--",
        *DIAGNOSTIC_IMPLEMENTATION_PATHS,
    ).strip()
    if drift:
        raise RuntimeError(
            "diagnostic implementation differs from the active lock: "
            + ", ".join(drift.splitlines())
        )
    return expected_protocol
