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
    return expected_protocol
