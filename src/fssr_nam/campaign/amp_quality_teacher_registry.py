"""Append-only evidence registry for AMP-QUALITY-TEACHER-v1."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from fssr_nam.campaign.amp_quality_teacher_v1 import (
    CAMPAIGN_VERSION,
    STAGES,
    load_protocol,
    parse_run_id,
)
from fssr_nam.reporting.ledger import REQUIRED_FIELDS as RUN_EVENT_REQUIRED_FIELDS

TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "invalid"})
REQUIRED_RUN_ARTIFACTS = (
    "resolved_config.yaml",
    "split.json",
    "seed.json",
    "command.json",
    "source_snapshot/manifest.json",
    "environment.json",
    "checkpoints/index.json",
    "metrics/per_file.json",
    "predictions/manifest.json",
    "history.json",
    "status.json",
)


class QualityTeacherRegistryError(RuntimeError):
    """Raised when campaign evidence is malformed or would rewrite history."""


def _read_json_lines(path: Path, label: str) -> list[dict[str, Any]]:
    if path.is_symlink():
        raise QualityTeacherRegistryError(f"{label} must be a regular file")
    if not path.exists():
        return []
    if not path.is_file():
        raise QualityTeacherRegistryError(f"{label} must be a regular file")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            event = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise QualityTeacherRegistryError(
                f"malformed {label} line {line_number}: {error}"
            ) from error
        if not isinstance(event, dict):
            raise QualityTeacherRegistryError(
                f"{label} line {line_number} must be an object"
            )
        events.append(event)
    return events


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _serialize_event(event: Mapping[str, Any], label: str) -> str:
    try:
        return json.dumps(
            dict(event), allow_nan=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as error:
        raise QualityTeacherRegistryError(
            f"{label} must contain finite JSON values"
        ) from error


def read_gate_events(path: Path) -> list[dict[str, Any]]:
    """Read gate events without accepting malformed registry lines."""
    return _read_json_lines(path, "quality-teacher gate registry")


def _validate_gate_event(event: Mapping[str, Any]) -> dict[str, Any]:
    required = {"campaign_version", "stage", "status", "evidence_path"}
    missing = sorted(required - set(event))
    if missing:
        raise QualityTeacherRegistryError(
            f"gate event missing required fields: {', '.join(missing)}"
        )
    if event.get("campaign_version") != CAMPAIGN_VERSION:
        raise QualityTeacherRegistryError("gate event campaign_version is invalid")
    stage = event.get("stage")
    if stage not in STAGES:
        raise QualityTeacherRegistryError(f"gate event has unknown stage: {stage!r}")
    status = event.get("status")
    if not isinstance(status, str) or not status:
        raise QualityTeacherRegistryError("gate event status must be non-empty")
    evidence_path = event.get("evidence_path")
    if not isinstance(evidence_path, str) or not evidence_path:
        raise QualityTeacherRegistryError("gate event evidence_path must be non-empty")
    normalized = dict(event)
    _serialize_event(normalized, "gate event")
    return normalized


def gate_decisions(path: Path) -> dict[str, str]:
    """Return one immutable decision per recorded campaign stage."""
    decisions: dict[str, str] = {}
    for raw_event in read_gate_events(path):
        event = _validate_gate_event(raw_event)
        stage = event["stage"]
        if stage in decisions:
            raise QualityTeacherRegistryError(f"duplicate gate event: {stage}")
        decisions[stage] = event["status"]
    return decisions


def append_gate_event(path: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    """Append a gate event, accepting only an identical idempotent replay."""
    normalized = _validate_gate_event(event)
    matches = [
        prior
        for prior in read_gate_events(path)
        if prior.get("stage") == normalized["stage"]
    ]
    if matches:
        if len(matches) == 1 and matches[0] == normalized:
            return matches[0]
        raise QualityTeacherRegistryError(
            f"gate {normalized['stage']} is append-only and already recorded"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = _serialize_event(normalized, "gate event")
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized + "\n")
    return normalized


def read_run_events(path: Path) -> list[dict[str, Any]]:
    """Read global run events without mutating or filtering the ledger."""
    return _read_json_lines(path, "run ledger")


def _artifact_names(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    run_artifacts = protocol.get("run_artifacts")
    if not isinstance(run_artifacts, Mapping):
        raise QualityTeacherRegistryError("protocol.run_artifacts must be a mapping")
    required = run_artifacts.get("required")
    if (
        not isinstance(required, Sequence)
        or isinstance(required, (str, bytes))
        or not all(isinstance(item, str) for item in required)
    ):
        raise QualityTeacherRegistryError(
            "protocol.run_artifacts.required must be a string sequence"
        )
    names = tuple(required)
    if names != REQUIRED_RUN_ARTIFACTS:
        raise QualityTeacherRegistryError(
            "protocol run artifact layout differs from PROTOCOL_LOCK"
        )
    for name in names:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
            raise QualityTeacherRegistryError(
                f"unsafe protocol artifact path: {name!r}"
            )
    return names


def _load_json_file(path: Path, label: str) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise QualityTeacherRegistryError(f"malformed {label}: {error}") from error


def _path_uses_symlink(path: Path, boundary: Path) -> bool:
    current = path
    while current != boundary:
        if current.is_symlink():
            return True
        parent = current.parent
        if parent == current:
            return True
        current = parent
    return boundary.is_symlink()


def validate_run_artifacts(
    root: Path,
    run_id: str,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Validate the frozen artifact layout for one terminal campaign run."""
    try:
        spec = parse_run_id(run_id)
    except (TypeError, ValueError, RuntimeError) as error:
        raise QualityTeacherRegistryError(
            f"invalid campaign run_id: {run_id!r}"
        ) from error

    active_protocol = load_protocol(root) if protocol is None else protocol
    artifact_names = _artifact_names(active_protocol)
    runs_root = root / "experiments" / "runs"
    run_dir = runs_root / run_id
    if run_dir.parent != runs_root:
        raise QualityTeacherRegistryError("run path escapes experiments/runs")
    if _path_uses_symlink(run_dir, root) or not run_dir.is_dir():
        raise QualityTeacherRegistryError(
            f"missing regular run directory: experiments/runs/{run_id}"
        )

    artifacts: dict[str, Path] = {}
    for name in artifact_names:
        artifact = run_dir.joinpath(*PurePosixPath(name).parts)
        if _path_uses_symlink(artifact, run_dir) or not artifact.is_file():
            raise QualityTeacherRegistryError(f"missing regular run artifact: {name}")
        artifacts[name] = artifact

    for name, artifact in artifacts.items():
        if name.endswith(".json"):
            _load_json_file(artifact, name)
    try:
        resolved_config = yaml.safe_load(
            artifacts["resolved_config.yaml"].read_text(encoding="utf-8")
        )
    except yaml.YAMLError as error:
        raise QualityTeacherRegistryError(
            f"malformed resolved_config.yaml: {error}"
        ) from error
    if not isinstance(resolved_config, Mapping):
        raise QualityTeacherRegistryError("resolved_config.yaml must be a mapping")

    seed_record = _load_json_file(artifacts["seed.json"], "seed.json")
    if (
        not isinstance(seed_record, Mapping)
        or isinstance(seed_record.get("seed"), bool)
        or seed_record.get("seed") != spec.seed
    ):
        raise QualityTeacherRegistryError("seed.json does not match parsed run_id")
    status_record = _load_json_file(artifacts["status.json"], "status.json")
    if not isinstance(status_record, Mapping):
        raise QualityTeacherRegistryError("status.json must be an object")
    if status_record.get("status") not in TERMINAL_RUN_STATUSES:
        raise QualityTeacherRegistryError(
            "status.json must record completed, failed, or invalid"
        )
    return artifacts


def _expected_results_path(run_id: str) -> str:
    return (PurePosixPath("experiments") / "runs" / run_id).as_posix()


def validate_run_event(
    root: Path,
    event: Mapping[str, Any],
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one global-ledger event and its immutable run artifacts."""
    missing = sorted(RUN_EVENT_REQUIRED_FIELDS - set(event))
    if missing:
        raise QualityTeacherRegistryError(
            f"run event missing required fields: {', '.join(missing)}"
        )
    run_id = event.get("run_id")
    if not isinstance(run_id, str):
        raise QualityTeacherRegistryError("run event run_id must be a string")
    try:
        spec = parse_run_id(run_id)
    except (TypeError, ValueError, RuntimeError) as error:
        raise QualityTeacherRegistryError(
            f"invalid campaign run_id: {run_id!r}"
        ) from error

    for field, expected in {
        "model": spec.family,
        "device": spec.device,
        "seed": spec.seed,
        "results_path": _expected_results_path(run_id),
    }.items():
        if event.get(field) != expected:
            raise QualityTeacherRegistryError(
                f"run event {field} does not match parsed run_id"
            )
    if event.get("phase") != spec.stage:
        raise QualityTeacherRegistryError(
            "run event phase does not match parsed run_id"
        )
    status = event.get("status")
    if status not in TERMINAL_RUN_STATUSES:
        raise QualityTeacherRegistryError(
            "run event status must be completed, failed, or invalid"
        )
    failure_reason = event.get("failure_reason")
    if failure_reason is not None and not isinstance(failure_reason, str):
        raise QualityTeacherRegistryError(
            "run event failure_reason must be a string or null"
        )
    if status in {"failed", "invalid"} and not failure_reason:
        raise QualityTeacherRegistryError(
            f"{status} run event requires a failure_reason"
        )

    artifacts = validate_run_artifacts(root, run_id, protocol)
    status_record = _load_json_file(artifacts["status.json"], "status.json")
    if status_record.get("status") != status:
        raise QualityTeacherRegistryError("run event status does not match status.json")
    artifact_reason = status_record.get("failure_reason")
    if status in {"failed", "invalid"} and artifact_reason != failure_reason:
        raise QualityTeacherRegistryError(
            "run event failure_reason does not match status.json"
        )

    normalized = dict(event)
    if normalized.get("campaign_version", CAMPAIGN_VERSION) != CAMPAIGN_VERSION:
        raise QualityTeacherRegistryError("run event campaign_version is invalid")
    _serialize_event(normalized, "run event")
    return normalized


def _infer_root(ledger_path: Path) -> Path:
    if ledger_path.parent.name == ".codex_campaign":
        return ledger_path.parent.parent
    return ledger_path.parent


def append_run_event(
    path: Path,
    event: Mapping[str, Any],
    root: Path | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one terminal run, accepting only an identical replay."""
    repository_root = _infer_root(path) if root is None else root
    normalized = validate_run_event(repository_root, event, protocol)
    run_id = normalized["run_id"]
    matches = [
        prior for prior in read_run_events(path) if prior.get("run_id") == run_id
    ]
    if matches:
        if len(matches) == 1 and matches[0] == normalized:
            return matches[0]
        raise QualityTeacherRegistryError(
            f"run {run_id} is append-only and already recorded"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = _serialize_event(normalized, "run event")
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized + "\n")
    return normalized


def validate_run_registration(
    root: Path,
    ledger_path: Path,
    run_id: str,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require exactly one valid ledger event for a terminal run directory."""
    matches = [
        event for event in read_run_events(ledger_path) if event.get("run_id") == run_id
    ]
    if len(matches) != 1:
        raise QualityTeacherRegistryError(
            f"run {run_id} requires exactly one ledger event, found {len(matches)}"
        )
    return validate_run_event(root, matches[0], protocol)


def validate_campaign_run_registrations(
    root: Path,
    ledger_path: Path,
    protocol: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Reject any campaign run directory or ledger event missing its counterpart."""
    prefix = "quality_teacher_v1_"
    runs_root = root / "experiments" / "runs"
    directory_ids = (
        {path.name for path in runs_root.iterdir() if path.name.startswith(prefix)}
        if runs_root.is_dir()
        else set()
    )
    event_ids = {
        event["run_id"]
        for event in read_run_events(ledger_path)
        if isinstance(event.get("run_id"), str) and event["run_id"].startswith(prefix)
    }
    missing_events = sorted(directory_ids - event_ids)
    missing_directories = sorted(event_ids - directory_ids)
    if missing_events:
        raise QualityTeacherRegistryError(
            f"campaign runs missing ledger events: {missing_events}"
        )
    if missing_directories:
        raise QualityTeacherRegistryError(
            f"ledger events missing campaign run directories: {missing_directories}"
        )
    for run_id in sorted(directory_ids):
        validate_run_registration(root, ledger_path, run_id, protocol)
    return tuple(sorted(directory_ids))
