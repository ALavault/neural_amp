"""Declarative, append-only orchestration for the prospective FSSR-R1 lineage.

This module deliberately does not train models or evaluate promotion gates.  It
validates the frozen declarations, expands the bounded matrices, checks explicit
gate decisions, reserves a never-reused run directory, and records terminal run
statuses in the global ledger.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from fssr_nam.reporting.ledger import append_run, read_runs

CAMPAIGN_VERSION = "FSSR-R1-v1"
DIAGNOSTIC_STAGES = ("competence", "factorial", "horizon", "cascade")
DIAGNOSTIC_STAGE_CAPS = {
    "competence": 3,
    "factorial": 8,
    "horizon": 4,
    "cascade": 4,
}
CONFIRMATORY_RUN_CAP = 76
STAGE_CAPS = {**DIAGNOSTIC_STAGE_CAPS, "confirm": CONFIRMATORY_RUN_CAP}
RUN_ID_STAGES = (
    "preflight",
    *DIAGNOSTIC_STAGES,
    "lock",
    "confirm",
    "benchmark",
    "audit",
)
STAGE_CONFIG_PATHS = {
    stage: Path(f"configs/training/r1_{stage}.yaml")
    for stage in (*DIAGNOSTIC_STAGES, "confirm")
}
PLACEHOLDER_LOSSES = frozenset(
    {"selected_by_factorial_gate", "frozen_by_confirmatory_lock"}
)
TERMINAL_STATUSES = frozenset({"completed", "failed", "stopped_by_gate"})
# A nonconforming, unauthorized implementation used this non-canonical
# identifier after the initial R1 lock but before any valid counted trajectory.
# Its failed ledger line is immutable and its artifacts are quarantined, so it
# must remain auditable without consuming a declared competence slot or making
# canonical planning impossible.  The incident motivates an amended lock.
QUARANTINED_NONCANONICAL_RUN_IDS = frozenset(
    {"r1_competence_bigmuff_wright_lstm64_wright_seed0_v1"}
)

_ID_TOKEN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STAGE_ALTERNATION = "|".join(re.escape(stage) for stage in RUN_ID_STAGES)
_RUN_ID = re.compile(
    rf"^r1_(?P<stage>{_STAGE_ALTERNATION})_"
    r"(?P<device>[a-z0-9][a-z0-9-]*)_"
    r"(?P<model>[a-z0-9][a-z0-9-]*)_"
    r"(?P<loss>[a-z0-9][a-z0-9-]*)_"
    r"seed(?P<seed>0|[1-9][0-9]*)_v1$"
)


class R1CampaignError(RuntimeError):
    """Base class for an R1 campaign invariant violation."""


class R1ConfigError(R1CampaignError):
    """A declarative R1 configuration is missing or inconsistent."""


class R1AuthorizationError(R1CampaignError):
    """A run is not authorized by the registered sequential gates."""


class R1RunReuseError(R1CampaignError):
    """A run identifier or immutable run directory would be reused."""


def _id_token(value: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    token = value.replace("_", "-")
    if not _ID_TOKEN.fullmatch(token):
        raise ValueError(f"invalid {field} token: {value!r}")
    return token


def make_run_id(stage: str, device: str, model: str, loss: str, seed: int) -> str:
    """Build the sole permitted R1 run identifier form."""
    if stage not in RUN_ID_STAGES:
        raise ValueError(f"unknown R1 stage: {stage}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return (
        f"r1_{stage}_{_id_token(device, 'device')}_"
        f"{_id_token(model, 'model')}_{_id_token(loss, 'loss')}_seed{seed}_v1"
    )


@dataclass(frozen=True, slots=True)
class RunSpec:
    """One declared R1 trajectory or confirmatory condition."""

    stage: str
    device: str
    model: str
    loss: str
    seed: int

    def __post_init__(self) -> None:
        make_run_id(self.stage, self.device, self.model, self.loss, self.seed)

    @property
    def run_id(self) -> str:
        return make_run_id(self.stage, self.device, self.model, self.loss, self.seed)

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "device": self.device,
            "model": self.model,
            "loss": self.loss,
            "seed": self.seed,
            "gate_requirements": list(gate_requirements(self)),
        }


def parse_run_id(run_id: str) -> RunSpec:
    """Parse a strict R1 identifier, rejecting ambiguous or versionless forms."""
    if not isinstance(run_id, str):
        raise ValueError("run_id must be a string")
    match = _RUN_ID.fullmatch(run_id)
    if match is None:
        raise ValueError(f"invalid R1 run_id: {run_id!r}")
    return RunSpec(
        stage=match.group("stage"),
        device=match.group("device"),
        model=match.group("model"),
        loss=match.group("loss"),
        seed=int(match.group("seed")),
    )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise R1ConfigError(f"missing R1 configuration: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise R1ConfigError(f"R1 configuration must be a mapping: {path}")
    return value


def _require_keys(config: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - set(config))
    if missing:
        raise R1ConfigError(f"{label} missing fields: {', '.join(missing)}")


def _require_exact_sequence(
    config: Mapping[str, Any], key: str, expected: Sequence[object], label: str
) -> None:
    value = config.get(key)
    if not isinstance(value, list) or value != list(expected):
        raise R1ConfigError(
            f"{label}.{key} must be exactly {list(expected)!r}, got {value!r}"
        )


def _validate_step_schedule(config: Mapping[str, Any], label: str) -> None:
    optimizer_steps = config.get("optimizer_steps")
    if optimizer_steps != 5000:
        raise R1ConfigError(f"{label}.optimizer_steps must equal 5000")
    schedule = config["phase_schedule"]
    if not isinstance(schedule, list) or not schedule:
        raise R1ConfigError(f"{label}.phase_schedule must be a non-empty list")
    previous_end = 0
    for index, phase in enumerate(schedule):
        if not isinstance(phase, dict):
            raise R1ConfigError(f"{label}.phase_schedule[{index}] must be a mapping")
        _require_keys(phase, {"name", "start_step", "end_step", "trainable"}, label)
        start = phase["start_step"]
        end = phase["end_step"]
        if start != previous_end + 1 or not isinstance(end, int) or end < start:
            raise R1ConfigError(f"{label}.phase_schedule is not contiguous")
        previous_end = end
    if previous_end != optimizer_steps:
        raise R1ConfigError(f"{label}.phase_schedule must end at step 5000")


def validate_stage_config(
    config: Mapping[str, Any], expected_stage: str | None = None
) -> None:
    """Validate one R1 stage declaration without evaluating any measurements."""
    common = {
        "schema_version",
        "campaign_version",
        "stage",
        "loss_mode",
        "checkpoint_steps",
        "phase_schedule",
        "receptive_field",
        "promotion_gate",
    }
    _require_keys(config, common, expected_stage or "R1 stage")
    stage = config["stage"]
    if stage not in STAGE_CONFIG_PATHS:
        raise R1ConfigError(f"unknown declared R1 stage: {stage!r}")
    if expected_stage is not None and stage != expected_stage:
        raise R1ConfigError(
            f"expected stage {expected_stage!r}, configuration declares {stage!r}"
        )
    label = f"r1_{stage}"
    if config["schema_version"] != 1:
        raise R1ConfigError(f"{label}.schema_version must equal 1")
    if config["campaign_version"] != CAMPAIGN_VERSION:
        raise R1ConfigError(f"{label}.campaign_version must equal {CAMPAIGN_VERSION!r}")
    if not isinstance(config["loss_mode"], str) or not config["loss_mode"]:
        raise R1ConfigError(f"{label}.loss_mode must be a non-empty string")
    checkpoints = config["checkpoint_steps"]
    if (
        not isinstance(checkpoints, list)
        or any(
            isinstance(step, bool) or not isinstance(step, int) for step in checkpoints
        )
        or checkpoints != sorted(set(checkpoints))
    ):
        raise R1ConfigError(f"{label}.checkpoint_steps must be sorted unique integers")
    if not isinstance(config["promotion_gate"], dict) or not config["promotion_gate"]:
        raise R1ConfigError(f"{label}.promotion_gate must be a non-empty mapping")
    if not isinstance(config["receptive_field"], (int, str, dict)):
        raise R1ConfigError(f"{label}.receptive_field has an invalid declaration")

    if stage == "competence":
        _require_exact_sequence(config, "seeds", [0, 1, 2], label)
        if (
            config.get("device") != "bigmuff"
            or config.get("model") != "lstm64"
            or config["loss_mode"] != "wright"
        ):
            raise R1ConfigError(f"{label} must declare bigmuff/lstm64/wright")
        if checkpoints:
            raise R1ConfigError(f"{label}.checkpoint_steps must be empty")
    elif stage == "factorial":
        _require_exact_sequence(config, "devices", ["fulltone", "bigmuff"], label)
        _require_exact_sequence(config, "models", ["a2", "s3"], label)
        _require_exact_sequence(config, "loss_modes", ["m4", "wright"], label)
        _require_exact_sequence(config, "seeds", [0], label)
        _require_exact_sequence(config, "checkpoint_steps", [200, 1000, 5000], label)
        _validate_step_schedule(config, label)
    elif stage == "horizon":
        _require_exact_sequence(config, "devices", ["fulltone", "bigmuff"], label)
        _require_exact_sequence(config, "models", ["rf31", "rf2047"], label)
        _require_exact_sequence(config, "seeds", [0], label)
        _require_exact_sequence(config, "checkpoint_steps", [200, 1000, 5000], label)
        _validate_step_schedule(config, label)
        rf2047 = config.get("rf2047")
        if not isinstance(rf2047, dict):
            raise R1ConfigError(f"{label}.rf2047 must be a mapping")
        if (
            rf2047.get("layers") != 10
            or rf2047.get("kernel_size") != 3
            or rf2047.get("channels") != 8
            or rf2047.get("dilations") != [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        ):
            raise R1ConfigError(f"{label}.rf2047 does not declare exact RF2047")
    elif stage == "cascade":
        _require_exact_sequence(config, "synthetic_models", ["mono", "cascade"], label)
        _require_exact_sequence(
            config, "physical_devices", ["bigmuff", "fulltone"], label
        )
        _require_exact_sequence(config, "physical_models", ["cascade"], label)
        _require_exact_sequence(config, "seeds", [0], label)
        _require_exact_sequence(config, "checkpoint_steps", [200, 1000, 5000], label)
        _validate_step_schedule(config, label)
    elif stage == "confirm":
        _require_exact_sequence(
            config,
            "devices",
            ["fulltone", "bigmuff", "blackstar", "ua1176"],
            label,
        )
        _require_exact_sequence(
            config,
            "families",
            [
                "a2",
                "lstm_cost_equivalent",
                "nablafx_graybox",
                "r1_ablation",
                "r1_final",
            ],
            label,
        )
        _require_exact_sequence(config, "primary_seeds", [0, 1, 2], label)
        _require_exact_sequence(config, "extension_seeds", [3, 4], label)
        _require_exact_sequence(config, "extension_families", ["a2", "r1_final"], label)
        _require_exact_sequence(config, "checkpoint_steps", [200, 1000, 5000], label)
        if config.get("expected_run_count") != CONFIRMATORY_RUN_CAP:
            raise R1ConfigError(f"{label}.expected_run_count must equal 76")
        if config.get("optimizer_steps") != 5000:
            raise R1ConfigError(f"{label}.optimizer_steps must equal 5000")


def validate_protocol_config(config: Mapping[str, Any]) -> None:
    """Validate the caps and exact confirmatory dimensions in protocol.yaml."""
    _require_keys(
        config,
        {"schema_version", "campaign_version", "diagnostic", "confirmation"},
        "r1_protocol",
    )
    if config["schema_version"] != 1 or config["campaign_version"] != CAMPAIGN_VERSION:
        raise R1ConfigError("invalid R1 protocol schema or campaign version")
    diagnostic = config["diagnostic"]
    confirmation = config["confirmation"]
    if not isinstance(diagnostic, dict) or not isinstance(confirmation, dict):
        raise R1ConfigError("R1 protocol diagnostic and confirmation must be mappings")
    if diagnostic.get("maximum_trajectories") != sum(DIAGNOSTIC_STAGE_CAPS.values()):
        raise R1ConfigError("R1 diagnostic maximum must equal 19")
    if diagnostic.get("stages") != DIAGNOSTIC_STAGE_CAPS:
        raise R1ConfigError("R1 diagnostic stage caps must be exactly 3/8/4/4")
    if confirmation.get("maximum_runs") != CONFIRMATORY_RUN_CAP:
        raise R1ConfigError("R1 confirmation maximum must equal 76")
    expected = {
        "devices": ["fulltone", "bigmuff", "blackstar", "ua1176"],
        "families": [
            "a2",
            "lstm_cost_equivalent",
            "nablafx_graybox",
            "r1_ablation",
            "r1_final",
        ],
        "primary_seeds": [0, 1, 2],
        "extension_seeds": [3, 4],
        "extension_families": ["a2", "r1_final"],
    }
    for key, value in expected.items():
        if confirmation.get(key) != value:
            raise R1ConfigError(f"R1 protocol confirmation.{key} is not frozen")


def validate_repository_configs(root: Path) -> dict[str, dict[str, Any]]:
    """Load and validate the protocol plus all five R1 stage declarations."""
    protocol = _load_yaml_mapping(root / "configs/r1/protocol.yaml")
    validate_protocol_config(protocol)
    configs: dict[str, dict[str, Any]] = {}
    for stage, relative_path in STAGE_CONFIG_PATHS.items():
        config = _load_yaml_mapping(root / relative_path)
        validate_stage_config(config, stage)
        configs[stage] = config
    return configs


def _expanded_stage(
    config: Mapping[str, Any], resolved_loss: str | None
) -> tuple[RunSpec, ...]:
    stage = config["stage"]
    loss = resolved_loss or config["loss_mode"]
    if stage == "competence":
        specs = [
            RunSpec(stage, config["device"], config["model"], loss, seed)
            for seed in config["seeds"]
        ]
    elif stage == "factorial":
        specs = [
            RunSpec(stage, device, model, loss_mode, seed)
            for device in config["devices"]
            for model in config["models"]
            for loss_mode in config["loss_modes"]
            for seed in config["seeds"]
        ]
    elif stage == "horizon":
        specs = [
            RunSpec(stage, device, model, loss, seed)
            for device in config["devices"]
            for model in config["models"]
            for seed in config["seeds"]
        ]
    elif stage == "cascade":
        specs = [
            RunSpec(stage, "synthetic", model, loss, seed)
            for model in config["synthetic_models"]
            for seed in config["seeds"]
        ]
        specs.extend(
            RunSpec(stage, device, model, loss, seed)
            for device in config["physical_devices"]
            for model in config["physical_models"]
            for seed in config["seeds"]
        )
    elif stage == "confirm":
        specs = list(_expanded_confirmatory(config, loss))
    else:  # pragma: no cover - validate_stage_config excludes this branch
        raise R1ConfigError(f"unsupported R1 stage: {stage}")
    return tuple(specs)


def expand_stage(
    config: Mapping[str, Any], *, resolved_loss: str | None = None
) -> tuple[RunSpec, ...]:
    """Expand every potential condition of one bounded R1 stage."""
    validate_stage_config(config)
    specs = _expanded_stage(config, resolved_loss)
    expected = STAGE_CAPS[config["stage"]]
    if len(specs) != expected or len({spec.run_id for spec in specs}) != expected:
        raise R1ConfigError(
            f"r1_{config['stage']} must expand to exactly {expected} unique runs"
        )
    return specs


def _expanded_confirmatory(
    config: Mapping[str, Any], loss_mode: str
) -> tuple[RunSpec, ...]:
    specs = [
        RunSpec("confirm", device, family, loss_mode, seed)
        for device in config["devices"]
        for family in config["families"]
        for seed in config["primary_seeds"]
    ]
    specs.extend(
        RunSpec("confirm", device, family, loss_mode, seed)
        for device in config["devices"]
        for family in config["extension_families"]
        for seed in config["extension_seeds"]
    )
    return tuple(specs)


def expand_confirmatory_conditions(
    config: Mapping[str, Any], *, loss_mode: str | None = None
) -> tuple[RunSpec, ...]:
    """Expand 60 primary plus 16 extension conditions, never more or fewer."""
    validate_stage_config(config, "confirm")
    specs = _expanded_confirmatory(config, loss_mode or config["loss_mode"])
    if len(specs) != CONFIRMATORY_RUN_CAP:
        raise R1ConfigError("R1 confirmation did not expand to exactly 76 runs")
    if len({spec.run_id for spec in specs}) != CONFIRMATORY_RUN_CAP:
        raise R1ConfigError("R1 confirmation expansion contains duplicate run IDs")
    return specs


def gate_requirements(spec: RunSpec) -> tuple[str, ...]:
    """Return the explicit promotion decisions required before a trajectory."""
    if spec.stage == "competence":
        return () if spec.seed == 0 else ("competence_seed0",)
    if spec.stage == "factorial":
        return ("competence",)
    if spec.stage == "horizon":
        return ("factorial",)
    if spec.stage == "cascade":
        requirements = ["horizon"]
        if spec.device != "synthetic":
            requirements.append("cascade_synthetic")
        return tuple(requirements)
    return ()


def _decision_passed(value: object) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value in {"passed", "promoted"}
    if isinstance(value, Mapping):
        return value.get("passed") is True or value.get("decision") in {
            "passed",
            "promoted",
        }
    return False


def _decision_failed(value: object) -> bool:
    if value is False:
        return True
    if isinstance(value, str):
        return value in {"failed", "rejected", "stopped_by_gate"}
    if isinstance(value, Mapping):
        return value.get("passed") is False or value.get("decision") in {
            "failed",
            "rejected",
            "stopped_by_gate",
        }
    return False


def validate_confirmatory_lock(lock: Mapping[str, Any]) -> None:
    """Require an explicit H1/H2 authorization and verified Python/C++ parity."""
    if lock.get("confirmation_runs_authorized") is not True:
        raise R1AuthorizationError("confirmatory lock does not authorize runs")
    if lock.get("hypothesis") not in {"H1", "H2"}:
        raise R1AuthorizationError("confirmatory lock must select H1 or H2")
    if lock.get("python_cpp_parity") is not True:
        raise R1AuthorizationError("confirmatory lock requires python_cpp_parity: true")
    if lock.get("external_report_only_locked") is not True:
        raise R1AuthorizationError("EXTERNAL_REPORT_ONLY must remain locked")
    for field in ("candidate", "ablation", "loss_mode"):
        if not isinstance(lock.get(field), str) or not lock[field]:
            raise R1AuthorizationError(f"confirmatory lock must freeze {field}")
    if lock["loss_mode"] in PLACEHOLDER_LOSSES:
        raise R1AuthorizationError("confirmatory lock has an unresolved loss mode")
    protocol_hash = lock.get("protocol_sha256")
    if not isinstance(protocol_hash, str) or not _SHA256.fullmatch(protocol_hash):
        raise R1AuthorizationError("confirmatory lock needs a valid protocol_sha256")


def validate_gate_authorization(
    spec: RunSpec,
    decisions: Mapping[str, object],
    *,
    confirmatory_lock: Mapping[str, Any] | None = None,
) -> None:
    """Reject a trajectory unless every preceding decision is explicit and passed."""
    missing = [
        gate
        for gate in gate_requirements(spec)
        if not _decision_passed(decisions.get(gate))
    ]
    if missing:
        raise R1AuthorizationError(
            f"{spec.run_id} is locked by gates: {', '.join(missing)}"
        )
    if spec.stage == "confirm":
        if confirmatory_lock is None:
            raise R1AuthorizationError("confirmation requires a confirmatory lock")
        validate_confirmatory_lock(confirmatory_lock)
        if spec.loss != confirmatory_lock["loss_mode"]:
            raise R1AuthorizationError(
                "confirmation loss does not match the confirmatory lock"
            )


def _validate_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


class R1Executor:
    """Reserve and finalize immutable R1 runs without executing training code."""

    def __init__(
        self,
        root: Path,
        *,
        stage_configs: Mapping[str, Mapping[str, Any]] | None = None,
        gate_decisions: Mapping[str, object] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.runs_dir = self.root / "experiments/runs"
        self.ledger_path = self.root / ".codex_campaign/RUN_LEDGER.jsonl"
        self._stage_configs = (
            {stage: dict(config) for stage, config in stage_configs.items()}
            if stage_configs is not None
            else None
        )
        self.gate_decisions = dict(gate_decisions or {})

    def stage_config(self, stage: str) -> dict[str, Any]:
        if stage not in STAGE_CONFIG_PATHS:
            raise R1ConfigError(f"stage has no declarative run matrix: {stage}")
        if self._stage_configs is None:
            config = _load_yaml_mapping(self.root / STAGE_CONFIG_PATHS[stage])
        else:
            if stage not in self._stage_configs:
                raise R1ConfigError(f"missing injected stage configuration: {stage}")
            config = dict(self._stage_configs[stage])
        validate_stage_config(config, stage)
        return config

    def plan(
        self, stage: str, *, resolved_loss: str | None = None
    ) -> tuple[RunSpec, ...]:
        return expand_stage(self.stage_config(stage), resolved_loss=resolved_loss)

    def _attempted_run_ids(self, stages: set[str]) -> set[str]:
        attempted: set[str] = set()
        for entry in read_runs(self.ledger_path):
            run_id = entry.get("run_id")
            if not isinstance(run_id, str) or not run_id.startswith("r1_"):
                continue
            if run_id in QUARANTINED_NONCANONICAL_RUN_IDS:
                continue
            parsed = parse_run_id(run_id)
            if parsed.stage in stages:
                attempted.add(run_id)
        if self.runs_dir.is_dir():
            for path in self.runs_dir.iterdir():
                if not path.is_dir() or not path.name.startswith("r1_"):
                    continue
                if path.name in QUARANTINED_NONCANONICAL_RUN_IDS:
                    continue
                parsed = parse_run_id(path.name)
                if parsed.stage in stages:
                    attempted.add(path.name)
        return attempted

    def quarantined_noncanonical_attempts(self) -> tuple[dict[str, Any], ...]:
        """Return immutable legacy attempts excluded from canonical R1 caps."""
        return tuple(
            dict(entry)
            for entry in read_runs(self.ledger_path)
            if entry.get("run_id") in QUARANTINED_NONCANONICAL_RUN_IDS
        )

    def stage_usage(self, stage: str) -> tuple[int, int]:
        """Return attempted unique IDs and the immutable cap for one run stage."""
        if stage not in STAGE_CAPS:
            raise R1ConfigError(f"stage has no trajectory cap: {stage}")
        return len(self._attempted_run_ids({stage})), STAGE_CAPS[stage]

    def _validate_fresh_capacity(self, spec: RunSpec) -> None:
        existing_ledger_ids = {
            entry.get("run_id") for entry in read_runs(self.ledger_path)
        }
        if spec.run_id in existing_ledger_ids:
            raise R1RunReuseError(f"run_id already registered: {spec.run_id}")
        run_dir = self.runs_dir / spec.run_id
        if run_dir.exists():
            raise R1RunReuseError(f"run directory already exists: {run_dir}")
        used, cap = self.stage_usage(spec.stage)
        if used >= cap:
            raise R1AuthorizationError(f"r1_{spec.stage} cap exhausted: {used}/{cap}")
        if spec.stage in DIAGNOSTIC_STAGES:
            diagnostic_used = len(self._attempted_run_ids(set(DIAGNOSTIC_STAGES)))
            if diagnostic_used >= sum(DIAGNOSTIC_STAGE_CAPS.values()):
                raise R1AuthorizationError(
                    f"R1 diagnostic cap exhausted: {diagnostic_used}/19"
                )

    def _validate_declared(
        self,
        spec: RunSpec,
        *,
        confirmatory_lock: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if spec.loss in PLACEHOLDER_LOSSES:
            raise R1AuthorizationError(
                f"{spec.run_id} has an unresolved gate-selected loss"
            )
        config = self.stage_config(spec.stage)
        resolved_loss = (
            spec.loss if spec.stage in {"horizon", "cascade", "confirm"} else None
        )
        declared = {
            candidate.run_id
            for candidate in expand_stage(config, resolved_loss=resolved_loss)
        }
        if spec.run_id not in declared:
            raise R1ConfigError(f"run is outside the declared matrix: {spec.run_id}")
        validate_gate_authorization(
            spec,
            self.gate_decisions,
            confirmatory_lock=confirmatory_lock,
        )
        return config

    def prepare_run(
        self,
        spec: RunSpec,
        *,
        data_sha256: str,
        commit: str,
        command: str = "",
        confirmatory_lock: Mapping[str, Any] | None = None,
        stopped_by_gate: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, object]:
        """Validate and reserve one fresh run; dry-run performs no filesystem write."""
        _validate_sha256(data_sha256, "data_sha256")
        if not isinstance(commit, str) or not commit:
            raise ValueError("commit must be a non-empty string")
        if stopped_by_gate is None:
            config = self._validate_declared(
                spec, confirmatory_lock=confirmatory_lock
            )
        else:
            if stopped_by_gate not in gate_requirements(spec):
                raise R1AuthorizationError(
                    f"{stopped_by_gate!r} is not a required gate for {spec.run_id}"
                )
            if not _decision_failed(self.gate_decisions.get(stopped_by_gate)):
                raise R1AuthorizationError(
                    f"gate stop requires an explicit failed decision: {stopped_by_gate}"
                )
            config = self.stage_config(spec.stage)
            resolved_loss = (
                spec.loss if spec.stage in {"horizon", "cascade", "confirm"} else None
            )
            declared = {
                candidate.run_id
                for candidate in expand_stage(config, resolved_loss=resolved_loss)
            }
            if spec.run_id not in declared:
                raise R1ConfigError(
                    f"gate-stopped run is outside the declared matrix: {spec.run_id}"
                )
        self._validate_fresh_capacity(spec)
        started = datetime.now().astimezone().isoformat(timespec="seconds")
        resolved = {
            "campaign_version": CAMPAIGN_VERSION,
            "stage_config": config,
            "selected_run": spec.as_dict(),
            "authorization": {
                "gate_decisions": {
                    gate: self.gate_decisions.get(gate, "not_reached")
                    for gate in gate_requirements(spec)
                },
                "confirmatory_lock": (
                    dict(confirmatory_lock) if confirmatory_lock is not None else None
                ),
                "stopped_by_gate": stopped_by_gate,
            },
        }
        config_bytes = yaml.safe_dump(resolved, sort_keys=False).encode("utf-8")
        config_hash = hashlib.sha256(config_bytes).hexdigest()
        run_dir = self.runs_dir / spec.run_id
        result: dict[str, object] = {
            "action": "would_reserve" if dry_run else "reserved",
            "run_id": spec.run_id,
            "run_directory": str(run_dir),
            "config_sha256": config_hash,
            "data_sha256": data_sha256,
        }
        if dry_run:
            return result

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(exist_ok=False)
        (run_dir / "config-resolved.yaml").write_bytes(config_bytes)
        (run_dir / "run-spec.json").write_text(
            json.dumps(spec.as_dict(), indent=2) + "\n", encoding="utf-8"
        )
        (run_dir / "command.txt").write_text(command.rstrip() + "\n", encoding="utf-8")
        (run_dir / "status.json").write_text(
            json.dumps({"status": "running", "started_at": started}, indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "registration.json").write_text(
            json.dumps(
                {
                    "commit": commit,
                    "config_sha256": config_hash,
                    "data_sha256": data_sha256,
                    "started_at": started,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return result

    def record_gate_stop(
        self,
        spec: RunSpec,
        *,
        gate: str,
        reason: str,
        data_sha256: str,
        commit: str,
        command: str = "",
    ) -> dict[str, object]:
        """Register a declared trajectory that an explicit failed gate forbids."""
        if not reason.strip():
            raise ValueError("gate stop requires a non-empty reason")
        self.prepare_run(
            spec,
            data_sha256=data_sha256,
            commit=commit,
            command=command,
            stopped_by_gate=gate,
        )
        run_dir = self.runs_dir / spec.run_id
        (run_dir / "gate-stop.json").write_text(
            json.dumps({"gate": gate, "reason": reason}, indent=2) + "\n",
            encoding="utf-8",
        )
        return self.finalize_run(
            spec,
            status="stopped_by_gate",
            failure_reason=reason,
        )

    def finalize_run(
        self,
        spec: RunSpec,
        *,
        status: str,
        failure_reason: str = "",
    ) -> dict[str, object]:
        """Append one terminal result, including failed and gate-stopped runs."""
        if status not in TERMINAL_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(TERMINAL_STATUSES)}, got {status!r}"
            )
        if status in {"failed", "stopped_by_gate"} and not failure_reason.strip():
            raise ValueError(f"{status} requires a non-empty failure_reason")
        if status == "completed" and failure_reason:
            raise ValueError("completed run cannot have a failure_reason")
        run_dir = self.runs_dir / spec.run_id
        if not run_dir.is_dir():
            raise R1CampaignError(f"run was not reserved: {spec.run_id}")
        persisted_spec = json.loads(
            (run_dir / "run-spec.json").read_text(encoding="utf-8")
        )
        if persisted_spec.get("run_id") != spec.run_id:
            raise R1CampaignError(f"reserved run spec mismatch: {spec.run_id}")
        current_status = json.loads(
            (run_dir / "status.json").read_text(encoding="utf-8")
        )
        if current_status.get("status") != "running":
            raise R1RunReuseError(f"run is already terminal: {spec.run_id}")
        registration = json.loads(
            (run_dir / "registration.json").read_text(encoding="utf-8")
        )
        finished = datetime.now().astimezone().isoformat(timespec="seconds")
        terminal = {
            "status": status,
            "started_at": current_status["started_at"],
            "finished_at": finished,
            "failure_reason": failure_reason,
        }
        (run_dir / "status.json").write_text(
            json.dumps(terminal, indent=2) + "\n", encoding="utf-8"
        )
        relative_path = str(run_dir.relative_to(self.root))
        entry = {
            "date": finished,
            "run_id": spec.run_id,
            "phase": f"R1_{spec.stage.upper()}",
            "model": spec.model,
            "device": spec.device,
            "seed": spec.seed,
            "commit": registration["commit"],
            "config_sha256": registration["config_sha256"],
            "data_sha256": registration["data_sha256"],
            "status": status,
            "failure_reason": failure_reason,
            "results_path": relative_path,
        }
        append_run(self.ledger_path, entry)
        return entry


def load_gate_decisions(path: Path) -> dict[str, object]:
    """Load explicit decisions from JSON; values are never computed here."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise R1ConfigError("gate decisions must be a JSON mapping")
    gates = value.get("gates", value)
    if not isinstance(gates, dict):
        raise R1ConfigError("gate decisions 'gates' field must be a mapping")
    return gates
