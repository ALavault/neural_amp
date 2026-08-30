"""Fail-closed orchestration contracts for prospective FSSR-R2-v1."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CAMPAIGN_VERSION = "FSSR-R2-v1"
SCREEN_INITIAL_TRAJECTORIES = 12
SCREEN_ADAA_TRAJECTORIES = 4
DEVELOPMENT_ROBUSTNESS_TRAJECTORIES = 16
INTERNAL_VALIDATION_TRAJECTORIES = 20
FINAL_CONDITIONS = 40
R2_STAGES = (
    "mechanism",
    "screen",
    "teacher",
    "distill",
    "robustness",
    "confirm",
    "benchmark",
    "listen",
)
_RUN_ID = re.compile(
    r"^r2_(?P<stage>[a-z0-9-]+)_(?P<device>[a-z0-9-]+)_"
    r"(?P<family>[a-z0-9-]+)_(?P<aa>[a-z0-9_]+)_seed(?P<seed>0|[1-9][0-9]*)_v1$"
)


class R2CampaignError(RuntimeError):
    """Base class for R2 campaign contract failures."""


class R2ConfigError(R2CampaignError):
    """Raised when preregistered repository configuration is inconsistent."""


class R2AuthorizationError(R2CampaignError):
    """Raised when a sequential R2 stage has not been unlocked."""


@dataclass(frozen=True)
class RunSpec:
    stage: str
    device: str
    family: str
    aa_mode: str
    seed: int

    @property
    def run_id(self) -> str:
        return make_run_id(
            self.stage, self.device, self.family, self.aa_mode, self.seed
        )


def make_run_id(stage: str, device: str, family: str, aa_mode: str, seed: int) -> str:
    """Create the one permitted immutable R2 identifier form."""
    if stage not in R2_STAGES:
        raise ValueError(f"unknown R2 stage: {stage}")
    if seed < 0:
        raise ValueError("R2 seed must be non-negative")
    run_id = f"r2_{stage}_{device}_{family}_{aa_mode}_seed{seed}_v1"
    parsed = _RUN_ID.fullmatch(run_id)
    if parsed is None:
        raise ValueError("R2 run identifier contains an invalid component")
    return run_id


def parse_run_id(run_id: str) -> RunSpec:
    """Parse an R2 ID, rejecting aliases, retries, and later versions."""
    match = _RUN_ID.fullmatch(run_id)
    if match is None or match.group("stage") not in R2_STAGES:
        raise ValueError(f"invalid R2 run_id: {run_id}")
    return RunSpec(
        stage=match.group("stage"),
        device=match.group("device"),
        family=match.group("family"),
        aa_mode=match.group("aa"),
        seed=int(match.group("seed")),
    )


def initial_screen_specs() -> tuple[RunSpec, ...]:
    """Return the frozen eight x2 and four dual-loss A2 trajectories."""
    specs = [
        RunSpec("screen", device, family, "full_island_x2", 0)
        for device in ("fulltone", "bigmuff")
        for family in ("aa-nam-m4", "aa-nam-wright", "aa-fssr-m4", "aa-fssr-wright")
    ]
    specs.extend(
        RunSpec("screen", device, f"a2-{loss}", "off", 0)
        for device in ("fulltone", "bigmuff")
        for loss in ("m4", "wright")
    )
    if len(specs) != SCREEN_INITIAL_TRAJECTORIES:
        raise AssertionError("internal R2 screen matrix count changed")
    return tuple(specs)


def adaa_screen_specs(promoted_loss: str) -> tuple[RunSpec, ...]:
    """Return the four challenger trajectories under the promoted loss."""
    if promoted_loss not in {"m4", "wright"}:
        raise ValueError("promoted R2 loss must be m4 or wright")
    specs = tuple(
        RunSpec("screen", device, f"{family}-{promoted_loss}", "adaa1", 0)
        for device in ("fulltone", "bigmuff")
        for family in ("aa-nam", "aa-fssr")
    )
    if len(specs) != SCREEN_ADAA_TRAJECTORIES:
        raise AssertionError("internal R2 ADAA matrix count changed")
    return specs


def teacher_specs(promoted_family: str) -> tuple[RunSpec, ...]:
    if promoted_family not in {"aa-nam", "aa-fssr"}:
        raise ValueError("promoted R2 family must be aa-nam or aa-fssr")
    return tuple(
        RunSpec("teacher", device, promoted_family, "teacher_x4", 0)
        for device in ("fulltone", "bigmuff")
    )


def development_robustness_specs(
    promoted_family: str, aa_mode: str
) -> tuple[RunSpec, ...]:
    """Return seeds 1--4 for A2 and the final candidate on dev devices."""
    _validate_candidate(promoted_family, aa_mode)
    specs = []
    for device in ("fulltone", "bigmuff"):
        for seed in (1, 2, 3, 4):
            specs.append(RunSpec("robustness", device, "a2", "off", seed))
            specs.append(RunSpec("robustness", device, promoted_family, aa_mode, seed))
    if len(specs) != DEVELOPMENT_ROBUSTNESS_TRAJECTORIES:
        raise AssertionError("internal R2 robustness matrix count changed")
    return tuple(specs)


def internal_validation_specs(
    promoted_family: str, aa_mode: str
) -> tuple[RunSpec, ...]:
    """Return the no-tuning Blackstar/UA five-seed training matrix."""
    _validate_candidate(promoted_family, aa_mode)
    specs = []
    for device in ("blackstar", "ua1176"):
        for seed in range(5):
            specs.append(RunSpec("confirm", device, "a2", "off", seed))
            specs.append(RunSpec("confirm", device, promoted_family, aa_mode, seed))
    if len(specs) != INTERNAL_VALIDATION_TRAJECTORIES:
        raise AssertionError("internal R2 validation matrix count changed")
    return tuple(specs)


def final_condition_keys() -> tuple[tuple[str, str, int], ...]:
    """Return the exact forty A2/candidate device/seed evaluation conditions."""
    conditions = tuple(
        (device, family, seed)
        for device in ("fulltone", "bigmuff", "blackstar", "ua1176")
        for family in ("a2", "candidate")
        for seed in range(5)
    )
    if len(conditions) != FINAL_CONDITIONS:
        raise AssertionError("internal R2 final condition count changed")
    return conditions


def _validate_candidate(family: str, aa_mode: str) -> None:
    if family not in {"aa-nam", "aa-fssr"}:
        raise ValueError("R2 candidate family must be aa-nam or aa-fssr")
    if aa_mode not in {"full_island_x2", "adaa1", "distilled_x2", "distilled_adaa1"}:
        raise ValueError("R2 candidate has an unsupported deployable AA mode")


def _require_equal(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise R2ConfigError(f"{label} must equal {expected!r}, got {value!r}")


def validate_protocol_config(config: dict[str, Any]) -> None:
    """Validate all decision-critical R2 thresholds and matrix declarations."""
    _require_equal(config.get("campaign_version"), CAMPAIGN_VERSION, "campaign_version")
    _require_equal(
        config.get("valid_terminal_verdicts"),
        ["GO-R2", "NO-GO-R2", "INVALID"],
        "valid_terminal_verdicts",
    )
    _require_equal(config.get("external_report_only_locked"), True, "external lock")
    _require_equal(
        config.get("r1_administrative_status"),
        "superseded_by_R2",
        "R1 administrative status",
    )
    decision = config.get("decision", {})
    expected_decision = {
        "confidence_level": 0.95,
        "esr_relative_improvement_lower_bound_minimum": 0.15,
        "asr_reduction_db_minimum": 10.0,
        "devices_won_minimum": 3,
        "mandatory_devices": ["blackstar", "ua1176"],
        "cpp_block_size": 64,
        "cpp_cpu_ratio_maximum": 1.25,
        "added_latency_samples_maximum": 48,
        "mushra_advantage_points_strictly_greater_than": 10.0,
        "mushra_lower_bound_strictly_greater_than": 0.0,
    }
    for name, expected in expected_decision.items():
        _require_equal(decision.get(name), expected, f"decision.{name}")
    mechanism = config.get("mechanism", {})
    _require_equal(len(mechanism.get("fixtures", [])), 6, "mechanism fixture count")
    for name, expected in {
        "spearman_rho_minimum": 0.90,
        "median_asr_gain_db_minimum": 10.0,
        "per_fixture_asr_gain_db_minimum": 6.0,
        "fundamental_complex_error_maximum": 1.0e-5,
    }.items():
        _require_equal(mechanism.get(name), expected, f"mechanism.{name}")
    screening = config.get("screening", {})
    for name, expected in {
        "initial_x2_trajectories": 8,
        "baseline_trajectories": 4,
        "adaa_challenger_trajectories": 4,
        "non_promoted_loss_stop_updates": 5000,
        "promoted_loss_final_updates": 15000,
        "checkpoints": [200, 1000, 5000, 15000],
    }.items():
        _require_equal(screening.get(name), expected, f"screening.{name}")
    confirmation = config.get("confirmation", {})
    _require_equal(
        confirmation.get("final_conditions"), FINAL_CONDITIONS, "final conditions"
    )
    bootstrap = confirmation.get("bootstrap", {})
    _require_equal(
        bootstrap.get("replicates"), 10_000, "confirmation bootstrap replicates"
    )
    _require_equal(bootstrap.get("seed"), 20_260_828, "confirmation bootstrap seed")
    _require_equal(
        bootstrap.get("hierarchy"), ["seed", "source"], "bootstrap hierarchy"
    )
    asr = config.get("asr", {})
    for name, expected in {
        "sample_rate_hz": 48_000,
        "dft_samples": 65_536,
        "k0": [1705, 8191, 12287],
        "amplitudes": [0.10, 0.25, 0.48],
        "frames": 6,
        "analyzed_frame": 6,
        "periodicity_error_db_maximum": -60.0,
    }.items():
        _require_equal(asr.get(name), expected, f"asr.{name}")
    benchmark = config.get("benchmark", {})
    for name, expected in {
        "block_sizes": [1, 16, 64, 128],
        "primary_block_size": 64,
        "repetitions": 30,
        "python_cpp_max_abs_error": 2.0e-5,
    }.items():
        _require_equal(benchmark.get(name), expected, f"benchmark.{name}")
    screening = config.get("screening", {})
    loss_promotion = screening.get("loss_promotion", {})
    for name, expected in {
        "metric": "median_validation_esr_at_5000_updates",
        "aggregation": "across_initial_x2_families_and_devices",
        "rule": "minimum",
        "exact_tie_breaker": "m4",
    }.items():
        _require_equal(
            loss_promotion.get(name), expected, f"screening.loss_promotion.{name}"
        )
    export = config.get("export", {})
    _require_equal(export.get("format"), "fssr-r2-native-v1", "export.format")
    _require_equal(export.get("families"), ["aa-nam", "aa-fssr"], "export.families")
    listening = config.get("listening", {})
    for name, expected in {
        "recruited_participants": 24,
        "retained_participants_minimum": 20,
        "excerpts": 8,
        "excerpts_per_device": 2,
        "excerpt_duration_seconds": 10,
        "preregistered_exclusion_rule": "configs/r2/mushra_exclusion.yaml",
    }.items():
        _require_equal(listening.get(name), expected, f"listening.{name}")
    listening_bootstrap = listening.get("bootstrap", {})
    _require_equal(listening_bootstrap.get("replicates"), 10_000, "MUSHRA replicates")
    _require_equal(listening_bootstrap.get("seed"), 20_260_829, "MUSHRA seed")


def validate_repository_configs(root: Path) -> dict[str, Any]:
    """Validate the preserved R2-v1 protocol, lock, and lineage entry."""
    protocol_path = root / "configs/r2/protocol.yaml"
    lock_path = root / ".codex_campaign/r2/PROTOCOL_LOCK.yaml"
    lineage_path = root / ".codex_campaign/LINEAGES.json"
    for path in (protocol_path, lock_path, lineage_path):
        if not path.is_file():
            raise R2ConfigError(f"missing canonical R2 file: {path}")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    validate_protocol_config(protocol)
    _require_equal(lock.get("campaign_version"), CAMPAIGN_VERSION, "lock campaign")
    _require_equal(lock.get("status"), "frozen_before_capture", "lock status")
    _require_equal(lock.get("scientific_runs_launched"), 0, "lock run count")
    lineage = yaml.safe_load(lineage_path.read_text(encoding="utf-8"))
    r2_lineage = lineage.get("lineages", {}).get("r2", {})
    _require_equal(r2_lineage.get("campaign_version"), CAMPAIGN_VERSION, "lineage")
    _require_equal(
        r2_lineage.get("historical_artifacts_immutable"),
        True,
        "R2-v1 history lock",
    )
    return protocol


_STAGE_REQUIREMENTS = {
    "capture": (("preflight",), ("passed",)),
    "mechanism": (("capture",), ("passed",)),
    "screen": (("mechanism_x2",), ("passed",)),
    "teacher": (("screen",), ("promoted",)),
    "distill": (("teacher", "deployable"), ("passed", "failed_double_gate")),
    "lock": (("final_candidate",), ("selected",)),
    "confirm": (("lock",), ("frozen",)),
    "benchmark": (("confirm_validation",), ("passed",)),
    "sealed_test": (
        ("confirm_validation", "python_cpp_parity", "benchmark"),
        ("passed", "passed", "passed"),
    ),
    "listen": (("sealed_esr", "sealed_asr", "cpu"), ("passed", "passed", "passed")),
    "audit": (("sealed_test", "listening"), ("opened_once", "complete")),
}


def validate_stage_authorization(stage: str, decisions: dict[str, str]) -> None:
    """Require every prior gate literally; missing evidence always fails closed."""
    if stage == "preflight":
        return
    if stage not in _STAGE_REQUIREMENTS:
        raise ValueError(f"unknown R2 authorization stage: {stage}")
    names, expected = _STAGE_REQUIREMENTS[stage]
    for name, status in zip(names, expected, strict=True):
        if decisions.get(name) != status:
            raise R2AuthorizationError(
                f"R2 {stage} requires {name}={status}; got {decisions.get(name)!r}"
            )


def validate_sealed_test_boundary(
    *,
    decisions: dict[str, str],
    internal_tests_open_count: int,
    external_report_only_locked: bool,
) -> None:
    """Authorize the single internal-test opening after parity and timing."""
    validate_stage_authorization("sealed_test", decisions)
    if internal_tests_open_count != 0:
        raise R2AuthorizationError("R2 internal sealed tests may be opened only once")
    if not external_report_only_locked:
        raise R2AuthorizationError("EXTERNAL_REPORT_ONLY must remain locked")
