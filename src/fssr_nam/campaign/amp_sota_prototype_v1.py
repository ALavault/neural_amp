"""Prospective contract for the metrics-first SOTA prototype campaign."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

BASE_CAMPAIGN_VERSION = "AMP-SOTA-PROTOTYPE-v1"
CAMPAIGN_VERSION = "AMP-SOTA-PROTOTYPE-v1.1"
CAMPAIGN_KEY = "amp_sota_prototype_v1_1"
PROTOCOL_PATH = Path("configs/amp_sota_prototype_v1/protocol.yaml")
AMENDMENT_PATH = Path("configs/amp_sota_prototype_v1_1/protocol.yaml")
CAMPAIGN_PATH = Path(".codex_campaign/amp_sota_prototype_v1_1")
DEVELOPMENT_DEVICES = ("fulltone", "bigmuff")
CONFIRMATION_DEVICES = ("blackstar", "ua1176")
PRIMARY_BASELINES = ("nam_a2_full", "nam_a2_lite")
SECONDARY_BASELINES = (
    "wright_lstm64",
    "nablafx_tcn_tfilm",
    "nablafx_s4_tfilm",
)


class SotaPrototypeConfigError(RuntimeError):
    """Raised when the prospective prototype contract is incomplete or changed."""


class SotaPrototypeAuthorizationError(RuntimeError):
    """Raised when a stage is attempted before its prerequisite gate."""


def _require_equal(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise SotaPrototypeConfigError(
            f"{label} changed: expected {expected!r}, got {value!r}"
        )


def validate_protocol_config(protocol: Mapping[str, Any]) -> None:
    """Reject decision-bearing drift before any scientific run."""
    _require_equal(protocol.get("schema_version"), 1, "schema_version")
    _require_equal(protocol.get("campaign_version"), BASE_CAMPAIGN_VERSION, "campaign")
    _require_equal(
        protocol.get("campaign_key"), "amp_sota_prototype_v1", "campaign_key"
    )
    _require_equal(protocol.get("parent_campaign"), "AMP-QUALITY-ARCH-v3", "parent")
    _require_equal(protocol.get("sample_rate_hz"), 48_000, "sample_rate")
    _require_equal(protocol.get("channels"), 1, "channels")
    _require_equal(protocol.get("precision"), "float32", "precision")

    boundaries = protocol.get("boundaries", {})
    for key in (
        "parent_runs_resumed",
        "fm9_outputs_allowed",
        "human_feedback_allowed_for_selection",
    ):
        _require_equal(boundaries.get(key), False, f"boundaries.{key}")
    for key in (
        "parent_results_used_for_hypotheses_only",
        "fm9_protocol_authoring_allowed",
        "external_report_only_locked",
    ):
        _require_equal(boundaries.get(key), True, f"boundaries.{key}")
    _require_equal(
        boundaries.get("physical_development_outputs_allowed"),
        list(DEVELOPMENT_DEVICES),
        "boundaries.development",
    )
    _require_equal(
        boundaries.get("physical_confirmation_outputs_locked"),
        list(CONFIRMATION_DEVICES),
        "boundaries.confirmation",
    )

    data = protocol.get("data_roles", {})
    _require_equal(data.get("development"), list(DEVELOPMENT_DEVICES), "data.dev")
    _require_equal(
        data.get("confirmation"), list(CONFIRMATION_DEVICES), "data.confirmation"
    )
    _require_equal(
        data.get("source_file_disjoint_splits_required"), True, "data.splits"
    )
    _require_equal(
        data.get("noncommercial_data_claim_boundary"),
        "research_only",
        "data.license",
    )

    baselines = protocol.get("baselines", {})
    _require_equal(
        baselines.get("primary"), list(PRIMARY_BASELINES), "baselines.primary"
    )
    _require_equal(
        baselines.get("secondary"), list(SECONDARY_BASELINES), "baselines.secondary"
    )
    _require_equal(baselines.get("nam_trainer_version"), "v0.13.0", "baselines.nam")
    _require_equal(
        baselines.get("same_pairs_and_splits_required"),
        True,
        "baselines.pairs",
    )

    screen = protocol.get("mechanism_screen", {})
    _require_equal(screen.get("evidence_tier"), "SYNTHETIC", "screen.tier")
    _require_equal(screen.get("evaluate_one_axis_at_a_time"), True, "screen.axes")
    _require_equal(screen.get("maximum_promotions_per_axis"), 1, "screen.promotions")
    axes = screen.get("axes", {})
    _require_equal(
        axes.get("approximant", {}).get("candidates"),
        ["quintic_hermite_c2", "safe_rational_4_3"],
        "screen.approximants",
    )
    _require_equal(
        axes.get("slow_control", {}).get("lookahead_samples_maximum"),
        0,
        "screen.lookahead",
    )
    _require_equal(
        axes.get("resampler", {}).get("candidates"),
        ["equiripple_halfband_polyphase"],
        "screen.resampler",
    )

    search = protocol.get("architecture_search", {})
    _require_equal(search.get("anchor"), "slow_long_tcn_x2", "search.anchor")
    _require_equal(search.get("maximum_rounds"), 3, "search.rounds")
    _require_equal(search.get("seeds"), [0, 1, 2], "search.seeds")
    _require_equal(
        search.get("stop_after_consecutive_rounds_below_improvement"),
        2,
        "search.stop",
    )
    _require_equal(
        search.get("result_dependent_retry_within_round_allowed"),
        False,
        "search.retry",
    )

    confirmation = protocol.get("confirmation", {})
    _require_equal(confirmation.get("seeds"), [0, 1, 2, 3, 4], "confirmation.seeds")
    _require_equal(
        confirmation.get("devices"), list(CONFIRMATION_DEVICES), "confirmation.devices"
    )
    _require_equal(
        confirmation.get("median_esr_improvement_minimum"), 0.10, "confirmation.esr"
    )
    _require_equal(
        confirmation.get("metric_relative_regression_maximum"),
        0.05,
        "confirmation.noninferiority",
    )

    prototype = protocol.get("prototype", {})
    _require_equal(prototype.get("interface"), "python_cli", "prototype.interface")
    _require_equal(
        prototype.get("block_parity_max_absolute_error"), 2.0e-5, "prototype.parity"
    )
    _require_equal(prototype.get("latency_samples_maximum"), 64, "prototype.latency")
    _require_equal(prototype.get("benchmark_block_size"), 128, "prototype.block")

    resources = protocol.get("resource_budget", {})
    _require_equal(resources.get("campaign_gpu_hours_maximum"), 324, "resources.gpu")
    _require_equal(resources.get("campaign_disk_gib_maximum"), 50, "resources.disk")


def load_protocol(root: Path) -> dict[str, Any]:
    path = root / PROTOCOL_PATH
    if not path.is_file():
        raise SotaPrototypeConfigError(f"missing prototype protocol: {path}")
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise SotaPrototypeConfigError("prototype protocol must be a mapping")
    validate_protocol_config(protocol)
    return protocol


def validate_amendment_config(amendment: Mapping[str, Any]) -> None:
    """Reject drift in the v1.1 measurement definitions."""
    _require_equal(amendment.get("schema_version"), 1, "amendment schema")
    _require_equal(amendment.get("campaign_version"), CAMPAIGN_VERSION, "amendment")
    _require_equal(amendment.get("thresholds_changed_from_v1"), False, "thresholds")
    observation = amendment.get("observation", {})
    _require_equal(
        observation.get("unit"), ["device", "seed", "source_file"], "observation unit"
    )
    _require_equal(
        observation.get("common_preroll_samples_after_alignment"), 14_400, "preroll"
    )
    metrics = amendment.get("metrics", {})
    _require_equal(
        metrics.get("mrstft", {}).get("fft_sizes"), [256, 1024, 4096], "MRSTFT"
    )
    log_mel = metrics.get("log_mel", {})
    for key, expected in {
        "sample_rate_hz": 48_000,
        "fft_size": 2048,
        "hop_size": 512,
        "bands": 128,
        "minimum_hz": 20,
        "maximum_hz": 24_000,
        "epsilon": 1.0e-10,
    }.items():
        _require_equal(log_mel.get(key), expected, f"log_mel.{key}")
    bootstrap = amendment.get("bootstrap", {})
    _require_equal(bootstrap.get("replicates"), 10_000, "bootstrap replicates")
    _require_equal(bootstrap.get("seed"), 20_260_830, "bootstrap seed")
    _require_equal(bootstrap.get("point_minimum"), 0.10, "bootstrap point")
    alias = amendment.get("confirmation_alias_proxy", {})
    _require_equal(alias.get("dft_samples"), 65_536, "alias DFT")
    _require_equal(alias.get("k0_values"), [1705, 8191, 12287], "alias k0")
    _require_equal(alias.get("amplitudes"), [0.10, 0.25, 0.48], "alias amplitudes")
    approximant = amendment.get("mechanism_screen", {}).get("approximant", {})
    _require_equal(approximant.get("train_grid"), "chebyshev", "screen train grid")
    _require_equal(approximant.get("train_samples"), 8193, "screen train samples")
    _require_equal(approximant.get("evaluation_samples"), 65_537, "screen eval")
    _require_equal(approximant.get("updates"), 2000, "screen updates")
    _require_equal(approximant.get("learning_rate"), 0.002, "screen LR")
    cost = amendment.get("native_cost", {})
    _require_equal(cost.get("block_size"), 128, "native block")
    _require_equal(cost.get("repetitions"), 30, "native repetitions")
    _require_equal(cost.get("rtf"), "compute_time_over_audio_duration", "RTF")
    training = amendment.get("training", {})
    _require_equal(training.get("profile"), "max", "training profile")
    _require_equal(training.get("seeds"), [0, 1, 2], "training seeds")
    _require_equal(training.get("learning_rate"), 0.001, "learning rate")
    _require_equal(training.get("training_chunk_samples"), 8192, "train chunk")
    _require_equal(
        training.get("checkpoint_updates"),
        [500, 1000, 2000, 5000, 10000, 15000],
        "training checkpoints",
    )
    _require_equal(
        amendment.get("boundaries", {}).get("formal_rerun_allowed"),
        False,
        "formal rerun",
    )


def validate_repository_state(
    root: Path, *, require_frozen: bool = True
) -> dict[str, Any]:
    protocol = load_protocol(root)
    if require_frozen:
        lock_path = root / ".codex_campaign/amp_sota_prototype_v1/PROTOCOL_LOCK.yaml"
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        _require_equal(lock, protocol, "prototype protocol lock")
        amendment = yaml.safe_load((root / AMENDMENT_PATH).read_text(encoding="utf-8"))
        amendment_lock = yaml.safe_load(
            (root / CAMPAIGN_PATH / "PROTOCOL_LOCK.yaml").read_text(encoding="utf-8")
        )
        _require_equal(amendment_lock, amendment, "v1.1 protocol lock")
        validate_amendment_config(amendment)
        _require_equal(
            amendment.get("exploratory_outputs_allowed_for_selection"),
            False,
            "exploratory selection",
        )
    parent_verdict = root / ".codex_campaign/amp_quality_arch_v3/VERDICT.json"
    if not parent_verdict.is_file():
        raise SotaPrototypeConfigError("v3 terminal verdict is required")
    for relative in (
        Path(protocol["data_roles"]["source_config"]),
        Path(protocol["fm9_protocol"]["config"]),
    ):
        if not (root / relative).is_file():
            raise SotaPrototypeConfigError(
                f"missing required protocol input: {relative}"
            )
    return protocol


def validate_stage_authorization(stage: str, decisions: Mapping[str, str]) -> None:
    prerequisites = {
        "preflight": (),
        "screen": (("preflight", "passed"),),
        "development": (("screen", "passed"),),
        "candidate_lock": (("development", "passed"),),
        "native": (("candidate_lock", "passed"),),
        "confirmation": (("native", "passed"),),
        "prototype": (("confirmation", "passed"),),
        "fm9_protocol": (),
    }
    if stage not in prerequisites:
        raise SotaPrototypeAuthorizationError(f"unknown prototype stage: {stage}")
    for required_stage, required_status in prerequisites[stage]:
        if decisions.get(required_stage) != required_status:
            raise SotaPrototypeAuthorizationError(
                f"{stage} requires {required_stage}={required_status}"
            )
