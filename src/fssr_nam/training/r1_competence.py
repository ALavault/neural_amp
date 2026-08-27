"""Locked controls for the native-rate Wright competence gate.

This module contains only protocol validation and gate arithmetic.  Training and
inference remain in :mod:`fssr_nam.training.wright`, while run reservation and
terminal registration remain in :mod:`fssr_nam.campaign.r1`.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fssr_nam.campaign.r1 import R1ConfigError, validate_stage_config


class CompetenceProtocolError(R1ConfigError):
    """The Wright competence declaration differs from the locked recipe."""


_LOCKED_FIELDS: dict[str, object] = {
    "schema_version": 1,
    "campaign_version": "FSSR-R1-v1",
    "stage": "competence",
    "tier": "INTERNAL_DEV",
    "device": "bigmuff",
    "setting": "S050_V100",
    "sample_rate": 44_100,
    "model": "lstm64",
    "loss_mode": "wright",
    "checkpoint_steps": [],
    "phase_schedule": [
        {
            "name": "official_wright_training",
            "start_epoch": 1,
            "end_epoch": 2_000,
            "trainable": ["all"],
        }
    ],
    "receptive_field": "stateful",
    "seeds": [0, 1, 2],
    "conditional_seed_rule": {
        "first_seed": 0,
        "additional_seeds": [1, 2],
        "launch_if_seed0_esr_test_at_most": 0.15,
    },
    "segment_samples": 22_050,
    "warmup_samples": 200,
    "tbptt_samples": 1_000,
    "batch_size": 50,
    "maximum_epochs": 2_000,
    "validation_frequency_epochs": 2,
    "optimizer": {
        "name": "Adam",
        "learning_rate": 5.0e-3,
        "weight_decay": 1.0e-4,
    },
    "scheduler": {
        "name": "ReduceLROnPlateau",
        "mode": "min",
        "factor": 0.5,
        "patience_validations": 5,
    },
    "early_stopping": {
        "patience_validations": 25,
        "comparison": "strict_improvement",
    },
    "loss": {
        "preemphasis": [-0.85, 1.0],
        "esr_preemphasized_weight": 0.75,
        "dc_weight": 0.25,
        "epsilon": 1.0e-5,
    },
    "checkpoint_policy": "lowest_validation_wright_loss",
    "promotion_gate": {
        "median_esr_test_at_most": 0.15,
        "maximum_seed_esr_test_at_most": 0.20,
        "required_seeds": 3,
    },
    "failure_action": "stop_r1_and_audit_loss_tbptt_alignment_weight_conversion",
}


def _exactly_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _exactly_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exactly_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def validate_competence_protocol(config: Mapping[str, Any]) -> None:
    """Reject any drift from the preregistered Wright competence recipe."""
    validate_stage_config(config, "competence")
    for field, expected in _LOCKED_FIELDS.items():
        actual = config.get(field)
        if not _exactly_equal(actual, expected):
            raise CompetenceProtocolError(
                f"r1_competence.{field} must be exactly {expected!r}, got {actual!r}"
            )


def _checked_esr(value: object, *, seed: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"seed {seed} ESR must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"seed {seed} ESR must be finite and non-negative")
    return result


def seed_zero_allows_additional_runs(
    config: Mapping[str, Any], seed_zero_esr: float
) -> bool:
    """Apply the inclusive seed-0 continuation threshold."""
    validate_competence_protocol(config)
    esr = _checked_esr(seed_zero_esr, seed=0)
    limit = float(config["conditional_seed_rule"]["launch_if_seed0_esr_test_at_most"])
    return esr <= limit


@dataclass(frozen=True, slots=True)
class CompetenceGateDecision:
    """Terminal competence decision backed by the exact observed seed set."""

    status: str
    seed_esr: tuple[tuple[int, float], ...]
    seed_zero_continuation: bool
    median_esr: float | None
    maximum_esr: float | None
    reason: str

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "passed": self.passed,
            "seed_esr": {str(seed): esr for seed, esr in self.seed_esr},
            "seed_zero_continuation": self.seed_zero_continuation,
            "median_esr": self.median_esr,
            "maximum_esr": self.maximum_esr,
            "reason": self.reason,
        }


def evaluate_competence_gate(
    config: Mapping[str, Any], seed_esr: Mapping[int, float]
) -> CompetenceGateDecision:
    """Evaluate seed-0 continuation, then the locked three-seed promotion gate."""
    validate_competence_protocol(config)
    if 0 not in seed_esr:
        raise ValueError("competence gate requires seed 0")
    observed: dict[int, float] = {}
    for seed, value in seed_esr.items():
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("competence seed keys must be integers")
        observed[seed] = _checked_esr(value, seed=seed)
    if set(observed) - {0, 1, 2}:
        raise ValueError("competence gate contains an undeclared seed")

    seed_zero_limit = float(
        config["conditional_seed_rule"]["launch_if_seed0_esr_test_at_most"]
    )
    if observed[0] > seed_zero_limit:
        if set(observed) != {0}:
            raise ValueError("additional seeds ran despite the failed seed-0 gate")
        return CompetenceGateDecision(
            status="failed",
            seed_esr=((0, observed[0]),),
            seed_zero_continuation=False,
            median_esr=None,
            maximum_esr=None,
            reason=(
                f"seed 0 test ESR {observed[0]:.9g} exceeds continuation limit "
                f"{seed_zero_limit:.9g}"
            ),
        )

    required = tuple(int(seed) for seed in config["seeds"])
    if set(observed) != set(required):
        missing = sorted(set(required) - set(observed))
        raise ValueError(
            f"passing seed 0 requires all preregistered seeds; missing {missing}"
        )
    ordered = tuple((seed, observed[seed]) for seed in required)
    values = [value for _, value in ordered]
    median = float(statistics.median(values))
    maximum = max(values)
    median_limit = float(config["promotion_gate"]["median_esr_test_at_most"])
    maximum_limit = float(config["promotion_gate"]["maximum_seed_esr_test_at_most"])
    passed = median <= median_limit and maximum <= maximum_limit
    reason = (
        f"median={median:.9g} (limit {median_limit:.9g}); "
        f"maximum={maximum:.9g} (limit {maximum_limit:.9g})"
    )
    return CompetenceGateDecision(
        status="passed" if passed else "failed",
        seed_esr=ordered,
        seed_zero_continuation=True,
        median_esr=median,
        maximum_esr=maximum,
        reason=reason,
    )
