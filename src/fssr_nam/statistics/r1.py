"""Paired hierarchical confirmation statistics for FSSR-R1."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EsrObservation:
    device: str
    source: str
    seed: int
    family: str
    esr: float


@dataclass(frozen=True)
class CostObservation:
    device: str
    seed: int
    family: str
    block_size: int
    ns_per_sample: float
    latency_samples: int


@dataclass(frozen=True)
class BootstrapSummary:
    observed: float
    lower_95: float
    upper_95: float
    one_sided_upper_95: float


@dataclass(frozen=True)
class ConfirmationDecision:
    verdict: str
    h1_passed: bool
    h2_passed: bool
    esr_improvement: BootstrapSummary
    esr_ratio: BootstrapSummary
    device_improvements: dict[str, float]
    devices_won: list[str]
    cpp_cost_ratio: float
    cpp_cpu_reduction: float
    added_latency_samples: int
    bootstrap_replicates: int
    bootstrap_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_positive(value: float, label: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _paired_esr(
    observations: Iterable[EsrObservation],
    *,
    baseline_family: str,
    candidate_family: str,
) -> dict[str, dict[int, dict[str, tuple[float, float]]]]:
    indexed: dict[tuple[str, int, str, str], float] = {}
    for observation in observations:
        if observation.family not in {baseline_family, candidate_family}:
            continue
        key = (
            observation.device,
            int(observation.seed),
            observation.source,
            observation.family,
        )
        if key in indexed:
            raise ValueError(f"duplicate ESR observation: {key}")
        indexed[key] = _finite_positive(observation.esr, "ESR")

    baseline_keys = {key[:3] for key in indexed if key[3] == baseline_family}
    candidate_keys = {key[:3] for key in indexed if key[3] == candidate_family}
    if not baseline_keys or baseline_keys != candidate_keys:
        missing_candidate = sorted(baseline_keys - candidate_keys)
        missing_baseline = sorted(candidate_keys - baseline_keys)
        raise ValueError(
            "ESR pairing is incomplete; "
            f"missing candidate={missing_candidate}, "
            f"missing baseline={missing_baseline}"
        )

    paired: dict[str, dict[int, dict[str, tuple[float, float]]]] = {}
    for device, seed, source in sorted(baseline_keys):
        paired.setdefault(device, {}).setdefault(seed, {})[source] = (
            indexed[(device, seed, source, baseline_family)],
            indexed[(device, seed, source, candidate_family)],
        )
    return paired


def _device_medians(
    paired: dict[str, dict[int, dict[str, tuple[float, float]]]],
) -> dict[str, tuple[float, float]]:
    medians: dict[str, tuple[float, float]] = {}
    for device, seeds in paired.items():
        pairs = [pair for sources in seeds.values() for pair in sources.values()]
        baseline = float(np.median([pair[0] for pair in pairs]))
        candidate = float(np.median([pair[1] for pair in pairs]))
        medians[device] = (baseline, candidate)
    return medians


def _global_metrics(
    device_medians: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    baseline = float(np.median([pair[0] for pair in device_medians.values()]))
    candidate = float(np.median([pair[1] for pair in device_medians.values()]))
    ratio = candidate / baseline
    return 1.0 - ratio, ratio


def paired_hierarchical_bootstrap(
    observations: Iterable[EsrObservation],
    *,
    baseline_family: str = "a2",
    candidate_family: str = "r1_final",
    replicates: int = 10_000,
    seed: int = 20_260_827,
) -> tuple[BootstrapSummary, BootstrapSummary, dict[str, float]]:
    """Resample paired seeds, then paired sources within each sampled seed.

    Devices are the fixed target population and are not resampled. Windows are
    deliberately absent from the interface and therefore cannot become
    pseudo-independent observations.
    """
    if replicates < 1:
        raise ValueError("replicates must be positive")
    paired = _paired_esr(
        observations,
        baseline_family=baseline_family,
        candidate_family=candidate_family,
    )
    observed_medians = _device_medians(paired)
    observed_improvement, observed_ratio = _global_metrics(observed_medians)
    device_improvements = {
        device: 1.0 - candidate / baseline
        for device, (baseline, candidate) in observed_medians.items()
    }

    generator = np.random.default_rng(seed)
    improvements = np.empty(replicates, dtype=np.float64)
    ratios = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        resampled_medians: dict[str, tuple[float, float]] = {}
        for device, seeds in paired.items():
            seed_ids = np.asarray(sorted(seeds), dtype=np.int64)
            sampled_seed_ids = generator.choice(
                seed_ids, size=seed_ids.size, replace=True
            )
            baseline_values: list[float] = []
            candidate_values: list[float] = []
            for sampled_seed in sampled_seed_ids:
                sources = seeds[int(sampled_seed)]
                source_ids = np.asarray(sorted(sources), dtype=object)
                sampled_source_ids = generator.choice(
                    source_ids, size=source_ids.size, replace=True
                )
                for sampled_source in sampled_source_ids:
                    baseline, candidate = sources[str(sampled_source)]
                    baseline_values.append(baseline)
                    candidate_values.append(candidate)
            resampled_medians[device] = (
                float(np.median(baseline_values)),
                float(np.median(candidate_values)),
            )
        improvements[replicate], ratios[replicate] = _global_metrics(resampled_medians)

    improvement_summary = BootstrapSummary(
        observed=observed_improvement,
        lower_95=float(np.quantile(improvements, 0.025)),
        upper_95=float(np.quantile(improvements, 0.975)),
        one_sided_upper_95=float(np.quantile(improvements, 0.95)),
    )
    ratio_summary = BootstrapSummary(
        observed=observed_ratio,
        lower_95=float(np.quantile(ratios, 0.025)),
        upper_95=float(np.quantile(ratios, 0.975)),
        one_sided_upper_95=float(np.quantile(ratios, 0.95)),
    )
    return improvement_summary, ratio_summary, device_improvements


def _paired_cost_ratio(
    observations: Iterable[CostObservation],
    *,
    baseline_family: str,
    candidate_family: str,
    block_size: int,
) -> tuple[float, int]:
    indexed: dict[tuple[str, int, str], CostObservation] = {}
    for observation in observations:
        if observation.block_size != block_size:
            continue
        if observation.family not in {baseline_family, candidate_family}:
            continue
        key = (observation.device, int(observation.seed), observation.family)
        if key in indexed:
            raise ValueError(f"duplicate CPU observation: {key}")
        _finite_positive(observation.ns_per_sample, "CPU time")
        if observation.latency_samples < 0:
            raise ValueError("latency must be non-negative")
        indexed[key] = observation

    baseline_keys = {key[:2] for key in indexed if key[2] == baseline_family}
    candidate_keys = {key[:2] for key in indexed if key[2] == candidate_family}
    if not baseline_keys or baseline_keys != candidate_keys:
        raise ValueError("CPU pairing is incomplete at the required block size")
    baseline_values = [
        indexed[(*key, baseline_family)].ns_per_sample for key in sorted(baseline_keys)
    ]
    candidate_values = [
        indexed[(*key, candidate_family)].ns_per_sample for key in sorted(baseline_keys)
    ]
    ratio = float(np.median(candidate_values) / np.median(baseline_values))
    added_latency = max(
        indexed[(*key, candidate_family)].latency_samples
        - indexed[(*key, baseline_family)].latency_samples
        for key in baseline_keys
    )
    return ratio, int(added_latency)


def decide_confirmation(
    esr_observations: Iterable[EsrObservation],
    cost_observations: Iterable[CostObservation],
    *,
    baseline_family: str = "a2",
    candidate_family: str = "r1_final",
    replicates: int = 10_000,
    seed: int = 20_260_827,
) -> ConfirmationDecision:
    improvement, ratio, device_improvements = paired_hierarchical_bootstrap(
        esr_observations,
        baseline_family=baseline_family,
        candidate_family=candidate_family,
        replicates=replicates,
        seed=seed,
    )
    required_devices = {"fulltone", "bigmuff", "blackstar", "ua1176"}
    if set(device_improvements) != required_devices:
        raise ValueError(
            "confirmation requires exactly fulltone, bigmuff, blackstar, and ua1176"
        )
    devices_won = sorted(
        device for device, value in device_improvements.items() if value > 0.0
    )
    cost_ratio, added_latency = _paired_cost_ratio(
        cost_observations,
        baseline_family=baseline_family,
        candidate_family=candidate_family,
        block_size=64,
    )
    cpu_reduction = 1.0 - cost_ratio
    mandatory_wins = {"blackstar", "ua1176"}.issubset(devices_won)
    h1_passed = bool(
        improvement.observed >= 0.05
        and improvement.lower_95 > 0.0
        and len(devices_won) >= 3
        and mandatory_wins
        and 0.95 <= cost_ratio <= 1.05
    )
    h2_passed = bool(
        ratio.one_sided_upper_95 <= 1.05
        and cpu_reduction >= 0.25
        and added_latency <= 0
    )
    verdict = "GO-A" if h1_passed else "GO-B" if h2_passed else "NO-GO-R1"
    return ConfirmationDecision(
        verdict=verdict,
        h1_passed=h1_passed,
        h2_passed=h2_passed,
        esr_improvement=improvement,
        esr_ratio=ratio,
        device_improvements=device_improvements,
        devices_won=devices_won,
        cpp_cost_ratio=cost_ratio,
        cpp_cpu_reduction=cpu_reduction,
        added_latency_samples=added_latency,
        bootstrap_replicates=replicates,
        bootstrap_seed=seed,
    )
