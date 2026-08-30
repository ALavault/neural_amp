from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn

import fssr_nam.inference.prototype_cli as prototype_cli
from fssr_nam.inference.prototype_cli import (
    align_declared_latency,
    benchmark_python_diagnostic,
    build_parser,
    render_streaming,
)
from fssr_nam.training.prototype import (
    DevelopmentSource,
    PrototypeCheckpointError,
    PrototypeDataBoundaryError,
    PrototypeModelConfig,
    build_checkpoint_payload,
    load_development_sources,
    load_prototype_checkpoint,
    train_development_model,
    validate_checkpoint_payload,
    write_checkpoint_once,
)


class _GainStream(nn.Module):
    latency_samples = 1

    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def stream(self, signal: torch.Tensor) -> torch.Tensor:
        return self.gain * signal

    def detach_stream_state(self) -> None:
        return None


def _write_manifest(root: Path) -> tuple[Path, Path]:
    selected_dir = root / "datasets/raw/internal_r1/fulltone/train"
    selected_dir.mkdir(parents=True)
    samples = np.linspace(-0.5, 0.5, 32, dtype=np.float32)
    sf.write(selected_dir / "input.wav", samples, 48_000, subtype="FLOAT")
    sf.write(selected_dir / "target.wav", samples, 48_000, subtype="FLOAT")
    manifest = {
        "name": "r1_physical",
        "target_sample_rate": 48_000,
        "external_report_only_accessed": False,
        "files": [
            {
                "device": "fulltone",
                "split": "train",
                "source_id": "source-a",
                "tier": "INTERNAL_DEV",
                "sample_rate": 48_000,
                "samples": len(samples),
                "input_path": ("datasets/raw/internal_r1/fulltone/train/input.wav"),
                "target_path": ("datasets/raw/internal_r1/fulltone/train/target.wav"),
            },
            {
                "device": "blackstar",
                "split": "train",
                "source_id": "sealed",
                "tier": "INTERNAL_VALIDATION",
                "sample_rate": 48_000,
                "samples": len(samples),
                "input_path": "audio/blackstar/absent-input.wav",
                "target_path": "audio/blackstar/absent-target.wav",
            },
        ],
    }
    split = {
        "leakage_check": "passed",
        "groups": {
            "fulltone": {
                "train": ["source-a"],
                "validation": ["source-b"],
                "test": ["source-c"],
            }
        },
    }
    manifest_path = root / "manifest.json"
    split_path = root / "split.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    split_path.write_text(json.dumps(split), encoding="utf-8")
    return manifest_path, split_path


def test_development_loader_filters_before_opening_sealed_audio(
    tmp_path: Path,
) -> None:
    manifest_path, split_path = _write_manifest(tmp_path)
    sources = load_development_sources(
        tmp_path,
        manifest_path,
        split_path,
        physical_device="fulltone",
        split="train",
    )
    assert [source.source_id for source in sources] == ["source-a"]
    with pytest.raises(PrototypeDataBoundaryError, match="Fulltone/BigMuff"):
        load_development_sources(
            tmp_path,
            manifest_path,
            split_path,
            physical_device="blackstar",
            split="train",
        )


def test_training_resets_at_each_source_boundary() -> None:
    model = _GainStream()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    sources = tuple(
        DevelopmentSource(
            physical_device="fulltone",
            split="train",
            source_id=f"source-{index}",
            input=np.linspace(-0.2, 0.2, 8, dtype=np.float32),
            target=np.linspace(-0.1, 0.3, 8, dtype=np.float32),
        )
        for index in range(2)
    )
    result = train_development_model(
        model,
        sources,
        optimizer=optimizer,
        compute_device=torch.device("cpu"),
        seed=0,
        updates=2,
        checkpoint_updates=(),
        chunk_samples=8,
        common_preroll_samples=0,
        projection_gain_weight=0.05,
        gradient_clip_norm=1.0,
    )
    assert result.updates == 2
    assert result.source_resets == 2
    assert model.reset_count == 2


def test_checkpoint_round_trip_is_versioned_and_strict(tmp_path: Path) -> None:
    config = PrototypeModelConfig(profile="slim")
    model = config.build()
    payload = build_checkpoint_payload(
        model,
        config,
        physical_device="fulltone",
        seed=0,
        update=500,
        source_ids=["source-a"],
    )
    path = tmp_path / "checkpoint.pt"
    write_checkpoint_once(path, payload)
    loaded = load_prototype_checkpoint(path)
    assert loaded.model_config == config
    assert loaded.model.latency_samples == payload["latency_samples"]
    assert loaded.training["source_ids"] == ["source-a"]

    divergent = dict(payload)
    divergent["unexpected"] = True
    with pytest.raises(PrototypeCheckpointError, match="envelope"):
        validate_checkpoint_payload(divergent)


def test_render_resets_once_and_declared_latency_alignment_is_exact() -> None:
    model = _GainStream()
    signal = np.arange(12, dtype=np.float32)
    rendered = render_streaming(
        model, signal, compute_device=torch.device("cpu"), block_size=5
    )
    assert np.array_equal(rendered, signal)
    assert model.reset_count == 1

    predicted, target = align_declared_latency(
        np.arange(5_010, dtype=np.float32),
        np.arange(5_010, dtype=np.float32),
        latency_samples=10,
        common_preroll_samples=100,
    )
    assert predicted[0] == 110
    assert target[0] == 100
    assert predicted.shape == target.shape == (4_900,)


def test_python_benchmark_reports_compute_over_audio_rtf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter((0, 1_000_000, 2_000_000, 3_000_000))
    monkeypatch.setattr(prototype_cli.time, "perf_counter_ns", lambda: next(clock))
    result = benchmark_python_diagnostic(
        _GainStream(),
        compute_device=torch.device("cpu"),
        block_size=128,
        repetitions=2,
        audio_seconds_per_repetition=128 / 48_000,
        warmup_blocks=0,
    )
    assert result["median_rtf"] == pytest.approx(0.375)
    assert result["p95_rtf"] == pytest.approx(0.375)
    assert result["native_required_for_gate"] is True
    assert result["scientific_eligible"] is False
    assert result["gate_decision"] is None


def test_cli_exposes_four_commands_and_dev_only_choices() -> None:
    parser = build_parser()
    assert (
        parser.parse_args(
            [
                "train",
                "--run-id",
                "sota_dev",
                "--physical-device",
                "fulltone",
                "--seed",
                "0",
            ]
        ).operation
        == "train"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "evaluate",
                "--checkpoint",
                "model.pt",
                "--physical-device",
                "blackstar",
                "--split",
                "test",
                "--output",
                "metrics.json",
            ]
        )
