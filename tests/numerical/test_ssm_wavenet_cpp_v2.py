"""Product native FiLM parity; synthetic fixtures need no captured audio."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import test_ssm_wavenet_cpp_parity as v1
import torch

from fssr_nam.models.ssm_wavenet import SSMWaveNet, zoh_to_foh

EXPORT = v1.EXPORT
runner = v1.runner


@torch.no_grad()
def reference_events(
    model: SSMWaveNet, x: np.ndarray, events: list[dict]
) -> np.ndarray:
    """FFT each SSM over the whole signal; apply time-varying FiLM between layers.

    This preserves history across knob changes without sharing the native recurrence.
    """
    model = copy.deepcopy(model)
    zoh_to_foh(model)
    model = model.double().eval()
    h = model.input_conv(torch.from_numpy(x).double()[None, None])
    films = [
        model.film_net(torch.tensor([event["controls"]], dtype=h.dtype)).chunk(
            len(model.blocks), dim=1
        )
        for event in events
    ]
    total = torch.zeros_like(h)
    for index, block in enumerate(model.blocks):
        v = block.pre_gate(block.ssm(h))
        for i, event in enumerate(events):
            end = events[i + 1]["sample"] if i + 1 < len(events) else len(x)
            scale, shift = films[i][index].unsqueeze(-1).chunk(2, dim=1)
            segment = v[..., event["sample"] : end]
            segment.copy_((1 + scale) * segment + shift)
        a, b = v.chunk(2, dim=1)
        gated = a.tanh() * b.sigmoid()
        total += block.skip_conv(gated)
        h = h + block.res_conv(gated)
    return model.output_net(total).numpy().ravel()


def native_events(
    runner: Path,
    model_path: Path,
    x: np.ndarray,
    events: list[dict],
    folder: Path,
    blocks: str = "1,7,64,333",
) -> tuple[np.ndarray, np.ndarray]:
    x.tofile(folder / "in.f32")
    (folder / "controls.json").write_text(json.dumps(events))
    out = folder / "out.f32"
    subprocess.run(
        [
            str(runner),
            str(model_path),
            str(folder / "in.f32"),
            str(out),
            blocks,
            "48000",
            str(folder / "controls.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return np.fromfile(out, dtype=np.float32), np.fromfile(
        str(out) + ".reset.f32", dtype=np.float32
    )


@pytest.mark.parametrize("head_act", ["relu", "leaky"])
def test_film_parity_controls_blocks_reset_causality(
    tmp_path: Path, runner: Path, head_act: str
) -> None:
    torch.manual_seed(42)
    model = SSMWaveNet(
        num_blocks=2,
        channels=4,
        state_dim=3,
        discretization="zoh",
        num_controls=2,
        head_act=head_act,
    ).eval()
    with torch.no_grad():
        model.film_net[2].weight.normal_(std=0.3)
        model.film_net[2].bias.normal_(std=0.1)
        # Keep both head branches observable; an untrained ReLU head can be closed.
        for block in model.blocks:
            block.skip_conv.bias.copy_(torch.tensor([-0.5, 0.5, -0.5, 0.5]))
        model.output_net[1].weight.copy_(torch.eye(4)[..., None])
        model.output_net[1].bias.copy_(torch.tensor([-0.5, 0.5, -0.5, 0.5]))
        model.output_net[3].weight.fill_(0.25)
    path = EXPORT.export_model(model, tmp_path / "model.json")
    x = (np.random.default_rng(42).normal(size=8192) * 0.2).astype(np.float32)
    events = [
        {"sample": 0, "controls": [0.0, 1.0]},
        {"sample": 2107, "controls": [0.5, 0.25]},
        {"sample": 5071, "controls": [1.0, 0.0]},
    ]
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        expected = reference_events(model, x, events)
        # Constant-command reference agrees with the ordinary model.forward path.
        ref = copy.deepcopy(model)
        zoh_to_foh(ref)
        ref.double()
        with torch.no_grad():
            direct = ref(
                torch.from_numpy(x).double()[None, None],
                torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            )
        np.testing.assert_allclose(
            reference_events(model, x, events[:1]),
            direct.numpy().ravel(),
            atol=1e-12,
            rtol=0,
        )
    finally:
        torch.set_num_threads(threads)
    one, reset = native_events(runner, path, x, events, tmp_path, str(len(x)))
    irregular, _ = native_events(runner, path, x, events, tmp_path)
    assert one.shape == expected.shape == x.shape
    assert np.isfinite(one).all()
    error = float(np.max(np.abs(one - expected)))
    print(f"{head_act=} max_abs={error:.9g}")
    assert error <= 3e-6  # Float32 coefficients/state versus float64 FFT, as in v1.
    np.testing.assert_array_equal(one, reset)
    np.testing.assert_array_equal(one, irregular)
    prefix, _ = native_events(runner, path, x[:1024], events[:1], tmp_path)
    np.testing.assert_array_equal(prefix, one[:1024])
    constant, _ = native_events(runner, path, x, events[:1], tmp_path)
    np.testing.assert_array_equal(constant[:2107], one[:2107])
    assert np.max(np.abs(constant[2107:] - one[2107:])) > 1e-3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cond": "poles"},
        {"cond_head": "additive", "cond_hidden": [16, 16]},
        {"cond_hidden": [16]},
        {"act_type": "odd", "head_act": "tanh"},
        {"head_act": "tanh"},
    ],
)
def test_v2_export_rejects_unsupported(tmp_path: Path, kwargs: dict) -> None:
    model = SSMWaveNet(
        num_blocks=1,
        channels=2,
        state_dim=2,
        num_controls=2,
        discretization="zoh",
        **kwargs,
    )
    with pytest.raises(ValueError):
        EXPORT.export_model(model, tmp_path / "bad.json")


@pytest.mark.parametrize(
    "controls", [[-0.01, 0.5], [1.01, 0.5], [0.5], [float("nan"), 0.5]]
)
def test_v2_native_rejects_controls(
    tmp_path: Path, runner: Path, controls: list[float]
) -> None:
    model = SSMWaveNet(
        num_blocks=1, channels=2, state_dim=2, num_controls=2, discretization="zoh"
    )
    path = EXPORT.export_model(model, tmp_path / "model.json")
    with pytest.raises(subprocess.CalledProcessError):
        native_events(
            runner,
            path,
            np.zeros(16, dtype=np.float32),
            [{"sample": 0, "controls": controls}],
            tmp_path,
        )


@pytest.mark.skipif(
    os.environ.get("FSSR_VS100R_NATIVE_CAPTURE") != "1",
    reason="opt-in local VS100R checkpoint and private capture (CPU)",
)
def test_vs100r_checkpoint_and_real_take(tmp_path: Path, runner: Path) -> None:
    import sys

    from scipy.signal import resample_poly

    sys.path.insert(0, str(v1.ROOT / "scripts"))
    import product_vs100r_eval as capture

    wrapper = capture.load("vs100r48fast_film_s42")
    model = next(m for m in wrapper.modules() if isinstance(m, SSMWaveNet))
    path = EXPORT.export_model(model, tmp_path / "vs100r.json")
    x = (np.random.default_rng(42).normal(size=48000) * 0.1).astype(np.float32)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        for controls in ([0.0, 1.0], [0.5, 0.25], [1.0, 0.0]):
            events = [{"sample": 0, "controls": controls}]
            expected = reference_events(model, x, events)
            actual, reset = native_events(runner, path, x, events, tmp_path)
            error = float(np.max(np.abs(actual - expected)))
            print(f"VS100R {controls=} max_abs={error:.9g}")
            assert np.isfinite(actual).all() and actual.shape == expected.shape
            assert error <= 1e-5
            np.testing.assert_array_equal(actual, reset)
        events = [
            {"sample": 0, "controls": [0.0, 1.0]},
            {"sample": 16007, "controls": [0.5, 0.25]},
            {"sample": 32011, "controls": [1.0, 0.0]},
        ]
        expected = reference_events(model, x, events)
        actual, reset = native_events(runner, path, x, events, tmp_path)
        whole, _ = native_events(runner, path, x, events, tmp_path, str(len(x)))
        error = float(np.max(np.abs(actual - expected)))
        print(f"VS100R changing controls max_abs={error:.9g}")
        assert error <= 1e-5
        np.testing.assert_array_equal(actual, reset)
        np.testing.assert_array_equal(actual, whole)
        take = next(t for t in capture.takes() if t["split"] == "test")
        loop, _ = capture.audio(take)
        x = resample_poly(loop, 1, 4).astype(np.float32)
        events = [{"sample": 0, "controls": [v / 10 for v in take["gain_contour"]]}]
        expected = reference_events(model, x, events)
        actual, reset = native_events(
            runner, path, x, events, tmp_path, "7,64,333,1024"
        )
        error = float(np.max(np.abs(actual - expected)))
        esr = float(np.sum((actual - expected) ** 2) / np.sum(expected**2))
        print(
            f"VS100R di_test order={take['order']} setting={take['setting']} "
            f"samples={len(x)} max_abs={error:.9g} ESR={esr:.9g}"
        )
        assert actual.shape == expected.shape and np.isfinite(actual).all()
        assert error <= 1e-5
        np.testing.assert_array_equal(actual, reset)
    finally:
        torch.set_num_threads(threads)
