"""Helpers of lead6_learned_features.py: frozen features of the VS100R arm C networks
(vs100r48fast_film_s42-44, last checkpoints) on CPU, the VS100R sections at 48 kHz, and
multi-target ridge readouts with the semantics of harness.FeatureProbe.

Features of one forward pass (float32, (channels, n) per representation):
- "pen": the penultimate activations, output_net[2] (16 channels); the network output
  is tanh(w . pen + b), so a static readout of pen holds the pre-tanh output;
- "blocks": the 8 gated SSM blocks' skip outputs (8 x 16 = 128 channels), whose sum
  feeds the output head;
- "out": the network output (1 channel).
Each section runs in one piece from the zero state, as product_vs100r_eval.run does.
The SSM kernels do not depend on the knobs (cond="film" acts after the SSM), so each
layer's kernel is computed once per length and cached (same values, less time).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.fft import irfft, next_fast_len, rfft
from scipy.linalg import solve
from scipy.signal import fftconvolve, resample_poly

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
import harness as h  # noqa: E402
import product_vs100r_eval as ev  # noqa: E402

from fssr_nam.models.ssm_wavenet import DiagonalSSMLayer  # noqa: E402
from fssr_nam.product import vs100r  # noqa: E402
from fssr_nam.product.probes import Probe, normal_equations, regularized  # noqa: E402

RUN = "vs100r48fast_film_s{}"
CACHE = h.CACHE / "lead6"


# ---------------------------------------------------------------- VS100R sections


def vs100r_takes() -> dict[int, dict]:
    """OD2, Power Dimension off takes of the VS100R sheet (product_vs100r_eval)."""
    return {t["order"]: t for t in ev.takes()}


def vs100r_section(order: int, section: str) -> tuple[np.ndarray, np.ndarray]:
    """(own loopback x, FX send y) of a take on one section: cut at 192 kHz at the
    layout offset, spikes repaired (vs100r.audio), decimated by 4 on its own, float64;
    cached in CACHE/sections48k and read back memory-mapped."""
    folder = CACHE / "sections48k" / f"take_{order:02d}"
    paths = [folder / f"{k}_{section}.npy" for k in ("x", "y")]
    if not all(p.exists() for p in paths):
        folder.mkdir(parents=True, exist_ok=True)
        pair = vs100r.audio(vs100r_takes()[order], section)
        for path, signal in zip(paths, pair, strict=True):
            partial = path.with_name(f".{path.stem}.partial.npy")
            np.save(partial, resample_poly(signal, 1, 4))
            partial.replace(path)
    x, y = (np.load(p, mmap_mode="r") for p in paths)
    return x, y


# ---------------------------------------------------------------- network features


def _cached(kernel):
    store: dict[int, torch.Tensor] = {}

    def get(length: int) -> torch.Tensor:
        if length not in store:
            store[length] = kernel(length)
        return store[length]

    return get


def load_model(seed: int, trained: bool = True):
    """Arm C network of a seed at its last checkpoint (trained=True), or the same
    architecture at a random initialisation drawn after torch.manual_seed(seed)
    (trained=False, the untrained control); CPU, eval mode, SSM kernels cached."""
    DiagonalSSMLayer.fft_size = "fast"  # the runs' FFT length; same convolution
    if trained:
        model = ev.load(RUN.format(seed))
    else:
        record = json.loads(
            (ev.bench.RESULTS_DIR / f"{RUN.format(seed)}.json").read_text()
        )
        args = argparse.Namespace(
            model=record["model"], device="vs100r", **record["ssm"]
        )
        torch.manual_seed(seed)
        model = ev.BlackBoxModel(ev.bench.build_processor(args)).eval()
    for layer in model.modules():
        if isinstance(layer, DiagonalSSMLayer):
            layer._kernel = _cached(layer._kernel)
    return model


def controls(gain: float, contour: float) -> torch.Tensor:
    """Knob vector as coreading_delay.controls: knob value / 10."""
    return torch.tensor([gain / 10, contour / 10], dtype=torch.float32)


@torch.no_grad()
def features(model, x: np.ndarray, knobs: torch.Tensor) -> dict[str, np.ndarray]:
    """Frozen features of x, one piece from the zero state, float32."""
    store: dict[str, torch.Tensor] = {}
    processor = model.processor
    hooks = [
        processor.output_net[2].register_forward_hook(
            lambda m, i, o: store.__setitem__("pen", o)
        )
    ]
    for k, block in enumerate(processor.blocks):
        hooks.append(
            block.register_forward_hook(
                lambda m, i, o, k=k: store.__setitem__(f"skip{k}", o[1])
            )
        )
    model.reset_states()
    signal = torch.from_numpy(np.asarray(x, dtype=np.float32))[None, None]
    out = model(signal, knobs[None])
    for hook in hooks:
        hook.remove()
    skips = [store[f"skip{k}"][0] for k in range(len(processor.blocks))]
    return {
        "pen": store["pen"][0].numpy(),
        "blocks": torch.cat(skips, dim=0).numpy(),
        "out": out[0].numpy(),
    }


# ---------------------------------------------------------------- readouts


def _rows_rms(u: np.ndarray, step: int = 2**18) -> np.ndarray:
    total = np.zeros(len(u))
    for a in range(0, u.shape[1], step):
        c = u[:, a : a + step].astype(np.float64)
        total += np.sum(c * c, axis=1)
    return np.maximum(np.sqrt(total / u.shape[1]), 1e-300)


class StaticReadout:
    """q = c + sum_k w_k f_k / s_k: harness.FeatureProbe with taps=1 (s_k = RMS of f_k
    on the fit rows, unpenalised constant, ridge times the mean Gram diagonal), the
    Gram computed once in chunks and solved for any number of targets and ridges.
    Linear in the target for a fixed ridge."""

    def __init__(self, u: np.ndarray, step: int = 2**18) -> None:
        self.u, self.step, self.scales = u, step, _rows_rms(u)
        k = len(u)
        self.gram = np.zeros((k + 1, k + 1))
        self.gram[0, 0] = u.shape[1]
        for c in self._chunks(u):
            self.gram[1:, 1:] += c @ c.T
            self.gram[0, 1:] += c.sum(axis=1)
        self.gram[1:, 0] = self.gram[0, 1:]

    def _chunks(self, u: np.ndarray):
        for a in range(0, u.shape[1], self.step):
            yield u[:, a : a + self.step].astype(np.float64) / self.scales[:, None]

    def rhs(self, y: np.ndarray) -> np.ndarray:
        out = np.zeros(len(self.u) + 1)
        for i, c in enumerate(self._chunks(self.u)):
            seg = np.asarray(y[i * self.step : (i + 1) * self.step], dtype=np.float64)
            out[0] += seg.sum()
            out[1:] += c @ seg
        return out

    def solve(self, rhs: np.ndarray, ridge: float) -> np.ndarray:
        probe = Probe(dc=True, ridge=ridge)
        return solve(regularized(probe, self.gram), rhs, assume_a="pos")

    def predict(self, theta: np.ndarray, u: np.ndarray) -> np.ndarray:
        return theta[0] + np.concatenate([theta[1:] @ c for c in self._chunks(u)])


class FirReadout:
    """harness.FeatureProbe (each feature through its own causal FIR of `taps`, RMS
    scales on the fit rows, unpenalised constant, relative ridge) with the library's
    Gram (probes.normal_equations) computed once and solved for several targets."""

    def __init__(self, u: np.ndarray, taps: int = 256) -> None:
        self.taps, self.scales = taps, _rows_rms(u)
        rows = [
            np.asarray(f, dtype=np.float64) / s
            for f, s in zip(u, self.scales, strict=True)
        ]
        self.degree = len(rows)
        self.gram, _ = normal_equations(Probe(self.degree, taps), rows, None)
        self.size = next_fast_len(len(rows[0]) + taps)
        self.spectra = [rfft(r, self.size) for r in rows]

    def rhs(self, y: np.ndarray) -> np.ndarray:
        target = rfft(np.asarray(y, dtype=np.float64), self.size)
        parts = [np.array([np.sum(y)])]
        parts += [
            irfft(np.conj(s) * target, self.size)[: self.taps] for s in self.spectra
        ]
        return np.concatenate(parts)

    def solve(self, rhs: np.ndarray, ridge: float) -> np.ndarray:
        probe = Probe(self.degree, self.taps, ridge)
        return solve(regularized(probe, self.gram), rhs, assume_a="pos")

    def predict(self, theta: np.ndarray, u: np.ndarray) -> np.ndarray:
        m, out = self.taps, np.full(u.shape[1], theta[0])
        for k, (f, s) in enumerate(zip(u, self.scales, strict=True)):
            row = np.asarray(f, dtype=np.float64) / s
            out += fftconvolve(row, theta[1 + k * m : 1 + (k + 1) * m])[: len(row)]
        return out


def readout(u: np.ndarray, taps: int) -> StaticReadout | FirReadout:
    return StaticReadout(u) if taps == 1 else FirReadout(u, taps)
