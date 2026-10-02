#!/usr/bin/env python3
"""T2 checks (THEORY.md): the polarity trap of the L1 + MR-STFT loss on the gain path.

Along y_hat = a*y near a = -1 (a = -1 + eps), with the bench's FlexibleLoss
(w1 * L1 + w2 * auraloss MR-STFT, default resolutions, SC + log-magnitude, eps 1e-8
on the power):
  L(eps) - L(0) = -w1 E|y| eps + w2 m |eps| + O(eps^2),
  m = mean over resolutions of (rho_r + f_r), f_r the fraction of STFT bins with
  power >= 1e-8 and rho_r = ||Y on those bins||_F / ||max(|Y|, 1e-4)||_F.
when no STFT power of the target equals 1e-8 exactly. So the right slope is
w2 m - w1 E|y| and the left slope -(w2 m + w1 E|y|): a = -1 is a strict local minimum of
the gain path if w1 E|y| < w2 m, and also at equality when some bin is above the floor
(the log term then gives w2 f (-log(1 - eps) - eps) > 0 on the right); not a minimum if
w1 E|y| > w2 m.
Symbolic: one-sided derivatives of the model loss. Numeric: real Big Muff test targets,
finite-difference slopes of the real loss against the prediction, two weightings,
targets scaled around a reference scale c0 = w2 m(y) / (w1 E|y|) (m varies with the
scale, so c0 is not the exact crossing), the equality case, the bins near the floor,
and the tanh-head statistic E[y^2 |y|] / E|y|. Fails loudly.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import sympy as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
bench = importlib.import_module("product_nablafx_bench")

EPS_POWER = 1e-8


def symbolic() -> None:
    e = sp.symbols("epsilon", real=True)
    w1, w2, mean_abs, rho, f = sp.symbols("w1 w2 E rho f", positive=True)
    a = -1 + e
    g = sp.Abs(a)
    loss = w1 * mean_abs * sp.Abs(a - 1) + w2 * (
        rho * sp.Abs(g - 1) + f * sp.Abs(sp.log(g))
    )
    right = sp.limit((loss - loss.subs(e, 0)) / e, e, 0, "+")
    left = sp.limit((loss - loss.subs(e, 0)) / e, e, 0, "-")
    assert sp.simplify(right - (w2 * (rho + f) - w1 * mean_abs)) == 0, right
    assert sp.simplify(left - (-(w2 * (rho + f) + w1 * mean_abs))) == 0, left
    print("symbolique : pente droite w2 m - w1 E|y| ; pente gauche -(w2 m + w1 E|y|)")


def powers(loss: bench.FlexibleLoss, y: torch.Tensor) -> list[torch.Tensor]:
    """STFT powers of y at each resolution of the real loss."""
    out = []
    for stft in loss.loss_functions[1].loss_fn.stft_losses:
        spec = torch.stft(
            y.view(-1, y.shape[-1]),
            stft.fft_size,
            stft.hop_size,
            stft.win_length,
            stft.window,
            return_complex=True,
        )
        out.append(spec.real**2 + spec.imag**2)
    return out


def floor_measure(loss: bench.FlexibleLoss, y: torch.Tensor) -> tuple[float, float]:
    """(m, f): means over resolutions of rho_r + f_r and of f_r."""
    m_terms, f_terms = [], []
    for power in powers(loss, y):
        kept = power >= EPS_POWER
        rho = (power[kept].sum() / power.clamp(min=EPS_POWER).sum()).sqrt()
        m_terms.append(float(rho) + float(kept.double().mean()))
        f_terms.append(float(kept.double().mean()))
    return sum(m_terms) / len(m_terms), sum(f_terms) / len(f_terms)


def near_floor(loss: bench.FlexibleLoss, y: torch.Tensor, h: float) -> float:
    """Fraction of bins whose power crosses the floor for a gain in [1 - h, 1 + h];
    asserts that none sits exactly on it."""
    crossing, total = 0, 0
    for power in powers(loss, y):
        assert not (power == EPS_POWER).any(), "bin exactly at the floor"
        band = (power >= EPS_POWER / (1 + h) ** 2) & (power < EPS_POWER / (1 - h) ** 2)
        crossing += int(band.sum())
        total += power.numel()
    return crossing / total


def slopes(loss: bench.FlexibleLoss, y: torch.Tensor, h: float) -> tuple[float, float]:

    def at(a: float) -> float:
        return float(loss(a * y, y)[-1])

    centre = at(-1.0)
    return (at(-1.0 + h) - centre) / h, (centre - at(-1.0 - h)) / h


def numeric() -> None:
    data = bench.data_module("test")
    data.setup("test")
    targets = torch.stack([t for _, t in data.test_dataset]).double()
    mean_abs = float(targets.abs().mean())
    print(f"cibles Big Muff test : {tuple(targets.shape)}, E|y| = {mean_abs:.5f}")
    tanh_stat = float((targets**2 * targets.abs()).mean()) / mean_abs
    peak = float(targets.abs().max())
    print(f"  tete tanh : E[y^2 |y|]/E|y| = {tanh_stat:.4f}, crete {peak:.4f}")
    for w1, w2 in ((1.0, 0.1), (0.5, 0.5)):
        loss = bench.FlexibleLoss(
            losses=[
                {"name": "l1_loss", "weight": w1, "alias": "l1"},
                {"name": "mrstft_loss", "weight": w2, "alias": "mrstft"},
            ]
        )
        for stft in loss.loss_functions[1].loss_fn.stft_losses:
            stft.window = stft.window.double()
        m, f = floor_measure(loss, targets)
        reference = (
            w2 * m / (w1 * mean_abs)
        )  # reference scale c0, not the exact crossing
        print(
            f"poids {w1}/{w2} : m = {m:.4f}, f = {f:.4f}, echelle c0 = {reference:.2f}"
        )
        for scale in (1.0, 0.8 * reference, 1.25 * reference):
            y = scale * targets
            (m_s, _), e_s = floor_measure(loss, y), float(y.abs().mean())
            print(
                f"  c = {scale:7.2f} : m = {m_s:.4f}, cases pres du plancher (h = 1e-3)"
                f" {near_floor(loss, y, 1e-3):.2e}"
            )
            right_pred, left_pred = w2 * m_s - w1 * e_s, -(w2 * m_s + w1 * e_s)
            for h in (1e-3, 1e-4):
                right, left = slopes(loss, y, h)
                print(
                    f"  c = {scale:7.2f}, h = {h:.0e} : droite {right:+.5f}"
                    f" (prevu {right_pred:+.5f}), gauche {left:+.5f}"
                    f" (prevu {left_pred:+.5f})"
                )
                tol = 0.02 * (w2 * m_s + w1 * e_s)
                assert abs(right - right_pred) < tol, (w1, w2, scale, h, right)
                assert abs(left - left_pred) < tol, (w1, w2, scale, h, left)
                assert (right > 0) == (right_pred > 0), (w1, w2, scale, h, right)
            print(f"    pente droite positive (minimum strict) : {right_pred > 0}")
    # Equality w1 E|y| = w2 m on the real targets (w2 = 0.1): the first-order terms
    # cancel and the log term leaves w2 f (-log(1 - h) - h) > 0 on the right.
    w2 = 0.1
    loss = bench.FlexibleLoss(
        losses=[
            {"name": "l1_loss", "weight": w2 * m / mean_abs, "alias": "l1"},
            {"name": "mrstft_loss", "weight": w2, "alias": "mrstft"},
        ]
    )
    for stft in loss.loss_functions[1].loss_fn.stft_losses:
        stft.window = stft.window.double()
    m, f = floor_measure(loss, targets)

    def at(a: float) -> float:
        return float(loss(a * targets, targets)[-1])

    centre = at(-1.0)
    for h in (1e-2, 1e-3):
        right, left = at(-1.0 + h) - centre, at(-1.0 - h) - centre
        predicted = w2 * f * (-math.log(1 - h) - h)
        print(
            f"egalite, h = {h:.0e} : increment droit {right:.3e}"
            f" (prevu {predicted:.3e}), gauche {left:.3e} (prevu {2 * w2 * m * h:.3e})"
        )
        assert right > 0 and left > 0, (h, right, left)


if __name__ == "__main__":
    torch.set_num_threads(4)
    symbolic()
    numeric()
