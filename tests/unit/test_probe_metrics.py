"""Deviation scores from linear-in-parameters analysis probes (FIR, Hammerstein)."""

from __future__ import annotations

import numpy as np
import pytest

from fssr_nam.product.fidelity import Protocol, analyze_pair, band_rms
from fssr_nam.product.probes import Probe, branches, fit, normal_equations, render

FS = 48_000
HAMMERSTEIN = Probe(degree=3, taps=8, ridge=1e-9)


def noise(seed: int, n: int, level: float = 0.2) -> np.ndarray:
    return level * np.random.default_rng(seed).standard_normal(n)


def segments(system, seed: int = 0) -> tuple[dict, dict]:
    x = {"A": noise(seed, 16_384), "B": noise(seed + 1, 8_192)}
    return x, {k: system(v) for k, v in x.items()}


def analyze(probe: Probe, x: dict, y_ref: dict, y_eval: dict) -> dict:
    protocol = Protocol(probe=probe, estimation=("A",), reading=("B",), skip_s=0.0)
    return analyze_pair(x, y_ref, y_eval, FS, protocol)["readings"]["B"]


def lag_matrix(u: list, taps: int, rows: int) -> np.ndarray:
    """Explicit lagged features, zero initial state, `rows` rows."""
    cols = []
    for signal in u:
        padded = np.concatenate((signal, np.zeros(rows - len(signal))))
        for lag in range(taps):
            cols.append(np.concatenate((np.zeros(lag), padded[: rows - lag])))
    return np.column_stack(cols)


def test_normal_equations_match_explicit_features() -> None:
    probe = Probe(degree=3, taps=5, dc=True)
    x, y = noise(3, 300), noise(4, 300)
    u, _ = branches(x, probe.degree)
    gram, rhs = normal_equations(probe, u, y)
    rows = len(x) + probe.taps - 1  # whole convolution support
    phi = lag_matrix(u, probe.taps, rows)
    const = np.r_[np.ones(len(x)), np.zeros(rows - len(x))][:, None]
    phi = np.hstack((const, phi))
    target = np.r_[y, np.zeros(rows - len(x))]
    np.testing.assert_allclose(gram, phi.T @ phi, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(rhs, phi.T @ target, rtol=1e-10, atol=1e-10)


def test_identity_scores_zero() -> None:
    x, y = segments(lambda v: np.tanh(3 * v))
    reading = analyze(HAMMERSTEIN, x, y, y)
    assert reading["d_probe"] == 0.0 and reading["d_audio"] == 0.0
    assert reading["bound"]["holds"]


def test_same_fit_error_but_different_systems_are_separated() -> None:
    """F = x + 0.1 x^3 and G = x + 0.2 x^3 are both exact cubic Hammersteins: their fit
    errors agree (both ~0) while the behavioural score is 0.1 RMS(x_B^3)."""
    x, y_ref = segments(lambda v: v + 0.1 * v**3)
    _, y_eval = segments(lambda v: v + 0.2 * v**3)
    reading = analyze(HAMMERSTEIN, x, y_ref, y_eval)
    fit_error_gap = abs(reading["r_eval"] - reading["r_ref"])
    expected = 0.1 * np.sqrt(np.mean(x["B"] ** 6))
    assert fit_error_gap < 1e-6 * expected
    assert reading["d_probe"] == pytest.approx(expected, rel=1e-4)
    assert reading["by_branch"]["order_3"] == pytest.approx(expected, rel=1e-4)
    assert reading["by_branch"]["order_1"] < 1e-4 * expected


def test_difference_outside_capacity_is_reported_as_unrepresented() -> None:
    """x[n] x[n-3] is no sum of filtered powers: a Hammerstein probe barely sees it,
    and the score must say so through the unrepresented part, not hide it."""
    x, y_ref = segments(lambda v: v)

    def cross(v: np.ndarray) -> np.ndarray:
        return v + 0.5 * v * np.r_[np.zeros(3), v[:-3]]

    _, y_eval = segments(cross)
    reading = analyze(HAMMERSTEIN, x, y_ref, y_eval)
    assert reading["d_probe"] < 0.1 * reading["d_audio"]
    assert reading["unrepresented"] == pytest.approx(reading["d_audio"], rel=0.05)
    assert reading["bound"]["holds"]


def test_noisy_repeats_score_far_below_a_small_real_change() -> None:
    def system(v: np.ndarray) -> np.ndarray:
        return np.tanh(2 * v)

    x, clean = segments(system)
    rng = np.random.default_rng(9)
    repeats = [
        {k: v + 1e-3 * rng.standard_normal(len(v)) for k, v in clean.items()}
        for _ in range(4)
    ]
    noise_scores = [
        analyze(HAMMERSTEIN, x, repeats[0], other)["d_probe"] for other in repeats[1:]
    ]
    _, louder = segments(lambda v: 1.01 * system(v))  # +0.09 dB
    change = analyze(HAMMERSTEIN, x, clean, louder)["d_probe"]
    assert max(noise_scores) < 0.1 * change


def test_score_is_stable_across_independent_excitations() -> None:
    scores = []
    for seed in (0, 10, 20):
        x, y_ref = segments(lambda v: v + 0.1 * v**3, seed)
        _, y_eval = segments(lambda v: v + 0.15 * v**3, seed)
        scores.append(analyze(HAMMERSTEIN, x, y_ref, y_eval)["d_norm"])
    assert max(scores) / min(scores) < 1.1


def test_linear_equivalence_and_gram_norm() -> None:
    """fit(y_eval) - fit(y_ref) = fit(y_eval - y_ref), and the rendered score equals
    the norm of that difference weighted by the Gram matrix of the reading features."""
    probe = Probe(degree=3, taps=6, ridge=1e-3)
    x, y_ref = segments(lambda v: np.tanh(2 * v))
    _, y_eval = segments(lambda v: np.tanh(2.2 * v))
    ref, ev = fit(probe, x["A"], y_ref["A"]), fit(probe, x["A"], y_eval["A"])
    diff = fit(probe, x["A"], y_eval["A"] - y_ref["A"])
    delta = ev.theta - ref.theta
    np.testing.assert_allclose(delta, diff.theta, rtol=1e-8, atol=1e-12)
    u, _ = branches(x["B"], probe.degree, ref.scales)
    phi = np.hstack((np.ones((len(x["B"]), 1)), lag_matrix(u, probe.taps, len(u[0]))))
    gram = phi.T @ phi / len(x["B"])
    rendered = render(ev, x["B"]) - render(ref, x["B"])
    assert np.sqrt(np.mean(rendered**2)) == pytest.approx(
        np.sqrt(delta @ gram @ delta), rel=1e-9
    )


def test_band_rms_adds_up_to_total() -> None:
    signal = noise(5, 4_800)
    edges = (100.0, 1_000.0, 10_000.0)
    bands = band_rms(signal, FS, edges)
    below = band_rms(signal, FS, (0.0, *edges))[0]
    total = np.sqrt(np.mean(signal**2))
    assert np.sqrt(below**2 + sum(b**2 for b in bands)) == pytest.approx(total)
