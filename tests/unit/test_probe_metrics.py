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


def test_harmonic_signature_matches_trigonometry() -> None:
    """For y = x + a3 x^3 driven by A sin, a change of a3 moves the fundamental by
    (3/4) da3 A^2 and the third harmonic by da3 A^2 / 4 (relative to D_1 = 1); an added
    a2 x^2 creates a second harmonic a2 A / 2, present only in the evaluated output."""
    from fssr_nam.product.defect_bench import synchronized_sweep
    from fssr_nam.product.fidelity import HarmonicProtocol, analyze_sweep_pair

    sweep_x, sweep = synchronized_sweep(5.0, FS)
    x = 0.3 * sweep_x
    y_ref = x + 0.1 * x**3
    protocol = HarmonicProtocol()
    cubic = analyze_sweep_pair(x, y_ref, x + 0.2 * x**3, FS, sweep, protocol)
    assert cubic["order_1"]["complex_rel"] == pytest.approx(0.75 * 0.1 * 0.09, rel=0.02)
    assert cubic["order_3"]["complex_rel"] == pytest.approx(0.1 * 0.09 / 4, rel=0.02)
    even = analyze_sweep_pair(x, y_ref, y_ref + 0.05 * x**2, FS, sweep, protocol)
    assert even["order_2"]["complex_rel"] == pytest.approx(0.05 * 0.3 / 2, rel=0.02)
    assert even["order_2"]["bins_only_eval"] > 0.9 * even["order_2"]["bins_observed"]
    same = analyze_sweep_pair(x, y_ref, y_ref, FS, sweep, protocol)
    assert same["total_complex_rel"] == 0.0


def volterra_probe():
    from fssr_nam.product.probes import VolterraLaguerre

    return VolterraLaguerre(size=4, pole=0.5, order=2, ridge=1e-10)


def in_basis(c12: float):
    """A system exactly in the probe's span: z_0 + c12 z_1 z_2."""
    from fssr_nam.product.probes import laguerre

    def system(v: np.ndarray) -> np.ndarray:
        z = laguerre(v, 4, 0.5)
        return z[:, 0] + c12 * z[:, 1] * z[:, 2]

    return system


def test_volterra_separates_exactly_representable_systems() -> None:
    from fssr_nam.product.probes import laguerre

    x, y_ref = segments(in_basis(0.3))
    _, y_eval = segments(in_basis(0.6))
    reading = analyze(volterra_probe(), x, y_ref, y_eval)
    z = laguerre(x["B"], 4, 0.5)
    expected = 0.3 * np.sqrt(np.mean((z[:, 1] * z[:, 2]) ** 2))
    assert abs(reading["r_eval"] - reading["r_ref"]) < 1e-6 * expected
    assert reading["d_probe"] == pytest.approx(expected, rel=1e-4)
    assert reading["by_branch"]["order_1"] < 1e-4 * expected


def test_volterra_sees_a_cross_interaction_that_hammerstein_cannot() -> None:
    x, y_ref = segments(in_basis(0.0))
    _, y_eval = segments(in_basis(0.5))
    volterra = analyze(volterra_probe(), x, y_ref, y_eval)
    hammerstein = analyze(HAMMERSTEIN, x, y_ref, y_eval)
    assert volterra["d_probe"] > 0.99 * volterra["d_audio"]
    assert hammerstein["d_probe"] < 0.5 * hammerstein["d_audio"]
    assert hammerstein["unrepresented"] > 0.5 * hammerstein["d_audio"]


def test_volterra_fit_is_linear_in_the_output() -> None:
    from fssr_nam.product.probes import fit_volterra

    probe = volterra_probe()
    x, y_ref = segments(lambda v: np.tanh(2 * v))
    _, y_eval = segments(lambda v: np.tanh(2.3 * v))
    a = fit_volterra(probe, x["A"], y_ref["A"])
    b = fit_volterra(probe, x["A"], y_eval["A"])
    diff = fit_volterra(probe, x["A"], y_eval["A"] - y_ref["A"])
    np.testing.assert_allclose(b.theta - a.theta, diff.theta, rtol=1e-7, atol=1e-10)


def wh_system(v: np.ndarray) -> np.ndarray:
    from scipy.signal import lfilter

    u = lfilter([1.0, -0.6], [1.0], v)
    return lfilter([0.5, 0.3, 0.2], [1.0], np.tanh(3 * u + 0.2) - np.tanh(0.2))


def small_wh():
    from fssr_nam.product.probes import WienerHammerstein

    return WienerHammerstein(taps_in=4, taps_out=8, starts=3, iterations=20)


def test_wh_reparametrisations_render_identically() -> None:
    """(h_in, phi(u)) -> (-h_in, phi(-u)) and (h_out, phi) -> (c h_out, phi / c) leave
    the output unchanged while the parameters move far apart: parameters are never
    compared, only renderings."""
    from fssr_nam.product.probes import WhState, fit_wh, wh_render

    x, y = segments(wh_system)
    probe = small_wh()
    best = fit_wh(probe, x["A"], y["A"]).states[0]
    mirrored = WhState(-best.h_in, best.coef[::-1].copy(), best.h_out, best.scale)
    rescaled = WhState(best.h_in, best.coef / 3.0, 3.0 * best.h_out, best.scale)
    reference = wh_render(best, probe, x["B"])
    for other in (mirrored, rescaled):
        np.testing.assert_allclose(
            wh_render(other, probe, x["B"]), reference, rtol=0, atol=1e-12
        )
    gap = np.linalg.norm(
        np.r_[mirrored.h_in, mirrored.coef] - np.r_[best.h_in, best.coef]
    )
    assert gap > 0.5 * np.linalg.norm(np.r_[best.h_in, best.coef])


def test_wh_starts_differ_more_in_parameters_than_in_output() -> None:
    from fssr_nam.product.probes import fit_wh, wh_render

    x, y = segments(wh_system)
    probe = small_wh()
    states = fit_wh(probe, x["A"], y["A"]).states
    rms = np.sqrt(np.mean(y["B"] ** 2))

    def params(s):
        return np.r_[s.h_in, s.coef * s.scale, s.h_out]

    render_gap = (
        np.sqrt(
            np.mean(
                (
                    wh_render(states[1], probe, x["B"])
                    - wh_render(states[0], probe, x["B"])
                )
                ** 2
            )
        )
        / rms
    )
    param_gap = np.linalg.norm(params(states[1]) - params(states[0])) / np.linalg.norm(
        params(states[0])
    )
    assert param_gap > 10 * render_gap


def test_wh_identity_scores_zero_and_reports_start_spread() -> None:
    x, y = segments(wh_system)
    reading = analyze(small_wh(), x, y, y)
    assert reading["d_probe"] == 0.0
    assert reading["start_spread_ref"] > 0.0
    assert reading["bound"]["holds"]


def test_reference_signature_is_cached_and_reused(tmp_path) -> None:
    x, y_ref = segments(lambda v: v + 0.1 * v**3)
    _, y_eval = segments(lambda v: v + 0.2 * v**3)
    protocol = Protocol(probe=HAMMERSTEIN, estimation=("A",), reading=("B",))
    first = analyze_pair(x, y_ref, y_eval, FS, protocol, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.pkl"))) == 1
    again = analyze_pair(x, y_ref, y_eval, FS, protocol, cache_dir=tmp_path)
    assert again["readings"] == first["readings"]
    other = {k: 1.01 * v for k, v in y_ref.items()}
    analyze_pair(x, other, y_eval, FS, protocol, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.pkl"))) == 2  # changed data, new signature


def test_own_input_fit_removes_an_input_mismatch() -> None:
    """Same system, but the evaluated take was driven by 1.01 x: fitted on its own
    input, its probe matches the reference probe, while the raw recordings differ."""

    def system(v: np.ndarray) -> np.ndarray:
        return v + 0.1 * v**3

    x, y_ref = segments(system)
    x_eval = {k: 1.01 * v for k, v in x.items()}
    y_eval = {k: system(v) for k, v in x_eval.items()}
    protocol = Protocol(probe=HAMMERSTEIN, estimation=("A",), reading=("B",))
    reading = analyze_pair(x, y_ref, y_eval, FS, protocol, x_eval=x_eval)["readings"][
        "B"
    ]
    assert reading["d_probe"] < 1e-3 * reading["d_audio"]
    same = analyze_pair(x, y_ref, y_ref, FS, protocol, x_eval=x)["readings"]["B"]
    assert same["d_probe"] == 0.0
