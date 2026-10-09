"""Analysis probes for audio-fidelity scores (product lane, no scientific claim).

A probe is an identification model fitted SEPARATELY to (x, y_ref) and (x, y_eval) on
estimation data A, then rendered on reading excitations B. The score compares the two
renderings, q_eval - q_ref, never the two fit errors: two different systems can have
the same fit error (F = x + 0.1 x^3 and G = x + 0.2 x^3 are both exact cubic
Hammersteins).

Probes here are linear in their parameters:
- `fir`: one causal FIR on x, a linear signature conditioned by the excitation (not
  "the linear part" of the device);
- `hammerstein`: parallel branches x^k, k = 1..degree, each through its own FIR.
Branch signals are scaled by their RMS on x_A, and the ridge is relative to the mean
diagonal of the Gram matrix: both depend on x_A only, so the fitted parameters are a
linear function of y, and fit(y_eval) - fit(y_ref) = fit(y_eval - y_ref) exactly.

Fit: least squares summed over the whole convolution support (rows 0 .. n + taps - 2,
y taken as zero past its end; the constant column only on rows 0 .. n - 1). The lagged
Gram matrix is then exactly block-Toeplitz and is built from FFT cross-correlations.
Rendering: causal convolution from a zero state, first n samples.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.linalg import solve
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class Probe:
    """Protocol of a linear-in-parameters probe; frozen before any test reading."""

    degree: int = 1  # 1: FIR probe; > 1: Hammerstein branches x^1 .. x^degree
    taps: int = 64
    ridge: float = 1e-6
    dc: bool = True

    @property
    def kind(self) -> str:
        return "fir" if self.degree == 1 else "hammerstein"

    def version(self) -> str:
        text = json.dumps({"kind": self.kind, **asdict(self)}, sort_keys=True)
        return f"{self.kind}-{hashlib.sha256(text.encode()).hexdigest()[:12]}"


@dataclass
class Fit:
    probe: Probe
    scales: np.ndarray  # RMS of each branch signal x^k on x_A
    theta: np.ndarray  # [dc] + degree * taps FIR coefficients


def branches(
    x: np.ndarray, degree: int, scales: np.ndarray | None = None
) -> tuple[list[np.ndarray], np.ndarray]:
    """Branch signals x^k / scale_k; scales default to their RMS on this x."""
    powers = [x**k for k in range(1, degree + 1)]
    if scales is None:
        scales = np.array([max(np.sqrt(np.mean(p**2)), 1e-300) for p in powers])
    return [p / s for p, s in zip(powers, scales, strict=True)], scales


def _spectra(signals: list, size: int) -> list:
    return [rfft(s, size) for s in signals]


def normal_equations(probe: Probe, u: list, y: np.ndarray | None):
    """Gram matrix and, if y is given, Phi^T y for branch signals u."""
    n, m = len(u[0]), probe.taps
    if n < m:
        raise ValueError(f"estimation data ({n} samples) shorter than {m} taps")
    size = next_fast_len(n + m)
    spectra = _spectra(u, size)
    lag = np.subtract.outer(np.arange(m), np.arange(m)) % size  # i - j
    width = len(u) * m + probe.dc
    gram = np.empty((width, width))
    first = int(probe.dc)
    for k, uk in enumerate(spectra):
        for j in range(k, len(u)):
            block = irfft(np.conj(uk) * spectra[j], size)[lag]
            rows, cols = first + k * m, first + j * m
            gram[rows : rows + m, cols : cols + m] = block
            gram[cols : cols + m, rows : rows + m] = block.T
    if probe.dc:
        gram[0, 0] = n
        for k, signal in enumerate(u):
            # sum over rows 0..n-1 of signal[row - i] = sum of signal[0 .. n-1-i]
            sums = np.concatenate(([0.0], np.cumsum(signal)))[n - np.arange(m)]
            gram[0, first + k * m : first + (k + 1) * m] = sums
            gram[first + k * m : first + (k + 1) * m, 0] = sums
    if y is None:
        return gram, None
    target = rfft(y, size)
    rhs = [irfft(np.conj(uk) * target, size)[:m] for uk in spectra]
    if probe.dc:
        rhs.insert(0, np.array([np.sum(y)]))
    return gram, np.concatenate(rhs)


def regularized(probe: Probe, gram: np.ndarray) -> np.ndarray:
    """Gram plus the relative ridge, on every column except the constant."""
    first = int(probe.dc)
    diagonal = np.diag(gram)[first:]
    out = gram.copy()
    out[first:, first:] += probe.ridge * diagonal.mean() * np.eye(len(diagonal))
    return out


def fit(probe: Probe, x: np.ndarray, y: np.ndarray) -> Fit:
    """Ridge least-squares fit of the probe on estimation data (x, y)."""
    u, scales = branches(x, probe.degree)
    gram, rhs = normal_equations(probe, u, y)
    theta = solve(regularized(probe, gram), rhs, assume_a="pos")
    return Fit(probe, scales, theta)


def contributions(result: Fit, x: np.ndarray) -> dict:
    """Rendered output of each branch (and the constant) on x, zero initial state."""
    probe, n = result.probe, len(x)
    u, _ = branches(x, probe.degree, result.scales)
    first, m = int(probe.dc), probe.taps
    out = {}
    if probe.dc:
        out["dc"] = np.full(n, result.theta[0])
    for k, signal in enumerate(u, start=1):
        h = result.theta[first + (k - 1) * m : first + k * m]
        out[f"order_{k}"] = fftconvolve(signal, h)[:n]
    return out


def render(result: Fit, x: np.ndarray) -> np.ndarray:
    """Probe output on x: the sum of its contributions."""
    return np.sum(list(contributions(result, x).values()), axis=0)


@dataclass(frozen=True)
class VolterraLaguerre:
    """Reduced Volterra probe on a common, frozen Laguerre basis.

    z_i = L_i * x, i < size, with the discrete Laguerre filters of pole `pole`; the
    output is c0 + sum a_i z_i + sum_{i<=j} a_ij z_i z_j (+ cubic terms if order 3),
    so cross interactions between basis outputs are included. Coefficients from
    different bases are not comparable: the basis is part of the protocol. A diagonal
    Hammerstein kernel x^k[n - m] is in general NOT in the span of these products, so
    this probe and the Hammerstein probe are not assumed to be nested."""

    size: int = 8
    pole: float = 0.8
    order: int = 2
    ridge: float = 1e-6
    dc: bool = True

    @property
    def kind(self) -> str:
        return f"volterra_laguerre_{self.order}"

    def version(self) -> str:
        text = json.dumps({"kind": self.kind, **asdict(self)}, sort_keys=True)
        return f"{self.kind}-{hashlib.sha256(text.encode()).hexdigest()[:12]}"


CHUNK = 65_536


def laguerre(x: np.ndarray, size: int, pole: float) -> np.ndarray:
    """(n, size) outputs of the discrete Laguerre filters, zero initial state."""
    from scipy.signal import lfilter

    out = np.empty((len(x), size))
    out[:, 0] = lfilter([np.sqrt(1 - pole**2)], [1.0, -pole], x)
    for i in range(1, size):
        out[:, i] = lfilter([-pole, 1.0], [1.0, -pole], out[:, i - 1])
    return out


def monomials(size: int, order: int) -> list[tuple[int, ...]]:
    """Index tuples i <= j <= ... of every product up to `order`, by order."""
    from itertools import combinations_with_replacement

    return [
        terms
        for degree in range(1, order + 1)
        for terms in combinations_with_replacement(range(size), degree)
    ]


def volterra_features(z: np.ndarray, terms: list) -> np.ndarray:
    return np.column_stack([np.prod(z[:, list(t)], axis=1) for t in terms])


@dataclass
class VolterraFit:
    probe: VolterraLaguerre
    scales: np.ndarray  # RMS of each product on x_A
    theta: np.ndarray  # [dc] + one coefficient per product


def _volterra_columns(probe: VolterraLaguerre, x: np.ndarray, scales=None):
    terms = monomials(probe.size, probe.order)
    z = laguerre(x, probe.size, probe.pole)
    if scales is None:
        sums = np.zeros(len(terms))
        for start in range(0, len(x), CHUNK):
            sums += np.sum(volterra_features(z[start : start + CHUNK], terms) ** 2, 0)
        scales = np.maximum(np.sqrt(sums / len(x)), 1e-300)
    return z, terms, scales


def fit_volterra(probe: VolterraLaguerre, x: np.ndarray, y: np.ndarray) -> VolterraFit:
    """Ridge fit with columns scaled by their RMS on x and a relative ridge: both
    depend on x only, so theta is linear in y."""
    z, terms, scales = _volterra_columns(probe, x)
    width = len(terms) + probe.dc
    gram, rhs = np.zeros((width, width)), np.zeros(width)
    for start in range(0, len(x), CHUNK):
        phi = volterra_features(z[start : start + CHUNK], terms) / scales
        if probe.dc:
            phi = np.hstack((np.ones((len(phi), 1)), phi))
        gram += phi.T @ phi
        rhs += phi.T @ y[start : start + CHUNK]
    first = int(probe.dc)
    diagonal = np.diag(gram)[first:]
    gram[first:, first:] += probe.ridge * diagonal.mean() * np.eye(len(diagonal))
    return VolterraFit(probe, scales, solve(gram, rhs, assume_a="pos"))


def volterra_contributions(result: VolterraFit, x: np.ndarray) -> dict:
    """Rendered output of the constant and of each order on x."""
    probe = result.probe
    z, terms, _ = _volterra_columns(probe, x, result.scales)
    first = int(probe.dc)
    out = {"dc": np.full(len(x), result.theta[0])} if probe.dc else {}
    for degree in range(1, probe.order + 1):
        pick = [i for i, t in enumerate(terms) if len(t) == degree]
        q = np.empty(len(x))
        for start in range(0, len(x), CHUNK):
            phi = volterra_features(z[start : start + CHUNK], [terms[i] for i in pick])
            q[start : start + CHUNK] = (phi / result.scales[pick]) @ result.theta[
                first + np.array(pick)
            ]
        out[f"order_{degree}"] = q
    return out


@dataclass(frozen=True)
class WienerHammerstein:
    """Wiener-Hammerstein analysis probe q = h_out * phi(h_in * x).

    phi is a cubic B-spline on a fixed knot grid over [-span, span], in units of the
    RMS of h_in * x on the estimation data, linear outside; no neural network. The
    parametrisation is ambiguous (gain and delay move between h_in, phi and h_out):
    parameters are never compared, only rendered outputs. Fit by alternating least
    squares (h_out, then phi, both linear; then a damped Gauss-Newton step on h_in)
    from `starts` random initialisations on the SAME data; their spread is kept."""

    taps_in: int = 16
    taps_out: int = 64
    knots: int = 12
    span: float = 4.0
    starts: int = 4
    iterations: int = 30
    ridge: float = 1e-6
    seed: int = 0

    @property
    def kind(self) -> str:
        return "wiener_hammerstein"

    def version(self) -> str:
        text = json.dumps({"kind": self.kind, **asdict(self)}, sort_keys=True)
        return f"{self.kind}-{hashlib.sha256(text.encode()).hexdigest()[:12]}"


@dataclass
class WhState:
    h_in: np.ndarray
    coef: np.ndarray
    h_out: np.ndarray
    scale: float  # u = h_in * x / scale: RMS units of the first h_in on the fit data
    cost: float = np.inf
    history: list = field(default_factory=list)  # cost after each refit and each step


@dataclass
class WienerHammersteinFit:
    probe: WienerHammerstein
    states: list  # one per start, lowest cost first


def _spline_basis(u: np.ndarray, probe: WienerHammerstein) -> np.ndarray:
    """(n, knots + 2) cubic B-spline design on [-span, span], linear extrapolation."""
    from scipy.interpolate import BSpline

    inner = np.linspace(-probe.span, probe.span, probe.knots)
    t = np.r_[[inner[0]] * 3, inner, [inner[-1]] * 3]
    edges = np.array([inner[0], inner[-1]])
    count = len(t) - 4
    slope = BSpline(t, np.eye(count), 3).derivative()(edges)  # exact basis slopes
    design = BSpline.design_matrix(np.clip(u, *edges), t, 3).toarray()
    low, high = u < edges[0], u > edges[1]
    design[low] += (u[low] - edges[0])[:, None] * slope[0]
    design[high] += (u[high] - edges[1])[:, None] * slope[1]
    return design


def _causal(signal: np.ndarray, h: np.ndarray) -> np.ndarray:
    return fftconvolve(signal, h)[: len(signal)]


def wh_render(state: WhState, probe: WienerHammerstein, x: np.ndarray) -> np.ndarray:
    """Probe output on x, zero initial state."""
    u = _causal(x, state.h_in) / state.scale
    return _causal(_spline_basis(u, probe) @ state.coef, state.h_out)


def _ridge_solve(gram: np.ndarray, rhs: np.ndarray, ridge: float) -> np.ndarray:
    diagonal = np.diag(gram).mean() or 1.0
    return solve(gram + ridge * diagonal * np.eye(len(gram)), rhs, assume_a="pos")


def _fit_h_out(state, probe, v, y, keep) -> None:
    """h_out by ridge least squares on the same kept samples as every other step."""
    columns = np.column_stack(
        [np.r_[np.zeros(m), v[: len(v) - m]] for m in range(probe.taps_out)]
    )[keep]
    state.h_out = _ridge_solve(columns.T @ columns, columns.T @ y[keep], probe.ridge)


def _fit_phi(state, probe, design, y, keep) -> None:
    columns = np.column_stack(
        [_causal(design[:, j], state.h_out) for j in range(design.shape[1])]
    )[keep]
    state.coef = _ridge_solve(columns.T @ columns, columns.T @ y[keep], probe.ridge)


def _step_h_in(state, probe, x, y, keep, damping: float) -> float:
    """One damped Gauss-Newton step on h_in (scale held), accepted if it lowers the
    cost; returns the new damping."""
    u = _causal(x, state.h_in) / state.scale
    delta = 1e-4
    slope = (
        _spline_basis(u + delta, probe) @ state.coef
        - _spline_basis(u - delta, probe) @ state.coef
    ) / (2 * delta)
    residual = (y - wh_render(state, probe, x))[keep]
    lagged = [np.r_[np.zeros(m), x[: len(x) - m]] for m in range(probe.taps_in)]
    jac = np.column_stack(
        [_causal(slope * column / state.scale, state.h_out) for column in lagged]
    )[keep]
    gram, rhs = jac.T @ jac, jac.T @ residual
    for _ in range(8):
        step = solve(gram + damping * np.diag(np.diag(gram) + 1e-300), rhs)
        trial = WhState(state.h_in + step, state.coef, state.h_out, state.scale)
        cost = float(np.mean((y - wh_render(trial, probe, x))[keep] ** 2))
        if cost < state.cost:
            state.h_in, state.cost = trial.h_in, cost
            return damping / 3
        damping *= 4
    return damping


def _normalize(state: WhState) -> None:
    """Unit-norm h_in with the scale divided by the same norm: u = h_in * x / scale,
    hence the rendered output, is unchanged (a gain step of h_in is kept)."""
    norm = float(np.linalg.norm(state.h_in))
    state.h_in, state.scale = state.h_in / norm, state.scale / norm


def _fit_start(probe, x, y, rng) -> WhState:
    keep = slice(memory(probe), None)
    h_in = np.r_[1.0, 0.1 * rng.standard_normal(probe.taps_in - 1)]
    grid = np.linspace(-probe.span, probe.span, probe.knots + 2)
    h_in = h_in / np.linalg.norm(h_in)
    scale = float(np.sqrt(np.mean(_causal(x, h_in) ** 2))) or 1.0  # spline in RMS units
    state = WhState(
        h_in, grid * (1 + 0.1 * rng.standard_normal(len(grid))), None, scale
    )
    damping = 1e-2
    for _ in range(probe.iterations):
        _normalize(state)
        design = _spline_basis(_causal(x, state.h_in) / state.scale, probe)
        _fit_h_out(state, probe, design @ state.coef, y, keep)
        _fit_phi(state, probe, design, y, keep)
        state.cost = float(np.mean((y - wh_render(state, probe, x))[keep] ** 2))
        state.history.append(state.cost)
        damping = _step_h_in(state, probe, x, y, keep, damping)
        state.history.append(state.cost)
    return state


def fit_wh(probe: WienerHammerstein, x: np.ndarray, y: np.ndarray):
    """Every start fitted on the same data; states sorted by final cost."""
    rng = np.random.default_rng(probe.seed)
    states = [_fit_start(probe, x, y, rng) for _ in range(probe.starts)]
    return WienerHammersteinFit(probe, sorted(states, key=lambda s: s.cost))


def memory(probe) -> int:
    """Samples after which the probe has forgotten its input (zero gap between
    estimation segments): the FIR lengths, or the Laguerre bank's decay to 1e-9."""
    if isinstance(probe, Probe):
        return probe.taps
    if isinstance(probe, WienerHammerstein):
        return probe.taps_in + probe.taps_out
    impulse = np.zeros(8_192)
    impulse[0] = 1.0
    bank = np.abs(laguerre(impulse, probe.size, probe.pole)).max(axis=1)
    return int(np.nonzero(bank > 1e-9 * bank.max())[0][-1]) + 1


def fit_probe(probe, x: np.ndarray, y: np.ndarray):
    if isinstance(probe, VolterraLaguerre):
        return fit_volterra(probe, x, y)
    if isinstance(probe, WienerHammerstein):
        return fit_wh(probe, x, y)
    return fit(probe, x, y)


def probe_contributions(result, x: np.ndarray) -> dict:
    if isinstance(result, VolterraFit):
        return volterra_contributions(result, x)
    if isinstance(result, WienerHammersteinFit):
        best = wh_render(result.states[0], result.probe, x)
        return {"wiener_hammerstein": best}
    return contributions(result, x)


def start_spread(result, x: np.ndarray) -> float:
    """Largest RMS gap, on x, between the best start and any other start of the same
    fit on the same data (0 for probes fitted in closed form)."""
    if not isinstance(result, WienerHammersteinFit):
        return 0.0
    best = wh_render(result.states[0], result.probe, x)
    return max(
        float(np.sqrt(np.mean((wh_render(s, result.probe, x) - best) ** 2)))
        for s in result.states
    )
