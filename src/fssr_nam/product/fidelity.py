"""Behavioural deviation scores between a reference and an evaluated output.

`analyze_pair` fits one analysis probe (`fssr_nam.product.probes`) separately to the
reference and to the evaluated output on the estimation segments A, renders both fits
on each reading segment B, and reports, per segment:
- d_probe = RMS(q_eval - q_ref), the probe's deviation score ("score d'écart", not a
  proven distance), and d_norm = d_probe / max(RMS(y_ref), floor);
- d_audio = RMS(y_eval - y_ref) on the same samples, for comparison;
- r_ref, r_eval: what each probe leaves unexplained, and the part of the difference the
  probe does not represent, RMS((y_eval - y_ref) - (q_eval - q_ref));
- the triangle-inequality check |d_audio - d_probe| <= r_ref + r_eval, which holds on
  these samples only, not on every input;
- decompositions of q_eval - q_ref by branch (basis-conditioned diagnostics, not a
  physical decomposition of the distortion) and by octave band.
No realignment, gain matching or normalisation is applied to either output: a common
acquisition correction belongs upstream, before both reach this function.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.fft import rfft

from fssr_nam.product.probes import (
    Probe,
    VolterraLaguerre,
    WienerHammerstein,
    fit_probe,
    memory,
    probe_contributions,
    start_spread,
)

OCTAVE_EDGES_HZ = (
    22.0,
    44.0,
    88.0,
    177.0,
    355.0,
    710.0,
    1420.0,
    2840.0,
    5680.0,
    11360.0,
)


@dataclass(frozen=True)
class Protocol:
    """Everything that defines a score; frozen on development data before test."""

    probe: Probe | VolterraLaguerre | WienerHammerstein
    estimation: tuple[str, ...]  # segment names used to fit (A)
    reading: tuple[str, ...]  # segment names the fits are rendered on (B)
    skip_s: float = 0.05  # unscored start of each reading segment (zero-state start)
    floor: float = 1e-4  # RMS under which a reference counts as quasi silent
    capacity_warning: float = 0.3  # r / RMS(y) above which the probe misses much
    edges_hz: tuple[float, ...] = field(default=OCTAVE_EDGES_HZ)

    def version(self) -> str:
        text = json.dumps(asdict(self), sort_keys=True)
        return f"protocol-{hashlib.sha256(text.encode()).hexdigest()[:12]}"


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal**2))) if len(signal) else 0.0


def joined(segments: list[np.ndarray], gap: int) -> np.ndarray:
    """Segments end to end with `gap` zeros between them, so that each one starts from
    a zero state for a probe with at most `gap` taps."""
    pieces = []
    for s in segments:
        pieces += [s, np.zeros(gap)]
    return np.concatenate(pieces[:-1])


def band_rms(signal: np.ndarray, fs: float, edges_hz: tuple[float, ...]) -> list:
    """RMS of the signal in each band [edges[i], edges[i+1]) and the last edge to fs/2;
    the squares of all bands plus below the first edge add up to RMS^2 (Parseval)."""
    n = len(signal)
    power = np.abs(rfft(signal)) ** 2
    power[1 : (n + 1) // 2] *= 2.0  # one-sided spectrum, DC and Nyquist counted once
    freqs = np.arange(len(power)) * fs / n
    bounds = [*edges_hz, fs / 2 + 1.0]
    return [
        float(np.sqrt(power[(freqs >= lo) & (freqs < hi)].sum()) / n)
        for lo, hi in pairwise(bounds)
    ]


def cached_fit(probe, x: np.ndarray, y: np.ndarray, cache_dir: Path | None):
    """fit_probe, reused from cache_dir when the probe version and the exact estimation
    data (x and y bytes) are unchanged: a reference signature is fitted once."""
    if cache_dir is None:
        return fit_probe(probe, x, y)
    digest = hashlib.sha256(probe.version().encode())
    digest.update(np.ascontiguousarray(x, np.float64).tobytes())
    digest.update(np.ascontiguousarray(y, np.float64).tobytes())
    path = Path(cache_dir) / f"{digest.hexdigest()}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    result = fit_probe(probe, x, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(result))
    return result


def fits(protocol: Protocol, x: dict, y_ref: dict, y_eval: dict, cache_dir=None):
    """The probe fitted on the estimation segments, once per output; the reference
    fit can come from cache_dir."""
    gap = memory(protocol.probe)
    xa = joined([x[name] for name in protocol.estimation], gap)
    ref = joined([y_ref[name] for name in protocol.estimation], gap)
    ev = joined([y_eval[name] for name in protocol.estimation], gap)
    return cached_fit(protocol.probe, xa, ref, cache_dir), fit_probe(
        protocol.probe, xa, ev
    )


def deviation(q_ref: dict, q_eval: dict, keep: slice) -> dict:
    """RMS of the difference of each contribution on the scored samples."""
    return {k: rms(q_eval[k][keep] - q_ref[k][keep]) for k in q_ref}


def read_segment(protocol: Protocol, fitted, x, y_ref, y_eval, fs: float) -> dict:
    """All readings on one reading segment."""
    keep = slice(round(protocol.skip_s * fs), None)
    parts_ref = probe_contributions(fitted[0], x)
    parts_eval = probe_contributions(fitted[1], x)
    q_ref, q_eval = sum(parts_ref.values()), sum(parts_eval.values())
    dq, dy = (q_eval - q_ref)[keep], (y_eval - y_ref)[keep]
    scale = rms(y_ref[keep])
    d_probe, d_audio = rms(dq), rms(dy)
    r_ref, r_eval = rms((y_ref - q_ref)[keep]), rms((y_eval - q_eval)[keep])
    return {
        "samples": len(dq),
        "rms_ref": scale,
        "quasi_silent": scale < protocol.floor,
        "d_probe": d_probe,
        "d_norm": d_probe / max(scale, protocol.floor),
        "d_audio": d_audio,
        "d_audio_norm": d_audio / max(scale, protocol.floor),
        "r_ref": r_ref,
        "r_eval": r_eval,
        "unrepresented": rms(dy - dq),
        "bound": {
            "gap": abs(d_audio - d_probe),
            "limit": r_ref + r_eval,
            "holds": abs(d_audio - d_probe) <= (r_ref + r_eval) * (1 + 1e-9) + 1e-15,
        },
        "by_branch": deviation(parts_ref, parts_eval, keep),
        # same data, other starts (Wiener-Hammerstein only; 0 for closed-form fits)
        "start_spread_ref": start_spread(fitted[0], x),
        "start_spread_eval": start_spread(fitted[1], x),
        "by_band": {
            "edges_hz": list(protocol.edges_hz),
            "probe": band_rms(dq, fs, protocol.edges_hz),
            "audio": band_rms(dy, fs, protocol.edges_hz),
        },
    }


def warnings_for(protocol: Protocol, name: str, reading: dict) -> list[str]:
    out = []
    if reading["quasi_silent"]:
        out.append(f"{name}: référence quasi silencieuse, normalisée par le plancher")
    for side in ("ref", "eval"):
        share = reading[f"r_{side}"] / max(reading["rms_ref"], protocol.floor)
        if share > protocol.capacity_warning:
            out.append(
                f"{name}: la sonde laisse {share:.2f} de la sortie {side} "
                "inexpliquée ; un faible score n'y garantit pas la fidélité"
            )
    spread = max(reading["start_spread_ref"], reading["start_spread_eval"])
    if spread > 0.5 * reading["d_probe"]:
        out.append(
            f"{name}: dispersion entre initialisations ({spread:.2e}) comparable "
            "à l'écart mesuré : sonde peu fiable ici, pas de classement"
        )
    if not reading["bound"]["holds"]:
        out.append(f"{name}: borne triangulaire violée (erreur de calcul)")
    return out


def analyze_pair(
    x: dict,
    y_ref: dict,
    y_eval: dict,
    fs: float,
    protocol: Protocol,
    metadata=None,
    cache_dir: Path | None = None,
) -> dict:
    """Deviation scores of y_eval against y_ref for one probe protocol.

    x, y_ref, y_eval map segment names to arrays of equal length per segment; the
    protocol names which segments estimate (A) and which are read (B). metadata
    (level, settings, session, calibration, initial state) is passed through. With
    cache_dir, the reference signature is reused while its protocol and data hold.
    """
    fitted = fits(protocol, x, y_ref, y_eval, cache_dir)
    readings, warnings = {}, []
    for name in protocol.reading:
        readings[name] = read_segment(
            protocol, fitted, x[name], y_ref[name], y_eval[name], fs
        )
        warnings += warnings_for(protocol, name, readings[name])
    return {
        "protocol": protocol.version(),
        "probe": protocol.probe.version(),
        "fs": fs,
        "metadata": metadata or {},
        "readings": readings,
        "warnings": warnings,
    }


@dataclass(frozen=True)
class HarmonicProtocol:
    """Harmonic-response signature D_k from an exponential sweep (Farina; Novak).

    Harmonic ORDER k (output component at k f) is not polynomial DEGREE n; the
    Hammerstein filters G_n follow from D_k through the mixing matrix measured by
    `product.hammerstein.calibration`. The score compares D_k, which are measured
    directly; G_n are an identification on top of them."""

    order: int = 5
    nfft: int = 8192
    band_hz: tuple[float, float] = (100.0, 8_000.0)
    floor_db: float = -80.0  # re max |D_1| of the reference: common observability floor

    def version(self) -> str:
        text = json.dumps(asdict(self), sort_keys=True)
        return f"harmonic-{hashlib.sha256(text.encode()).hexdigest()[:12]}"


def harmonic_signature(x, y, fs: float, sweep, protocol: HarmonicProtocol, m=None):
    """D_k (and G_n) of y driven by the sweep x; pass m to reuse a calibration."""
    from fssr_nam.product.hammerstein import calibration, identify

    if m is None:
        m = calibration(x, lambda s: s, round(fs), sweep, protocol.order, protocol.nfft)
    return identify(
        y, x, lambda s: s, round(fs), sweep, protocol.order, protocol.nfft, m
    ), m


def order_deviation(a, b, scale: float, floor: float) -> dict:
    """Complex and magnitude deviation of one harmonic response, on observable bins.
    A component present only in the evaluated output is counted, never masked."""
    observed = (np.abs(a) > floor) | (np.abs(b) > floor)
    both = (np.abs(a) > floor) & (np.abs(b) > floor)
    only_eval = int(np.sum((np.abs(b) > floor) & (np.abs(a) <= floor)))
    only_ref = int(np.sum((np.abs(a) > floor) & (np.abs(b) <= floor)))
    complex_dev = rms(np.abs(b - a)[observed]) / scale if observed.any() else 0.0
    level = 20 * np.log10(np.abs(b[both]) / np.abs(a[both])) if both.any() else []
    return {
        "complex_rel": float(complex_dev),
        "magnitude_db_rms": rms(np.asarray(level)) if len(level) else 0.0,
        "level_ref_db": 20 * np.log10(max(rms(np.abs(a)), 1e-300) / scale),
        "level_eval_db": 20 * np.log10(max(rms(np.abs(b)), 1e-300) / scale),
        "bins_observed": int(observed.sum()),
        "bins_only_eval": only_eval,
        "bins_only_ref": only_ref,
    }


def analyze_sweep_pair(x, y_ref, y_eval, fs: float, sweep, protocol: HarmonicProtocol):
    """Harmonic-response deviation of y_eval against y_ref for one sweep x."""
    ref, m = harmonic_signature(x, y_ref, fs, sweep, protocol)
    ev, _ = harmonic_signature(x, y_eval, fs, sweep, protocol, m)
    band = (ref.freqs >= protocol.band_hz[0]) & (ref.freqs <= protocol.band_hz[1])
    scale = rms(np.abs(ref.d[0][band]))
    floor = 10 ** (protocol.floor_db / 20) * np.abs(ref.d[0][band]).max()
    orders = {
        f"order_{k + 1}": order_deviation(ref.d[k][band], ev.d[k][band], scale, floor)
        for k in range(protocol.order)
    }
    total = float(np.sqrt(sum(v["complex_rel"] ** 2 for v in orders.values())))
    return {"protocol": protocol.version(), "total_complex_rel": total, **orders}
