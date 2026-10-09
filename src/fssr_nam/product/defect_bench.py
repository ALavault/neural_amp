"""Known-defect bench to calibrate deviation scores (product lane, synthetic only).

A DSP reference chain and variants that differ by one controlled defect. They exist to
measure what each score detects; they are not emulators and never stand for hardware.
Inputs may be real DI recordings; outputs are always computed here.

Chain: first-order high-pass (pre) -> asymmetric saturation tanh(g u + b) - tanh(b),
optionally with an envelope that lowers g (sag) -> added polynomial / cross terms ->
second-order low-pass (post) -> gain, peaking EQ at 2 kHz, first-order allpass,
fractional delay, echo, additive noise.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq
from scipy.signal import butter, lfilter

from fssr_nam.alignment.delay import apply_fractional_delay


@dataclass(frozen=True)
class Chain:
    pre_hz: float = 120.0
    drive: float = 4.0
    bias: float = 0.15
    a2: float = 0.0
    a3: float = 0.0
    a7: float = 0.0  # beyond a degree-5 probe
    cross: float = 0.0  # u[n] u[n - CROSS_LAG]
    sag: float = 0.0  # drive divided by 1 + sag * envelope
    release_ms: float = 50.0
    post_hz: float = 5_000.0
    gain: float = 1.0
    eq_db: float = 0.0
    allpass: float = 0.3
    delay: float = 0.0
    echo: float = 0.0  # at ECHO_S, beyond the probes' memory
    noise_db: float | None = None  # additive white noise re output RMS


CROSS_LAG = 5
ECHO_S = 0.03
ATTACK_MS = 2.0
STATIC = Chain()
DYNAMIC = Chain(sag=2.0)

# defect -> (reference chain, field, three severities)
DEFECTS = {
    "gain": (STATIC, "gain", (1.005, 1.02, 1.06)),
    "coloration_db": (STATIC, "eq_db", (0.5, 1.5, 3.0)),
    "delay_samples": (STATIC, "delay", (0.1, 0.3, 1.0)),
    "allpass": (STATIC, "allpass", (0.32, 0.4, 0.6)),
    "quadratic": (STATIC, "a2", (0.02, 0.1, 0.3)),
    "cubic": (STATIC, "a3", (0.02, 0.1, 0.3)),
    "redistribution_bias": (STATIC, "bias", (0.2, 0.3, 0.45)),
    "prefilter_hz": (STATIC, "pre_hz", (150.0, 250.0, 500.0)),
    "cross_term": (STATIC, "cross", (0.05, 0.2, 0.5)),
    "release_ms": (DYNAMIC, "release_ms", (60.0, 100.0, 200.0)),
    "noise_db": (STATIC, "noise_db", (-80.0, -60.0, -40.0)),
    "echo_beyond_memory": (STATIC, "echo", (0.01, 0.05, 0.2)),
    "order_beyond_degree": (STATIC, "a7", (0.1, 0.5, 2.0)),
}


THD_LEVEL_DB = -10.0  # peak level of the 1 kHz sine where redistribution keeps THD


def variant(defect: str, severity: float) -> tuple[Chain, Chain]:
    """(reference, variant) for one defect at one severity. The redistribution defect
    changes the asymmetry and retunes the drive so that THD stays that of the
    reference at THD_LEVEL_DB: the harmonics move between orders, not in total."""
    base, name, _ = DEFECTS[defect]
    changed = replace(base, **{name: severity})
    if defect == "redistribution_bias":
        target = thd(base, THD_LEVEL_DB)
        drive = brentq(
            lambda d: thd(replace(changed, drive=d), THD_LEVEL_DB) - target,
            0.2 * base.drive,
            5.0 * base.drive,
            xtol=1e-10,
        )
        changed = replace(changed, drive=drive)
    return base, changed


def thd(chain: Chain, level_db: float, fs: float = 48_000.0) -> float:
    """THD (harmonics 2..10 over the fundamental) of a 1 kHz sine at level_db peak."""
    n = 4_800  # 100 periods
    t = np.arange(round(0.2 * fs) + n) / fs
    y = apply(chain, 10 ** (level_db / 20) * np.sin(2 * np.pi * 1_000 * t), fs)
    spectrum = np.abs(np.fft.rfft(y[-n:]))
    harmonics = spectrum[100 * np.arange(1, 11)]
    return float(np.sqrt(np.sum(harmonics[1:] ** 2)) / harmonics[0])


def envelope(u: np.ndarray, fs: float, release_ms: float) -> np.ndarray:
    """Peak follower of |u|, attack ATTACK_MS, release release_ms."""
    up = 1 - np.exp(-1 / (ATTACK_MS * 1e-3 * fs))
    down = 1 - np.exp(-1 / (release_ms * 1e-3 * fs))
    out, state = np.empty_like(u), 0.0
    for i, value in enumerate(np.abs(u)):
        state += (up if value > state else down) * (value - state)
        out[i] = state
    return out


def saturate(chain: Chain, u: np.ndarray, fs: float) -> np.ndarray:
    drive = chain.drive
    if chain.sag:
        drive = drive / (1 + chain.sag * envelope(u, fs, chain.release_ms))
    v = np.tanh(drive * u + chain.bias) - np.tanh(chain.bias)
    v += chain.a2 * u**2 + chain.a3 * u**3 + chain.a7 * u**7
    if chain.cross:
        v += chain.cross * u * np.r_[np.zeros(CROSS_LAG), u[:-CROSS_LAG]]
    return v


def peaking(fs: float, gain_db: float, f0: float = 2_000.0, q: float = 1.0):
    """RBJ peaking biquad; gain 0 dB is the identity."""
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / fs
    alpha = np.sin(w) / (2 * q)
    b = [1 + alpha * a, -2 * np.cos(w), 1 - alpha * a]
    den = [1 + alpha / a, -2 * np.cos(w), 1 - alpha / a]
    return b, den


def colour(chain: Chain, w: np.ndarray, fs: float) -> np.ndarray:
    w = chain.gain * lfilter(*peaking(fs, chain.eq_db), w)
    w = lfilter([chain.allpass, 1.0], [1.0, chain.allpass], w)
    if chain.delay:
        w = apply_fractional_delay(w, chain.delay)
    if chain.echo:
        lag = round(ECHO_S * fs)
        w = w + chain.echo * np.r_[np.zeros(lag), w[:-lag]]
    return w


def apply(chain: Chain, x: np.ndarray, fs: float, seed: int = 0) -> np.ndarray:
    """Output of the chain for input x, zero initial state."""
    u = lfilter(*butter(1, chain.pre_hz, "highpass", fs=fs), x)
    v = saturate(chain, u, fs)
    w = colour(chain, lfilter(*butter(2, chain.post_hz, fs=fs), v), fs)
    if chain.noise_db is not None:
        level = 10 ** (chain.noise_db / 20) * np.sqrt(np.mean(w**2))
        w = w + level * np.random.default_rng(seed).standard_normal(len(w))
    return w


# Excitations: explicit level (RMS or peak, dBFS) and seed. Families are analysed
# separately; within a family, estimation and reading use different seeds, and the DI
# uses different source recordings (di_train sources for A, di_test sources for B).
LEVEL_KIND = {
    "noise": "rms",
    "multisine": "rms",
    "di": "rms",
    "sweep": "peak",
    "two_tone": "peak",
    "plucks": "peak",
}
LEVELS_DB = {"rms": (-30.0, -20.0, -12.0), "peak": (-20.0, -10.0, -3.0)}
DI_SOURCE = "datasets/raw/internal/valvestate_vs100r/reamp_valvestate_192k.wav"


def at_level(x: np.ndarray, level_db: float, kind: str) -> np.ndarray:
    current = np.sqrt(np.mean(x**2)) if kind == "rms" else np.max(np.abs(x))
    return x * 10 ** (level_db / 20) / current


def synchronized_sweep(seconds: float, fs: float, f0=20.0, f1=20_000.0):
    """Novak's synchronized swept sine: L f0 integer, so every harmonic response
    starts in phase. Returns the signal and its `Sweep` (product.hammerstein)."""
    from fssr_nam.product.hammerstein import Sweep

    lag = max(1, round(f0 * seconds / np.log(f1 / f0))) / f0
    duration = lag * np.log(f1 / f0)
    t = np.arange(round(duration * fs)) / fs
    x = np.sin(2 * np.pi * f0 * lag * (np.exp(t / lag) - 1))
    return x, Sweep(f0=f0, f1=f1, seconds=duration)


def two_tone_signal(seconds: float, fs: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f1 = rng.uniform(300.0, 1_500.0)
    f2 = f1 * rng.uniform(1.08, 1.5)
    t = np.arange(round(seconds * fs)) / fs
    phases = rng.uniform(0, 2 * np.pi, 2)
    return np.sin(2 * np.pi * f1 * t + phases[0]) + np.sin(
        2 * np.pi * f2 * t + phases[1]
    )


def multisine_signal(seconds: float, fs: float, seed: int) -> np.ndarray:
    """Random-phase odd multisine, 1 Hz grid, 21 Hz-19.999 kHz; one excited odd line in
    four is left out as a detection line for even and odd distortion."""
    period = round(fs)
    lines = np.arange(21, 20_000, 2)
    lines = lines[np.arange(len(lines)) % 4 != 3]
    spectrum = np.zeros(period // 2 + 1, complex)
    phases = np.random.default_rng(seed).uniform(0, 2 * np.pi, len(lines))
    spectrum[lines] = np.exp(1j * phases)
    one = np.fft.irfft(spectrum, period)
    return np.tile(one, int(np.ceil(seconds)))[: round(seconds * fs)]


def di_signal(part: str, fs: float) -> np.ndarray:
    """VS100R source DI section (`di_train` or `di_test`), resampled to fs."""
    import soundfile as sf
    from scipy.signal import resample_poly

    from fssr_nam.product.vs100r import RATE, ROOT, section

    entry = section(part)
    x = sf.read(
        ROOT / DI_SOURCE,
        dtype="float64",
        start=entry["start"],
        stop=entry["start"] + entry["samples"],
    )[0]
    return resample_poly(x, round(fs), RATE)


def excitation(family: str, level_db: float, seconds: float, seed: int, fs: float):
    """One excitation at an explicit level. For "di", seed 0 is the estimation part
    (di_train sources) and seed 1 the reading part (di_test sources)."""
    from fssr_nam.data.excitations import colored_noise, procedural_plucks

    if family == "sweep":
        x = synchronized_sweep(seconds * (1 + 0.1 * (seed % 3)), fs)[0]
    elif family == "two_tone":
        x = two_tone_signal(seconds, fs, seed)
    elif family == "multisine":
        x = multisine_signal(seconds, fs, seed)
    elif family == "noise":
        x = colored_noise(round(fs), seconds, seed).astype(np.float64)
    elif family == "plucks":
        x = procedural_plucks(round(fs), seconds, seed).astype(np.float64)
    elif family == "di":
        x = di_signal("di_train" if seed == 0 else "di_test", fs)
    else:
        raise ValueError(f"unknown family: {family}")
    return at_level(x, level_db, LEVEL_KIND[family])
