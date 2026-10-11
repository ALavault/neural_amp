#!/usr/bin/env python3
"""Lead 2: Wiener-Hammerstein features fitted by gradient on a reference take, then
frozen; the probe is a linear readout on them (exploration, CPU only).

Question: the frozen probes leave 0.48 (gain 0) to 0.80 (gains 2.5-7.5) of the Nolly
output unexplained (baseline.json). Linear-in-parameter structures with fixed
features (more memory, higher degree, saturating bases, Wiener with a BLA or a fixed
high-pass prefilter) plateaued near 0.68-0.75 at G >= 2.5 (scratch study). Does a
nonlinearity placed between LEARNED filters, fitted end to end by gradient on the
reference take and then frozen, give a probe that represents the other takes better,
while staying linear in the target (fit(y_eval) - fit(y_ref) = fit(y_eval - y_ref))?

Model (`WH`, torch, float32): x / RMS(x_A) -> S stages [1x1 channel mix -> causal
depthwise FIR of `taps` per channel -> per-channel static nonlinearity, a sum of
`units` tanh with learned gain, bias and weight] -> post FIR (256 taps) from the last
stage's C channels, plus a direct 256-tap FIR on x and a constant. Trained on random
segments of A = di_train of the reference take (MSE on the output scaled by its RMS
on A), early-stopped on V = di_validation (full-section ratio every EVAL steps).

Probe (`Readout`): the trained model's last-stage channels, frozen, plus x, each
through its own 256-tap FIR, a constant, ridge 1e-4 relative: h.FeatureProbe with a
Gram matrix computed once per frozen feature set (same arithmetic, checked against
h.FeatureProbe). The features read inputs["x"] only, never y_ref.

Protocol: features trained on take 20 (G5, centre; takes 1 and 39 are its repeats,
1.4e-4 of the RMS apart). Selection (config, seed) on V only: mean readout capacity
on V over the 5 H2 gain takes. The automatic selectors (config and seed, drive set)
read V only; the full model's own B ratio on take 20 is computed with each run and
logged before selection (not used by the selector); the readout B reads come after
selection. Labels: take 20 (and its repeats 1, 39) with features from take
20 is a nonlinear fit to that take, not a frozen-probe reading; the per-take models
("upper") are nonlinear in y. Gate-4 takes (EXCLUDED) are measured and flagged.

Run from the repository root, CPU, 3 threads (long: nohup setsid nice -n 10):
    uv run python diagnosis/probe_capacity/lead2_wh_gradient_frozen.py run
    ... jvm        # JVM410H OD1 key number (Q-0014), needs `run` first
    ... drives     # drive bank: frozen features at several input scalings
    ... multitake  # shared features trained on A of the 5 gain takes
    ... transfer   # features of each per-take model read across the gain axis
    ... summary    # derived tables (spread, A/B gap, comparison per gain)
Trained models are cached in tmp/probe_capacity/lead2/ (outside the repository);
every number reported goes to lead2_wh_gradient_frozen.json, written after each block.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.fft import irfft, next_fast_len, rfft
from scipy.linalg import cho_factor, cho_solve
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as h

from fssr_nam.product.probes import Probe, normal_equations, regularized

torch.set_num_threads(3)
HERE = h.HERE
OUT = HERE / "lead2_wh_gradient_frozen.json"
STORE = h.CACHE / "lead2"
REF_TAKE = 20  # G5 centre; 1 and 39 are its repeats
CONFIGS = {
    "wh1": {"stages": 1, "channels": 8, "taps": 128, "units": 4},
    "wh2": {"stages": 2, "channels": 8, "taps": 128, "units": 4},
    "wh3": {"stages": 3, "channels": 8, "taps": 128, "units": 4},
}
SEEDS = {"wh1": (0, 1, 2), "wh2": (0, 1, 2), "wh3": (0, 1, 2)}
STEPS, BATCH, SEGMENT, EVAL, LR = 1500, 8, 16384, 100, 3e-3
POST_TAPS = 256
READOUT_TAPS, READOUT_RIDGE = 256, 1e-4
UPPER_TAKES = (30, 16, 6, 7)  # per-take models at the other H2 gains
# Scratch study (session scratchpad capacity_split.json / run_a.json), fit A read B,
# same sections: Wiener with a BLA prefilter and with a fixed 300 Hz high-pass.
SCRATCH = {
    "wiener_bla_256": {"0": 0.3809, "2.5": 0.7285, "5": 0.7483, "7.5": 0.7476,
                       "10": 0.6832},
    "wiener_hp300_256": {"0": 0.5204, "2.5": 0.6141, "5": 0.6919, "7.5": 0.6961,
                         "10": 0.7104},
}  # fmt: skip


def log(*args) -> None:
    print(time.strftime("%H:%M:%S"), *args, flush=True)


# ---------------------------------------------------------------- model


def causal_fft_conv(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """x (batch, C, L) with kernel (C, K) per channel, or kernel (O, C, K) summed
    over C; causal, zero initial state, first L samples."""
    length = x.shape[-1]
    n = 1 << (length + kernel.shape[-1] - 1).bit_length()
    spectrum = torch.fft.rfft(x, n)
    response = torch.fft.rfft(kernel, n)
    if kernel.dim() == 2:
        return torch.fft.irfft(spectrum * response, n)[..., :length]
    mixed = torch.einsum("bcf,ocf->bof", spectrum, response)
    return torch.fft.irfft(mixed, n)[..., :length]


class WH(nn.Module):
    """Multi-channel Wiener-Hammerstein cascade on x scaled to unit RMS."""

    def __init__(
        self, stages: int, channels: int, taps: int, units: int, seed: int, heads=1
    ) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        c, self.taps = channels, taps
        self.mix = nn.ParameterList()
        self.fir = nn.ParameterList()
        self.gain = nn.ParameterList()
        self.bias = nn.ParameterList()
        self.weight = nn.ParameterList()
        self.offset = nn.ParameterList()
        delta = torch.zeros(c, taps)
        delta[:, 0] = 1
        for s in range(stages):
            if s == 0:
                mix = torch.randn(c, 1, generator=gen)
            else:
                mix = torch.eye(c) + 0.1 * torch.randn(c, c, generator=gen)
            self.mix.append(nn.Parameter(mix))
            self.fir.append(
                nn.Parameter(delta + 0.05 * torch.randn(c, taps, generator=gen))
            )
            # drives spread from 0.3 to 30 (log), shuffled per channel
            drive = torch.logspace(-0.5, 1.5, c * units)
            drive = drive[torch.randperm(c * units, generator=gen)].view(c, units)
            self.gain.append(nn.Parameter(drive))
            self.bias.append(nn.Parameter(0.3 * torch.randn(c, units, generator=gen)))
            self.weight.append(
                nn.Parameter(torch.randn(c, units, generator=gen) / math.sqrt(units))
            )
            self.offset.append(nn.Parameter(torch.zeros(c)))
        # one output head per target take (heads > 1: shared channels, multi-take)
        post = 0.01 * torch.randn(heads, c, POST_TAPS, generator=gen)
        self.post = nn.Parameter(post)
        direct = 0.01 * torch.randn(heads, 1, POST_TAPS, generator=gen)
        self.direct = nn.Parameter(direct)
        self.constant = nn.Parameter(torch.zeros(heads, 1))
        self.feature_memory = stages * (taps - 1)
        self.memory = self.feature_memory + POST_TAPS - 1

    def channels(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, 1, L) -> last stage's channels (batch, C, L)."""
        z = x
        for mix, fir, a, b, w, o in zip(
            self.mix, self.fir, self.gain, self.bias, self.weight, self.offset,
            strict=True,
        ):  # fmt: skip
            u = causal_fft_conv(torch.einsum("oc,bcl->bol", mix, z), fir)
            v = torch.tanh(a[None, :, :, None] * u[:, :, None, :] + b[None, :, :, None])
            z = (w[None, :, :, None] * v).sum(2) + o[None, :, None]
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.channels(x), x], 1)
        return (
            causal_fft_conv(z, torch.cat([self.post, self.direct], 1)) + self.constant
        )


def chunked(fn, x: np.ndarray, context: int, chunk: int = 1 << 19) -> np.ndarray:
    """Apply a causal map of finite memory `context` to a long 1-D signal by chunks
    with `context` samples of left history (exact; zero state at the start)."""
    out = []
    with torch.no_grad():
        for start in range(0, len(x), chunk):
            lo = max(0, start - context)
            piece = torch.from_numpy(np.ascontiguousarray(x[lo : start + chunk]))
            y = fn(piece.float()[None, None])[0]
            out.append(y[:, start - lo :].double().numpy())
    return np.concatenate(out, axis=-1)


class Trained:
    """A trained WH with its input and output scales (RMS on the training A); one
    output scale per head."""

    def __init__(self, model: WH, x_scale: float, y_scale) -> None:
        self.model, self.x_scale = model.eval(), x_scale
        self.y_scale = np.atleast_1d(np.asarray(y_scale, dtype=float))

    def output(self, x: np.ndarray) -> np.ndarray:
        """(N,) for one head, (heads, N) otherwise."""
        q = chunked(self.model, x / self.x_scale, self.model.memory)
        q = q * self.y_scale[:, None]
        return q[0] if len(q) == 1 else q

    def channels(self, x: np.ndarray, drive: float = 1.0) -> np.ndarray:
        z = x * (drive / self.x_scale)
        return chunked(self.model.channels, z, self.model.feature_memory)


def train(
    x_a: np.ndarray,
    y_a: np.ndarray,
    x_v: np.ndarray,
    y_v: np.ndarray,
    spec: dict,
    seed: int,
    skip: int = h.SKIP,
) -> tuple[Trained, dict]:
    """Adam on random segments of A, cosine learning rate, best state on V. y_a and
    y_v may be (heads, N): one head per target, loss and V ratio averaged."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    y_a, y_v = np.atleast_2d(y_a), np.atleast_2d(y_v)
    x_scale, y_scale = h.nr.rms(x_a), np.array([h.nr.rms(v) for v in y_a])
    xa = torch.from_numpy(np.asarray(x_a) / x_scale).float()
    ya = torch.from_numpy(y_a / y_scale[:, None]).float()
    model = WH(**spec, seed=seed, heads=len(y_a))
    warm = model.memory
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=LR * 0.03)
    best, best_state, best_step, curve, running = math.inf, None, 0, [], []
    start = time.time()
    for step in range(1, STEPS + 1):
        model.train()
        starts = rng.integers(0, len(xa) - SEGMENT - warm, BATCH)
        xb = torch.stack([xa[s : s + SEGMENT + warm] for s in starts])[:, None]
        yb = torch.stack([ya[:, s : s + SEGMENT + warm] for s in starts])
        err = (model(xb) - yb)[..., warm:]
        power = yb[..., warm:].pow(2).mean((0, 2))
        loss = (err.pow(2).mean((0, 2)) / power).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running.append(loss.item())
        if step % EVAL == 0 or step == STEPS:
            model.eval()
            trained = Trained(model, x_scale, y_scale)
            q = np.atleast_2d(trained.output(x_v))
            v = float(
                np.mean([residual(a, b, skip) for a, b in zip(y_v, q, strict=True)])
            )
            curve.append(
                {"step": step, "train_esr_mean": float(np.mean(running)), "v_ratio": v}
            )
            running = []
            if v < best:
                best, best_step = v, step
                best_state = copy.deepcopy(model.state_dict())
            log(f"  step {step} train ESR {curve[-1]['train_esr_mean']:.4f} V {v:.4f}")
    model.load_state_dict(best_state)
    info = {
        "best_step": best_step,
        "best_v_ratio": best,
        "seconds": time.time() - start,
        "curve": curve,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    return Trained(model, x_scale, y_scale), info


def load_model(saved: dict, spec: dict, seed: int) -> Trained:
    """A cached model; single-head states saved with a scalar constant load too."""
    state = dict(saved["state"])
    heads = state["post"].shape[0]
    state["constant"] = state["constant"].reshape(heads, 1)
    model = WH(**spec, seed=seed, heads=heads)
    model.load_state_dict(state)
    return Trained(model, saved["x_scale"], saved["y_scale"])


def residual(y: np.ndarray, q: np.ndarray, skip: int = h.SKIP) -> float:
    """RMS((y - q)[skip:]) / max(RMS(y[skip:]), floor): h.ratio at any skip."""
    return h.nr.rms((y - q)[skip:]) / max(h.nr.rms(y[skip:]), h.FLOOR)


def trained_on_take(take: int, name: str, seed: int) -> tuple[Trained, dict]:
    """Model fitted on A of a Nolly take, V early stopping; cached in STORE."""
    path = STORE / f"take_{take:02d}_{name}_s{seed}.pt"
    spec = CONFIGS[name]
    if path.exists():
        saved = torch.load(path, weights_only=False)
        return load_model(saved, spec, seed), saved["info"]
    log(f"train take {take} {name} seed {seed}")
    x = h.load("x", (h.FIT, h.VALID))
    y = h.load(take, (h.FIT, h.VALID))
    trained, info = train(x[h.FIT], y[h.FIT], x[h.VALID], y[h.VALID], spec, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": trained.model.state_dict(),
            "x_scale": trained.x_scale,
            "y_scale": trained.y_scale,
            "info": info,
        },
        path,
    )
    return trained, info


# ---------------------------------------------------------------- frozen probe


class Features:
    """[x] + the frozen model's last-stage channels at each drive (x scaled by
    drive before the model). Reads inputs["x"] only; memoised per signal."""

    def __init__(self, trained: Trained, drives: tuple[float, ...] = (1.0,)) -> None:
        self.trained, self.drives, self.memo = trained, drives, {}

    def __call__(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        x = np.asarray(inputs["x"])
        key = (len(x), float(np.sum(x)), float(np.dot(x, x)))
        if key not in self.memo:
            out = [x]
            for g in self.drives:
                out += list(self.trained.channels(x, g))
            self.memo[key] = out
        return self.memo[key]


class Readout(h.FeatureProbe):
    """h.FeatureProbe whose Gram matrix, its Cholesky factor and the feature spectra
    on the fit section are computed once per fit signal and reused across targets.
    Same scales, constant, ridge and right-hand side as h.FeatureProbe."""

    def __init__(self, features, taps: int = READOUT_TAPS, ridge: float = 1e-4):
        super().__init__(features, taps, ridge)
        self.cache: tuple | None = None

    def fit(self, inputs: dict[str, np.ndarray], target: np.ndarray) -> Readout:
        x = np.asarray(inputs["x"])
        key = (len(x), float(np.sum(x)), float(np.dot(x, x)))
        if self.cache is None or self.cache[0] != key:
            self.cache = None  # free the previous spectra first
            u = self.features(inputs)
            scales = [max(h.nr.rms(f), 1e-300) for f in u]
            probe = Probe(degree=len(u), taps=self.taps, ridge=self.ridge)
            normed = [f / s for f, s in zip(u, scales, strict=True)]
            gram, _ = normal_equations(probe, normed, None)
            size = next_fast_len(len(x) + self.taps)
            spectra = [rfft(f, size) for f in normed]
            factor = cho_factor(regularized(probe, gram))
            self.cache = (key, scales, factor, spectra, size)
        _, scales, factor, spectra, size = self.cache
        target = np.asarray(target)
        spectrum = rfft(target, size)
        rhs = [np.array([np.sum(target)])]
        rhs += [irfft(np.conj(s) * spectrum, size)[: self.taps] for s in spectra]
        self.scales, self.theta = scales, cho_solve(factor, np.concatenate(rhs))
        return self


def read_take(probe: Readout, take: int, sections=(h.READ,)) -> dict[str, np.ndarray]:
    """Fit the readout on A of the take, predict on each section."""
    probe.fit(h.section_inputs(h.FIT), h.load(take, (h.FIT,))[h.FIT])
    return {s: np.asarray(probe.predict(h.section_inputs(s))) for s in sections}


def capacities(probe: Readout, takes, sections) -> dict:
    out = {}
    for t in takes:
        q = read_take(probe, t, sections)
        y = h.load(t, sections)
        out[str(t)] = {s: h.ratio(y[s], q[s]) for s in sections}
        out[str(t)]["excluded"] = t in h.EXCLUDED
    return out


# ---------------------------------------------------------------- experiment


def gains_of(values: dict) -> dict:
    return {f"{g:g}": values[str(t)] for g, t in h.GAIN_TAKES.items()}


def save(result: dict) -> None:
    OUT.write_text(json.dumps(h.nr.finite(result), indent=1) + "\n")


def baselines() -> dict:
    base = json.loads((HERE / "baseline.json").read_text())["probes"]
    out = {
        name: {"capacity_by_gain": base[name]["capacity_by_gain"]}
        for name in ("hammerstein_5", "volterra_laguerre_3")
    }
    for name in out:
        out[name]["capacity_by_level"] = {
            g: [b["capacity"] for b in bins]
            for g, bins in base[name]["capacity_by_level"].items()
        }
        out[name]["difference_share_one_knob"] = {
            f"{s['ref']}->{s['eval']}": s["share_from_x"]
            for s in base[name]["difference_share_one_knob"]
        }
        out[name]["capacity_by_take"] = {
            t: v["value"] for t, v in base[name]["capacity_by_take"].items()
        }
    out["scratch"] = {
        "source": "session scratchpad capacity_split.json and run_a.json (outside "
        "the repository, copied to 4 decimals, not re-run here)",
        **SCRATCH,
    }
    return out


def instrument_check() -> dict:
    """Readout with features x^1..x^5 must reproduce Hammerstein 5 (take 2: 0.8056)
    and equal h.FeatureProbe on the same features."""
    powers = lambda i: [np.asarray(i["x"]) ** k for k in range(1, 6)]  # noqa: E731
    mine = h.output_capacity(Readout(powers), 2)
    theirs = h.output_capacity(h.FeatureProbe(powers), 2)
    return {
        "readout_x1_x5_take2": mine,
        "feature_probe_x1_x5_take2": theirs,
        "frozen_h5_take2": 0.8056428222917209,
        "gap": abs(mine - 0.8056428222917209),
        "pass": abs(mine - theirs) < 1e-9 and abs(mine - 0.8056428222917209) < 1e-9,
    }


def run() -> None:
    result = json.loads(OUT.read_text()) if OUT.exists() else {}
    result["protocol"] = (
        "features: WH trained by Adam on A=di_train of take 20, early stop on "
        "V=di_validation; frozen; readout = [x] + last-stage channels, 256-tap FIR "
        "each, constant, ridge 1e-4 relative, fitted on A of each take, read on "
        "B=di_test (48 kHz, SKIP 2400, floor 1e-4). Selection on V only."
    )
    result["settings"] = {
        "configs": CONFIGS,
        "seeds": SEEDS,
        "steps": STEPS,
        "batch": BATCH,
        "segment": SEGMENT,
        "eval_every": EVAL,
        "lr": LR,
        "post_taps": POST_TAPS,
        "readout_taps": READOUT_TAPS,
        "readout_ridge": READOUT_RIDGE,
        "reference_take": REF_TAKE,
    }
    result["baselines"] = baselines()
    if "instrument_check" not in result:
        result["instrument_check"] = instrument_check()
        log("instrument", result["instrument_check"])
        save(result)
    gain_takes = list(h.GAIN_TAKES.values())
    runs = result.setdefault("runs", {})
    # 1. train on take 20, readout capacity on A and V at the 5 gains (selection)
    for name in CONFIGS:
        for seed in SEEDS[name]:
            key = f"{name}_s{seed}"
            if key in runs and "readout_AV" in runs[key]:
                continue
            trained, info = trained_on_take(REF_TAKE, name, seed)
            y = h.load(REF_TAKE, (h.FIT, h.VALID, h.READ))
            x = h.load("x", (h.FIT, h.VALID, h.READ))
            full = {s: h.ratio(y[s], trained.output(x[s])) for s in y}
            probe = Readout(Features(trained))
            av = capacities(probe, gain_takes, (h.FIT, h.VALID))
            runs[key] = {
                "config": name,
                "seed": seed,
                "train": info,
                "full_model_take20": full,
                "readout_AV": gains_of(av),
                "selection_score_mean_V": float(
                    np.mean([v[h.VALID] for v in av.values()])
                ),
            }
            log(key, "V mean", runs[key]["selection_score_mean_V"], "full", full)
            save(result)
    # 2. selection on V
    chosen = min(runs, key=lambda k: runs[k]["selection_score_mean_V"])
    result["selection"] = {
        "criterion": "min over (config, seed) of the mean readout capacity on V "
        "over the 5 H2 gain takes (30, 16, 20, 6, 7)",
        "chosen": chosen,
        "scores": {k: r["selection_score_mean_V"] for k, r in runs.items()},
    }
    save(result)
    # 3. B reads: every run at the 5 gains (spread), the chosen one everywhere
    for key, r in runs.items():
        if "readout_B" in r:
            continue
        trained, _ = trained_on_take(REF_TAKE, r["config"], r["seed"])
        r["readout_B"] = gains_of(
            capacities(Readout(Features(trained)), gain_takes, (h.READ,))
        )
        log(key, "B", {g: v[h.READ] for g, v in r["readout_B"].items()})
        save(result)
    best = runs[chosen]
    trained, _ = trained_on_take(REF_TAKE, best["config"], best["seed"])
    probe = Readout(Features(trained))
    if "chosen_B" not in result:
        result["chosen_B"] = chosen_reads(probe)
        save(result)
    # 4. per-take models (upper reference, nonlinear in y)
    upper = result.setdefault("upper_per_take", {})
    for take in UPPER_TAKES:
        if str(take) in upper:
            continue
        model, info = trained_on_take(take, best["config"], best["seed"])
        x = h.load("x", (h.FIT, h.VALID, h.READ))
        y = h.load(take, (h.FIT, h.VALID, h.READ))
        own = capacities(Readout(Features(model)), [take], (h.FIT, h.VALID, h.READ))
        upper[str(take)] = {
            "gain": h.ROWS[take]["G"],
            "train": info,
            "full_model": {s: h.ratio(y[s], model.output(x[s])) for s in y},
            "readout_own_features": own[str(take)],
        }
        log("upper", take, upper[str(take)]["full_model"])
        save(result)


def chosen_reads(probe: Readout) -> dict:
    """B reads of the V-selected frozen probe: every take, levels at the 5 gains,
    one-knob difference shares, linearity in y."""
    out = {"capacity_by_take": capacities(probe, sorted(h.ROWS), (h.READ,))}
    log("all takes done")
    x_b = h.section_inputs(h.READ)["x"]
    levels = {}
    for g, t in h.GAIN_TAKES.items():
        q = read_take(probe, t)[h.READ]
        levels[f"{g:g}"] = h.level_capacity_from(x_b, h.load(t, (h.READ,))[h.READ], q)
    out["capacity_by_level"] = levels
    shares = []
    for p in h.pairs(("one_knob",), excluded=True):
        shares.append(
            {
                **p,
                "knob": h.knob(p),
                "setting": h.ROWS[p["eval"]]["setting"],
                "share_from_x": h.difference_share(probe, p["ref"], p["eval"]),
            }
        )
    out["difference_share_one_knob"] = shares
    log("shares done")
    out["linearity_check_1_7"] = h.linearity_check(probe, 1, 7)
    return out


def jvm() -> None:
    """JVM410H OD1 PreampOut (ToneTwist, 44.1 kHz, real_pairs.jvm_audio): A = 0-150 s,
    B = 215-300 s, A/B source independence NOT established, no repeats. Model (the
    config chosen on Nolly V, seed 0) trained on 0-120 s of the centre setting G050,
    early stopping on 120-150 s; readout fitted on 0-150 s, read on B once."""
    from real_pairs import jvm_audio

    result = json.loads(OUT.read_text())
    name, seed = result["runs"][result["selection"]["chosen"]]["config"], 0
    centre = "B050_M050_T050_G050"
    rate, x, y = jvm_audio("PreampOut", centre)
    skip, split = round(0.05 * rate), 120 * rate
    path = STORE / f"jvm_preampout_{centre}_{name}_s{seed}.pt"
    if path.exists():
        saved = torch.load(path, weights_only=False)
        trained, info = load_model(saved, CONFIGS[name], seed), saved["info"]
    else:
        log(f"train jvm {centre} {name} seed {seed}")
        xa, ya = x["A"], y["A"]
        trained, info = train(
            xa[:split], ya[:split], xa[split:], ya[split:], CONFIGS[name], seed, skip
        )
        torch.save(
            {
                "state": trained.model.state_dict(),
                "x_scale": trained.x_scale,
                "y_scale": trained.y_scale,
                "info": info,
            },
            path,
        )
    out = {
        "data": "JVM410H OD1 PreampOut, ToneTwist, 44.1 kHz; A 0-150 s, B 215-300 s "
        "of the same input file (independence not established, no repeats)",
        "config": name,
        "seed": seed,
        "train_split": "0-120 s fit, 120-150 s early stopping, of the centre G050",
        "skip": skip,
        "train": info,
        "full_model_centre": {
            "A_fit_0_120": residual(
                y["A"][:split], trained.output(x["A"][:split]), skip
            ),
            "A_stop_120_150": residual(
                y["A"][split:], trained.output(x["A"][split:]), skip
            ),
            "B": residual(y["B"], trained.output(x["B"]), skip),
        },
        "earlier_results": {
            "source": "diagnosis/probe_metrics/gap_checks_jvm_saturating.json",
            "hammerstein_5": {"G050": 0.7066, "G000": 0.0598, "G100": 0.6569},
            "saturating_basis": {"G050": 0.6721, "G000": 0.0585, "G100": 0.5866},
        },
        "readout": {},
    }
    probe = Readout(Features(trained))
    for setting in (centre, "B050_M050_T050_G100", "B050_M050_T050_G000"):
        _, _, ys = jvm_audio("PreampOut", setting)
        probe.fit({"x": x["A"]}, ys["A"])
        h5 = h.FrozenProbe("hammerstein_5").fit({"x": x["A"]}, ys["A"])
        out["readout"][setting] = {
            "readout_A": residual(ys["A"], probe.predict({"x": x["A"]}), skip),
            "readout_B": residual(ys["B"], probe.predict({"x": x["B"]}), skip),
            "hammerstein_5_B_recomputed": residual(
                ys["B"], h5.predict({"x": x["B"]}), skip
            ),
            "features_from_this_setting": setting == centre,
        }
        log("jvm", setting, out["readout"][setting])
    result = json.loads(OUT.read_text())  # the run may have written meanwhile
    result["jvm"] = out
    save(result)


DRIVE_SETS = ((1.0,), (0.5, 1.0, 2.0), (0.25, 0.5, 1.0, 2.0, 4.0))


def gain_shares(probe: Readout) -> list[dict]:
    """Difference share from x on the 4 one-knob gain pairs."""
    out = []
    for p in h.pairs(("one_knob",)):
        if h.knob(p) == "G":
            share = h.difference_share(probe, p["ref"], p["eval"])
            out.append({**p, "setting": h.ROWS[p["eval"]]["setting"], "share": share})
    return out


def drive_bank() -> None:
    """Frozen features of the V-selected take-20 model at fixed input scalings
    (drives): [x] + channels(x * g) for g in the set; still target-independent, so
    linear in y. Set chosen on V (mean over the 5 gain takes), then B once."""
    result = json.loads(OUT.read_text())
    chosen = result["runs"][result["selection"]["chosen"]]
    trained, _ = trained_on_take(REF_TAKE, chosen["config"], chosen["seed"])
    gain_takes = list(h.GAIN_TAKES.values())
    candidates = {}
    for drives in DRIVE_SETS:
        av = capacities(Readout(Features(trained, drives)), gain_takes, (h.VALID,))
        label = ",".join(f"{g:g}" for g in drives)
        candidates[label] = {
            "drives": list(drives),
            "V": {g: v[h.VALID] for g, v in gains_of(av).items()},
            "mean_V": float(np.mean([v[h.VALID] for v in av.values()])),
        }
        log("drives", label, candidates[label])
    label = min(candidates, key=lambda k: candidates[k]["mean_V"])
    probe = Readout(Features(trained, tuple(candidates[label]["drives"])))
    reads = capacities(probe, gain_takes, (h.FIT, h.READ))
    x_b = h.section_inputs(h.READ)["x"]
    levels = {}
    for g, t in h.GAIN_TAKES.items():
        q = read_take(probe, t)[h.READ]
        levels[f"{g:g}"] = h.level_capacity_from(x_b, h.load(t, (h.READ,))[h.READ], q)
    out = {
        "model": result["selection"]["chosen"],
        "criterion": "min mean readout capacity on V over the 5 gain takes",
        "candidates": candidates,
        "chosen": label,
        "capacity_by_gain": gains_of(reads),
        "capacity_by_level": levels,
        "difference_share_gain_pairs": gain_shares(probe),
    }
    log("drives B", out["capacity_by_gain"])
    result = json.loads(OUT.read_text())
    result["drive_bank"] = out
    save(result)


def multitake() -> None:
    """Shared channels trained jointly on A of the 5 gain takes (one output head per
    take, mean normalised MSE), V early stopping; frozen; readout on every take.
    LABEL: the features saw A of the 5 gain takes (not B); every other take is a
    frozen-probe reading. Config chosen on Nolly V for take 20, seed 0."""
    result = json.loads(OUT.read_text())
    name, seed = result["runs"][result["selection"]["chosen"]]["config"], 0
    takes = list(h.GAIN_TAKES.values())
    path = STORE / f"multi_gain_{name}_s{seed}.pt"
    if path.exists():
        saved = torch.load(path, weights_only=False)
        trained, info = load_model(saved, CONFIGS[name], seed), saved["info"]
    else:
        log(f"train multi-take {takes} {name} seed {seed}")
        x = h.load("x", (h.FIT, h.VALID))
        y = {t: h.load(t, (h.FIT, h.VALID)) for t in takes}
        trained, info = train(
            x[h.FIT],
            np.stack([y[t][h.FIT] for t in takes]),
            x[h.VALID],
            np.stack([y[t][h.VALID] for t in takes]),
            CONFIGS[name],
            seed,
        )
        torch.save(
            {
                "state": trained.model.state_dict(),
                "x_scale": trained.x_scale,
                "y_scale": trained.y_scale,
                "info": info,
            },
            path,
        )
    x = h.load("x", (h.FIT, h.VALID, h.READ))
    full = {s: trained.output(x[s]) for s in x}
    probe = Readout(Features(trained))
    out = {
        "config": name,
        "seed": seed,
        "trained_on_A_of": takes,
        "train": info,
        "full_model_by_gain": {
            f"{g:g}": {s: h.ratio(h.load(t, (s,))[s], full[s][k]) for s in x}
            for k, (g, t) in enumerate(h.GAIN_TAKES.items())
        },
        "readout_by_gain": gains_of(capacities(probe, takes, (h.FIT, h.VALID, h.READ))),
        "capacity_by_take": capacities(probe, sorted(h.ROWS), (h.READ,)),
        "difference_share_one_knob": [
            {
                **p,
                "knob": h.knob(p),
                "setting": h.ROWS[p["eval"]]["setting"],
                "share_from_x": h.difference_share(probe, p["ref"], p["eval"]),
            }
            for p in h.pairs(("one_knob",), excluded=True)
        ],
        "linearity_check_1_7": h.linearity_check(probe, 1, 7),
    }
    log("multitake", out["readout_by_gain"])
    result = json.loads(OUT.read_text())
    result["multi_take"] = out
    save(result)


def transfer() -> None:
    """Transfer matrix across the gain axis, no training: features frozen from the
    chosen take-20 model and from each per-take model (same config and seed), readout
    on the 5 gain takes (V, B); gain-pair difference shares with each feature set,
    flagged clean when neither take of the pair is the features' training take; and
    RMS(d)/RMS(y_ref) on B for every one-knob pair (no fit)."""
    result = json.loads(OUT.read_text())
    chosen = result["runs"][result["selection"]["chosen"]]
    gain_takes = list(h.GAIN_TAKES.values())
    matrix, shares = {}, []
    for source in (REF_TAKE, *UPPER_TAKES):
        trained, _ = trained_on_take(source, chosen["config"], chosen["seed"])
        probe = Readout(Features(trained))
        reads = capacities(probe, gain_takes, (h.VALID, h.READ))
        matrix[str(source)] = {
            "features_gain": h.ROWS[source]["G"],
            "readout_by_gain": gains_of(reads),
        }
        for p in h.pairs(("one_knob",)):
            if h.knob(p) != "G":
                continue
            twins = {1, 20, 39} if source == REF_TAKE else {source}
            shares.append(
                {
                    **p,
                    "features_from_take": source,
                    "clean": not ({p["ref"], p["eval"]} & twins),
                    "share_from_x": h.difference_share(probe, p["ref"], p["eval"]),
                }
            )
        log("transfer", source, {g: v[h.READ] for g, v in gains_of(reads).items()})
    magnitude = {}
    for p in h.pairs(("one_knob",), excluded=True):
        y_ref = h.load(p["ref"], (h.READ,))[h.READ]
        d = h.difference(p["ref"], p["eval"], h.READ)
        magnitude[f"{p['ref']}->{p['eval']}"] = h.nr.rms(d[h.SKIP :]) / h.nr.rms(
            y_ref[h.SKIP :]
        )
    result = json.loads(OUT.read_text())
    result["transfer"] = {
        "note": "features from take T read on take T are nonlinear fits, not frozen "
        "readings; take 20's frozen reading is the one from another take's features",
        "matrix": matrix,
        "gain_pair_shares": shares,
        "difference_rms_over_ref_rms_B": magnitude,
    }
    save(result)


LABELS = {
    "difference_shares": "chosen_B shares use features trained on take 20 and every "
    "one-knob pair has ref in (1, 20, 39): r_ref is a nonlinear fit, not a frozen "
    "reading; transfer.gain_pair_shares with clean=True are fully frozen readings",
    "take_20_1_39": "features trained on A of take 20 (1 and 39 are its repeats, "
    "1.4e-4 apart): a nonlinear fit to that take, not a frozen-probe reading",
    "upper_per_take": "a model trained on A of the take itself: nonlinear in y, not "
    "a probe; its readout on its own features is not linear in y either",
    "multi_take": "features trained on A of the 5 gain takes: those 5 and the "
    "repeats 1 and 39 of take 20 are not frozen-probe readings; the 35 other takes "
    "are",
    "by_gain_row_5": "take 20 is the features' training take: its frozen_readout_* "
    "values are a nonlinear fit (the transfer matrix gives its frozen readings)",
    "excluded": "gate-4 takes (9, 12, 13, 18, 23, 24, 38, 41) measured, flagged",
    "jvm": "A/B source independence not established, no repeats",
}


REVIEW = {
    "by": "Codex (gpt-6-astra, xhigh), read-only, no computation, 2026-10-11",
    "fixed": [
        "multi-take label: takes 1 and 39 (repeats of take 20) are not frozen "
        "readings either",
        "take-20 row of summary.by_gain flagged as a nonlinear fit",
        "scratch numbers: provenance stated (outside the repository, not re-run)",
        "docstring: the full model's B ratio of every run is computed and logged "
        "before selection (the selectors read V only)",
    ],
    "declined": [
        "memo/Gram cache keyed by (length, sum, sum of squares): only the source "
        "sections A, V, B (and 2 JVM segments) reach it, their keys differ",
        "stale chosen/upper blocks if the candidate grid changes: the grid did not "
        "change after selection; re-running `run` gave the same selection",
    ],
    "disclosed": "one human decision saw B: after wh3_s0 (V-best, full-model B "
    "logged), wh3 was kept at 3 seeds instead of being cut to 1; the V-selected "
    "run was already trained and did not change",
}


def summary() -> None:
    """Derived numbers from the JSON only (no data read): seed spread on B, A/B gap
    of the chosen probe, and one comparison row per gain."""
    result = json.loads(OUT.read_text())
    runs, base = result["runs"], result["baselines"]
    chosen = runs[result["selection"]["chosen"]]
    gains = list(chosen["readout_B"])
    spread = {}
    for name in CONFIGS:
        mine = [r for r in runs.values() if r["config"] == name]
        spread[name] = {
            "seeds": [r["seed"] for r in mine],
            "readout_B_min_max": {
                g: [
                    min(r["readout_B"][g][h.READ] for r in mine),
                    max(r["readout_B"][g][h.READ] for r in mine),
                ]
                for g in gains
            },
            "full_model_take20_B_min_max": [
                min(r["full_model_take20"][h.READ] for r in mine),
                max(r["full_model_take20"][h.READ] for r in mine),
            ],
            "mean_V_scores": [r["selection_score_mean_V"] for r in mine],
        }
    table = {}
    for g in gains:
        take = str(h.GAIN_TAKES[float(g)])
        row = {
            "take": int(take),
            "hammerstein_5_B": base["hammerstein_5"]["capacity_by_gain"][g],
            "volterra_laguerre_3_B": base["volterra_laguerre_3"]["capacity_by_gain"][g],
            "scratch_wiener_bla_B": base["scratch"]["wiener_bla_256"][g],
            "scratch_wiener_hp300_B": base["scratch"]["wiener_hp300_256"][g],
            "frozen_readout_A": chosen["readout_AV"][g][h.FIT],
            "frozen_readout_V": chosen["readout_AV"][g][h.VALID],
            "frozen_readout_B": chosen["readout_B"][g][h.READ],
        }
        row["frozen_A_to_B_gap"] = row["frozen_readout_B"] - row["frozen_readout_A"]
        upper = result.get("upper_per_take", {}).get(take)
        if upper:
            row["upper_full_model_B"] = upper["full_model"][h.READ]
            row["upper_own_readout_B"] = upper["readout_own_features"][h.READ]
        if int(take) == REF_TAKE:
            row["upper_full_model_B"] = chosen["full_model_take20"][h.READ]
            row["note"] = LABELS["by_gain_row_5"]
        if "drive_bank" in result:
            row["drive_bank_B"] = result["drive_bank"]["capacity_by_gain"][g][h.READ]
        if "multi_take" in result:
            row["multi_take_B"] = result["multi_take"]["readout_by_gain"][g][h.READ]
        if "transfer" in result:
            row["transfer_B_by_features_take"] = {
                t: m["readout_by_gain"][g][h.READ]
                for t, m in result["transfer"]["matrix"].items()
            }
        table[g] = row
    by_take = result["chosen_B"]["capacity_by_take"]
    usable = {int(t): v[h.READ] for t, v in by_take.items() if not v["excluded"]}
    frozen_only = [v for t, v in usable.items() if t not in (1, 20, 39)]
    if "multi_take" in result:
        seen = {*h.GAIN_TAKES.values(), 1, 39}
        multi = [
            v[h.READ]
            for t, v in result["multi_take"]["capacity_by_take"].items()
            if not v["excluded"] and int(t) not in seen
        ]
    result["summary"] = {
        "labels": LABELS,
        "review": REVIEW,
        "chosen": result["selection"]["chosen"],
        "seed_spread": spread,
        "by_gain": table,
        "usable_takes_B": {
            "all_34": [min(usable.values()), float(np.median(list(usable.values()))),
                       max(usable.values())],
            "without_1_20_39": [min(frozen_only), float(np.median(frozen_only)),
                                max(frozen_only)],
            "below_0_3_without_1_20_39": sum(v < 0.3 for v in frozen_only),
            "count_without_1_20_39": len(frozen_only),
        },
        "multi_take_usable_frozen_only_B": {
            "takes": "usable, without the 5 gain takes and 1, 39",
            "min_median_max": [min(multi), float(np.median(multi)), max(multi)],
            "below_0_3": sum(v < 0.3 for v in multi),
            "count": len(multi),
        }
        if "multi_take" in result
        else None,
    }  # fmt: skip
    save(result)
    print(json.dumps(result["summary"], indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = {
        "run": run,
        "jvm": jvm,
        "drives": drive_bank,
        "multitake": multitake,
        "transfer": transfer,
        "summary": summary,
    }
    parser.add_argument("command", choices=tuple(commands))
    commands[parser.parse_args().command]()


if __name__ == "__main__":
    main()
