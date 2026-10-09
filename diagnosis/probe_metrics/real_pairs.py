#!/usr/bin/env python3
"""Deviation scores on real recorded pairs, no neural model (NOTE.md, phase 6).

VS100R FX send (2026-09-26), analysed at 48 kHz (both tracks decimated by 4):
- repeats of the centre setting (takes 1, 20, 39): the real floor of each score;
- negative controls: one of bass / middle / treble moved from the centre. The FX send
  is taken before the tone stack, so these should look like repeats;
- positive changes: gain or contour moved from the centre.
Each knob take is paired with the centre take nearest in time. Common input: the
reference take's loopback. The evaluated take's loopback and FX send are shifted
TOGETHER by the integer lag that aligns its loopback to the reference loopback (a
common acquisition correction; the input-output relation is untouched); the
remaining input mismatch is reported, and a section whose lag changes inside it (a
slip) is excluded. A = di_train, B = di_test (independent sources) and pink noise.
Harmonic signature on the three sweeps (not synchronised: measured mixing matrix).

Marshall JVM410H OD1 (ToneTwist, 44.1 kHz), one-knob ladders around the centre,
PreampOut and SpeakerOut; common dry input. A = 0-150 s, B = 215-300 s of the same
input file: disjoint segments, source independence NOT established.
Writes real_pairs.csv. CPU only.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, resample_poly

from fssr_nam.metrics.spectral import multi_resolution_stft_error
from fssr_nam.metrics.time import error_to_signal_ratio
from fssr_nam.product import vs100r
from fssr_nam.product.fidelity import (
    HarmonicProtocol,
    Protocol,
    analyze_pair,
    analyze_sweep_pair,
)
from fssr_nam.product.hammerstein import Sweep

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibrate import frozen_probes  # noqa: E402

FS = 48_000.0
SWEEPS = ("sweep_-20dBFS", "sweep_-10dBFS", "sweep_-3dBFS")
READ = ("di_test", "pink_noise_-30dBFS_rms")
SKIP = round(0.05 * FS)
JVM = vs100r.ROOT / "datasets/raw/external/tone_twist_param/extracted"
JVM = JVM / "Marshall-JVM410H-ChOD1-trimmed"
CENTRE = {"G": 5.0, "C": 5.0, "B": 5.0, "M": 5.0, "T": 5.0}


def vs100r_pairs() -> list[dict]:
    takes = [
        t
        for t in vs100r.sheet()
        if t["channel"] == "OD2" and t["power_dimension"] == "off"
    ]
    centre = [t for t in takes if t["knobs"] == CENTRE]
    pairs = [
        {"kind": "repeat", "ref": a, "eval": b}
        for i, a in enumerate(centre)
        for b in centre[i + 1 :]
    ]
    for take in takes:
        moved = [k for k in CENTRE if take["knobs"][k] != CENTRE[k]]
        if len(moved) != 1:
            continue
        kind = "negative_control" if moved[0] in "BMT" else "positive"
        nearest = min(centre, key=lambda c: abs(c["order"] - take["order"]))
        pairs.append({"kind": kind, "knob": moved[0], "ref": nearest, "eval": take})
    return pairs


def lag(reference: np.ndarray, other: np.ndarray, span: int = 64) -> int:
    """Integer lag (samples) that best aligns `other` to `reference`."""
    c = correlate(other, reference, mode="full", method="fft")
    mid = len(reference) - 1
    return int(np.argmax(np.abs(c[mid - span : mid + span + 1])) - span)


def shifted(signal: np.ndarray, k: int) -> np.ndarray:
    """signal advanced by k samples (zero filled), same length."""
    if k >= 0:
        return np.r_[signal[k:], np.zeros(k)]
    return np.r_[np.zeros(-k), signal[:k]]


def common_section(ref_take, eval_take, name: str) -> dict:
    loop_r, fx_r = (resample_poly(s, 1, 4) for s in vs100r.audio(ref_take, name))
    loop_e, fx_e = (resample_poly(s, 1, 4) for s in vs100r.audio(eval_take, name))
    quarter = len(loop_r) // 4
    lags = [
        lag(loop_r[:quarter], loop_e[:quarter]),
        lag(loop_r[-quarter:], loop_e[-quarter:]),
    ]
    loop_e, fx_e = shifted(loop_e, lags[0]), shifted(fx_e, lags[0])
    keep = slice(SKIP, len(loop_r) - abs(lags[0]) - SKIP)
    mismatch = np.sqrt(
        np.mean((loop_e - loop_r)[keep] ** 2) / np.mean(loop_r[keep] ** 2)
    )
    return {
        "x": loop_r,
        "y_ref": fx_r,
        "y_eval": fx_e,
        "lag": lags[0],
        "slip_inside": lags[0] != lags[1],
        "input_mismatch_db": float(20 * np.log10(max(mismatch, 1e-12))),
    }


def probe_rows(sections: dict, meta: dict) -> list[dict]:
    x = {k: v["x"] for k, v in sections.items()}
    y_ref = {k: v["y_ref"] for k, v in sections.items()}
    y_eval = {k: v["y_eval"] for k, v in sections.items()}
    rows = []
    for metric, probe in frozen_probes().items():
        protocol = Protocol(probe=probe, estimation=("di_train",), reading=READ)
        out = analyze_pair(x, y_ref, y_eval, FS, protocol)
        for segment, reading in out["readings"].items():
            rows.append(
                {
                    **meta,
                    "metric": metric,
                    "segment": segment,
                    "score": reading["d_norm"],
                    "d_audio_norm": reading["d_audio_norm"],
                    "unrepresented": reading["unrepresented"]
                    / max(reading["rms_ref"], protocol.floor),
                    "r_ref": reading["r_ref"] / max(reading["rms_ref"], protocol.floor),
                    "bound_holds": reading["bound"]["holds"],
                }
            )
    return rows


def baseline_rows(sections: dict, meta: dict) -> list[dict]:
    rows = []
    for segment in READ:
        ref, ev = (sections[segment][k][SKIP:] for k in ("y_ref", "y_eval"))
        rows += [
            {
                **meta,
                "metric": "esr",
                "segment": segment,
                "score": error_to_signal_ratio(ev, ref),
            },
            {
                **meta,
                "metric": "mrstft",
                "segment": segment,
                "score": multi_resolution_stft_error(ev, ref),
            },
        ]
    return rows


def sweep_rows(sections: dict, meta: dict) -> list[dict]:
    rows = []
    sweep = Sweep(f0=20.0, f1=20_000.0, seconds=5.0)
    for name in SWEEPS:
        s = sections[name]
        out = analyze_sweep_pair(
            s["x"], s["y_ref"], s["y_eval"], FS, sweep, HarmonicProtocol()
        )
        rows.append(
            {
                **meta,
                "metric": "harmonic_total",
                "segment": name,
                "score": out["total_complex_rel"],
            }
        )
        for k in range(1, 6):
            entry = out[f"order_{k}"]
            rows.append(
                {
                    **meta,
                    "metric": f"harmonic_order_{k}",
                    "segment": name,
                    "score": entry["complex_rel"],
                    "bins_only_eval": entry["bins_only_eval"],
                }
            )
    return rows


def vs100r_rows() -> list[dict]:
    rows = []
    for pair in vs100r_pairs():
        start = time.time()
        names = ("di_train", *READ, *SWEEPS)
        sections = {n: common_section(pair["ref"], pair["eval"], n) for n in names}
        meta = {
            "device": "vs100r",
            "kind": pair["kind"],
            "knob": pair.get("knob", ""),
            "ref": pair["ref"]["setting"] + f"#{pair['ref']['order']}",
            "eval": pair["eval"]["setting"] + f"#{pair['eval']['order']}",
            "low_signal": pair["eval"]["low_signal"] or pair["ref"]["low_signal"],
            "lags": json.dumps({n: s["lag"] for n, s in sections.items()}),
            "input_mismatch_db": max(s["input_mismatch_db"] for s in sections.values()),
            "slip_inside": any(s["slip_inside"] for s in sections.values()),
        }
        if meta["slip_inside"]:
            rows.append({**meta, "metric": "excluded", "segment": "slip"})
            continue
        rows += probe_rows(sections, meta) + baseline_rows(sections, meta)
        rows += sweep_rows(sections, meta)
        print(
            f"{meta['kind']} {meta['ref']}->{meta['eval']} {time.time() - start:.0f}",
            flush=True,
        )
    return rows


def jvm_audio(output: str, setting: str):
    rate = sf.info(JVM / "DRY/trainval/trainval.input.wav").samplerate
    x = sf.read(JVM / "DRY/trainval/trainval.input.wav", dtype="float64")[0]
    y = sf.read(
        JVM / f"{output}/trainval/{setting}/{setting}.trainval.target.wav",
        dtype="float64",
    )[0]
    parts = {"A": slice(0, 150 * rate), "B": slice(215 * rate, 300 * rate)}
    return (
        rate,
        {k: x[s] for k, s in parts.items()},
        {k: y[s] for k, s in parts.items()},
    )


def jvm_rows() -> list[dict]:
    centre = "B050_M050_T050_G050"
    settings = sorted(p.name for p in (JVM / "PreampOut/trainval").iterdir())
    rows = []
    for output in ("PreampOut", "SpeakerOut"):
        rate, x, y_ref = jvm_audio(output, centre)
        for setting in settings:
            moved = [
                a != b
                for a, b in zip(setting.split("_"), centre.split("_"), strict=True)
            ]
            if setting == centre or sum(moved) != 1:
                continue
            _, _, y_eval = jvm_audio(output, setting)
            meta = {
                "device": "jvm410h_" + output,
                "kind": "ladder",
                "knob": "BMTG"[moved.index(True)],
                "ref": centre,
                "eval": setting,
            }
            for metric, probe in frozen_probes().items():
                protocol = Protocol(probe=probe, estimation=("A",), reading=("B",))
                reading = analyze_pair(x, y_ref, y_eval, rate, protocol)["readings"][
                    "B"
                ]
                rows.append(
                    {
                        **meta,
                        "metric": metric,
                        "segment": "B",
                        "score": reading["d_norm"],
                        "d_audio_norm": reading["d_audio_norm"],
                        "unrepresented": reading["unrepresented"]
                        / max(reading["rms_ref"], protocol.floor),
                        "bound_holds": reading["bound"]["holds"],
                    }
                )
            skip = round(0.05 * rate)
            rows.append(
                {
                    **meta,
                    "metric": "esr",
                    "segment": "B",
                    "score": error_to_signal_ratio(
                        y_eval["B"][skip:], y_ref["B"][skip:]
                    ),
                }
            )
            rows.append(
                {
                    **meta,
                    "metric": "mrstft",
                    "segment": "B",
                    "score": multi_resolution_stft_error(
                        y_eval["B"][skip:], y_ref["B"][skip:]
                    ),
                }
            )
            print(f"jvm {output} {setting}", flush=True)
    return rows


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    rows = []
    if which in ("all", "vs100r"):
        rows += vs100r_rows()
    if which in ("all", "jvm"):
        rows += jvm_rows()
    fields = sorted({k for row in rows for k in row})
    with (HERE / f"real_pairs_{which}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
