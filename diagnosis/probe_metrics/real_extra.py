#!/usr/bin/env python3
"""Existing magnitude, phase and envelope metrics on the real pairs (post hoc, NOTE.md).

Same pairs and sections as real_pairs.py. They compare recordings directly, so on the
VS100R only sections with a common input (input_common) are kept for any reading; the
flag is written with each row. Writes real_extra_<device>.csv. CPU only.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from fssr_nam.metrics.spectral import magnitude_error, phase_error
from fssr_nam.metrics.transient import envelope_error

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from real_pairs import READ, SKIP, common_section, jvm_audio, vs100r_pairs  # noqa: E402

JVM_CENTRE = "B050_M050_T050_G050"


def metrics(ev, ref, rate: int) -> dict:
    return {
        "magnitude": magnitude_error(ev, ref),
        "phase": phase_error(ev, ref),
        "envelope": envelope_error(ev, ref, sample_rate=rate),
    }


def vs100r_rows() -> list[dict]:
    rows = []
    for pair in vs100r_pairs():
        for segment in READ:
            s = common_section(pair["ref"], pair["eval"], segment)
            ref, ev = s["y_ref"][SKIP:], s["y_eval"][SKIP:]
            for name, value in metrics(ev, ref, 48_000).items():
                rows.append(
                    {
                        "device": "vs100r",
                        "kind": pair["kind"],
                        "knob": pair.get("knob", ""),
                        "ref": f"{pair['ref']['setting']}#{pair['ref']['order']}",
                        "eval": f"{pair['eval']['setting']}#{pair['eval']['order']}",
                        "low_signal": pair["eval"]["low_signal"]
                        or pair["ref"]["low_signal"],
                        "segment": segment,
                        "input_common": s["common"],
                        "metric": name,
                        "score": value,
                    }
                )
        print(pair["eval"]["order"], flush=True)
    return rows


def jvm_rows() -> list[dict]:
    from real_pairs import JVM

    settings = sorted(p.name for p in (JVM / "PreampOut/trainval").iterdir())
    rows = []
    for output in ("PreampOut", "SpeakerOut"):
        rate, _, y_ref = jvm_audio(output, JVM_CENTRE)
        for setting in settings:
            parts = zip(setting.split("_"), JVM_CENTRE.split("_"), strict=True)
            moved = [a != b for a, b in parts]
            if setting == JVM_CENTRE or sum(moved) != 1:
                continue
            _, _, y_eval = jvm_audio(output, setting)
            skip = round(0.05 * rate)
            ref, ev = y_ref["B"][skip:], y_eval["B"][skip:]
            for name, value in metrics(ev, ref, rate).items():
                rows.append(
                    {
                        "device": "jvm410h_" + output,
                        "kind": "ladder",
                        "knob": "BMTG"[moved.index(True)],
                        "ref": JVM_CENTRE,
                        "eval": setting,
                        "segment": "B",
                        "metric": name,
                        "score": value,
                    }
                )
    return rows


def main() -> None:
    which = sys.argv[1]
    rows = vs100r_rows() if which == "vs100r" else jvm_rows()
    fields = sorted({k for r in rows for k in r})
    with (HERE / f"real_extra_{which}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
