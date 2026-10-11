#!/usr/bin/env python3
"""Numbers of dossier P3 (sample-rate transfer of SSM-WaveNet), into numbers.json.

Reads existing result files only (demo/sr_transfer, demo/nablafx_bench,
demo/chain_ultrasonic); trains nothing and runs no model. Closed forms evaluated here:
the T4 gauge (1 - 48000/fs)/2 and the ESR ratio of a pure delay (ratio of squared
gauges). Thresholds of the pre-registrations are copied as constants, with the file
that fixed them as source.

    uv run python paper/p3_sample_rate/numbers.py

Threshold regexes quote the French documents, typographic minus and times signs
included (RUF001 ignored for this file).
"""

# ruff: noqa: RUF001
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

# This directory holds numbers.py, which would shadow the stdlib module `numbers`
# (imported by numpy, statistics): replace it by paper/ on the import path, which
# also makes numbers_lib importable.
HERE = Path(__file__).resolve().parent
sys.path[0] = str(HERE.parent)
from numbers_lib import ROOT, Numbers, load, sci  # noqa: E402

SR = "demo/sr_transfer"
SEEDS = (925, 736, 688)
RATES = {"88": 88200, "96": 96000, "192": 192000}
N = Numbers()


def path(seed: int, suffix: str, model: str = "ssm_wavenet") -> str:
    return f"{SR}/sr_transfer_paired_{model}_s{seed}_last{suffix}.json"


def esr(file: str, arm: str) -> float:
    return load(file)["arms"][arm]["esr"]


def srcs(*suffixes: str, model: str = "ssm_wavenet") -> str:
    return "; ".join(path(s, x, model) for x in suffixes for s in SEEDS)


def span(key: str, values: list[float], source: str, digits: int = 3, sc: bool = False):
    """key_min and key_max; sc: scientific text."""
    for name, v in (("min", min(values)), ("max", max(values))):
        text = sci(v, digits) if sc else None
        N.add(f"{key}_{name}", v, source, digits, text)


SUPERSCRIPT = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")


def number(text: str) -> float:
    """French text of a number: 1,15 ; 1/2 ; 10⁻³ ; 5·10⁻⁴."""
    if "/" in text:
        num, den = text.split("/")
        return int(num) / int(den)
    power = re.fullmatch(r"(?:(\d+(?:,\d+)?)·)?10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)", text)
    if power:
        mantissa = float((power.group(1) or "1").replace(",", "."))
        return mantissa * 10 ** int(power.group(2).translate(SUPERSCRIPT))
    return float(text.replace(",", "."))


def parsed(doc: str, pattern: str) -> list[float]:
    """Numbers of a pre-registration, read from the phrase that states them.

    The regex is anchored on the surrounding words and must match exactly once.
    """
    hits = list(re.finditer(pattern, (ROOT / doc).read_text()))
    if len(hits) != 1:
        raise ValueError(f"{doc}: {len(hits)} matches for {pattern!r}")
    return [number(g) for g in hits[0].groups()]


def gauge(fs: int) -> float:
    """T4: delay unit (1 - 48000/fs)/2, in samples at 48 kHz."""
    return (1 - 48000 / fs) / 2


# Down to 44 100 Hz, against the device (predictions.md P1-P3, P7; posthoc controls).
last = {s: path(s, "") for s in SEEDS}
src_last = srcs("")
span(
    "down_around_48k_ratio",
    [esr(last[s], "44k1_around") / esr(last[s], "48k") for s in SEEDS],
    src_last,
)
span(
    "down_delta_around_ratio",
    [esr(last[s], "44k1_delta") / esr(last[s], "44k1_around") for s in SEEDS],
    src_last,
)
span(
    "down_naive_delta_ratio",
    [esr(last[s], "44k1_naive") / esr(last[s], "44k1_delta") for s in SEEDS],
    src_last,
    1,
)
p3 = parsed(f"{SR}/predictions.md", r"donc naive/delta entre (\d+,\d+) et (\d+,\d+)\.")
N.add("down_p3_prevu_min", p3[0], f"{SR}/predictions.md", 2)
span(
    "down_lead_delta_ratio",
    [esr(last[s], "44k1_delta_lead") / esr(last[s], "44k1_delta") for s in SEEDS],
    src_last,
    2,
)
span(
    "up_lead_delta_ratio",
    [
        esr(last[s], f"{r}k_delta_lead_vs_around")
        / esr(last[s], f"{r}k_delta_vs_around")
        for s in SEEDS
        for r in ("96", "192")
    ],
    src_last,
    2,
)
for k in ("2", "4"):
    span(
        f"down_up{k}_around_ratio",
        [
            esr(last[s], f"44k1_up{k}_delta") / esr(last[s], "44k1_around")
            for s in SEEDS
        ],
        src_last,
        2,
    )
N.add("down_p3_prevu_max", p3[1], f"{SR}/predictions.md", 1)
post = {s: path(s, "_posthoc") for s in SEEDS}
span(
    "down_interior_delta_around_ratio",
    [
        esr(post[s], "44k1_delta_interior")
        / esr(post[s], "44k1_around_wholefile_interior")
        for s in SEEDS
    ],
    srcs("_posthoc"),
)
span(
    "down_interior_around_delta_ratio",
    [
        esr(post[s], "44k1_around_wholefile_interior")
        / esr(post[s], "44k1_delta_interior")
        for s in SEEDS
    ],
    srcs("_posthoc"),
    2,
)

# Up, uncorrected Delta rescaling: output brought back to 48 kHz, against the model
# at 48 kHz in the common band.
hold = {s: path(s, "_holdtest") for s in SEEDS}
src_hold = srcs("_holdtest")
for s in SEEDS:
    for r in RATES:
        N.add(
            f"up_delta_esr_s{s}_{r}",
            esr(hold[s], f"{r}k_delta_band48_vs_48k_interior"),
            hold[s],
            4,
        )
up = {
    (s, r): esr(hold[s], f"{r}k_delta_band48_vs_48k_interior")
    for s in SEEDS
    for r in RATES
}
span("up_delta_esr", list(up.values()), src_hold)
span("up_delta_esr_88_96", [up[s, r] for s in SEEDS for r in ("88", "96")], src_hold)
span("up_delta_ratio_192_96", [up[s, "192"] / up[s, "96"] for s in SEEDS], src_hold, 2)
span("up_delta_ratio_88_96", [up[s, "88"] / up[s, "96"] for s in SEEDS], src_hold, 2)
# T1 closed form: phi(z) = 1/2 + z/12 - z^3/720 + O(z^5), so a slow mode (z -> 0) is
# advanced by phi(0) = 1/2 sample; series coefficients evaluated at z = 0.
phi0 = parsed(
    "theory/T1_hold_moments.md",
    r"φ\(z\) = −1/z − 1/\(e\^\{−z\} − 1\) = (\d+/\d+) \+ z/12",
)
N.add("t1_phi0", phi0[0], "theory/T1_hold_moments.md", 1)
for r, fs in RATES.items():
    N.add(f"t4_jauge_{r}", gauge(fs), "THEORY.md", 3)
N.add("up_retard_pur_ratio_192_96", (gauge(192000) / gauge(96000)) ** 2, "THEORY.md", 2)
N.add("up_retard_pur_ratio_88_96", (gauge(88200) / gauge(96000)) ** 2, "THEORY.md", 2)
span(
    "up_p5_esr96",
    [esr(last[s], "96k_delta_vs_around") for s in SEEDS],
    src_last,
    4,
)
p5 = parsed(
    f"{SR}/predictions.md", r"Prédiction : ESR de (\d+,\d+) à\s+(\d+,\d+) à 96 kHz\."
)
N.add("up_p5_prevu_min", p5[0], f"{SR}/predictions.md", 3)
N.add("up_p5_prevu_max", p5[1], f"{SR}/predictions.md", 2)
inband = max(
    abs(
        esr(post[s], f"{r}k_delta_vs_around_interior")
        / esr(post[s], f"{r}k_delta_band48_vs_48k_interior")
        - 1
    )
    for s in SEEDS
    for r in RATES
)
N.add("up_inband_ecart_rel_max", inband, srcs("_posthoc"), 1, sci(inband, 1))
span(
    "up_naive_esr",
    [esr(last[s], f"{r}k_naive_vs_around") for s in SEEDS for r in ("96", "192")],
    src_last,
    2,
)
span(
    "up_naive_delta_ratio",
    [
        esr(last[s], f"{r}k_naive_vs_around") / esr(last[s], f"{r}k_delta_vs_around")
        for s in SEEDS
        for r in ("96", "192")
    ],
    src_last,
    1,
)

# One-coefficient hold correction (predictions_hold.md H1-H4).
hold_esr = {
    (s, r): esr(hold[s], f"{r}k_delta_hold_band48_vs_48k_interior")
    for s in SEEDS
    for r in RATES
}
for (s, r), v in hold_esr.items():
    N.add(f"hold_esr_s{s}_{r}", v, hold[s], 4)
span(
    "hold_reduction",
    [up[s, r] / hold_esr[s, r] for s in SEEDS for r in ("96", "192")],
    src_hold,
    2,
)
span("hold_esr_88_96", [hold_esr[s, r] for s in SEEDS for r in ("88", "96")], src_hold)
span(
    "hold_m20db_ratio",
    [
        esr(hold[s], f"{r}k_delta_m20db_band48_vs_48k_interior") / up[s, r]
        for s in SEEDS
        for r in ("96", "192")
    ],
    src_hold,
    2,
)
span(
    "hold_ratio_192_96",
    [hold_esr[s, "192"] / hold_esr[s, "96"] for s in SEEDS],
    src_hold,
    2,
)
span(
    "hold_44k1_hausse_pct",
    [
        100
        * (
            esr(hold[s], "44k1_delta_hold_interior")
            / esr(hold[s], "44k1_delta_interior")
            - 1
        )
        for s in SEEDS
    ],
    src_hold,
    1,
)

# Two-coefficient correction (predictions_hold.md H6-H8).
two = {s: path(s, "_twotap") for s in SEEDS}
src_two = srcs("_twotap")
two_esr = {
    (s, r): esr(two[s], f"{r}k_delta_2tap_band48_vs_48k_interior")
    for s in SEEDS
    for r in RATES
}
for (s, r), v in two_esr.items():
    N.add(f"twotap_esr_s{s}_{r}", v, two[s], 2, sci(v, 2))
span("twotap_esr", list(two_esr.values()), src_two, 1, sc=True)
span("twotap_db", [10 * math.log10(v) for v in two_esr.values()], src_two, 1)
span(
    "twotap_reduction",
    [up[k] / two_esr[k] for k in two_esr],
    f"{src_hold}; {src_two}",
    0,
)
span(
    "twotap_ratio_192_96",
    [two_esr[s, "192"] / two_esr[s, "96"] for s in SEEDS],
    src_two,
    2,
)
h7 = parsed(
    f"{SR}/predictions_hold.md", r"192/96 de `delta_2tap` ≥ (\d+,\d+) \(τ⁴ prédit"
)
N.add("twotap_h7_seuil", h7[0], f"{SR}/predictions_hold.md", 1)
N.add("twotap_tau4_ratio", (gauge(192000) / gauge(96000)) ** 4, "THEORY.md", 1)
span(
    "twotap_identity_48k",
    [esr(two[s], "48k_two_tap_identity_esr") for s in SEEDS],
    src_two,
    1,
    sc=True,
)
span(
    "twotap_44k1_ratio",
    [
        esr(two[s], "44k1_delta_2tap_interior") / esr(hold[s], "44k1_delta_interior")
        for s in SEEDS
    ],
    f"{src_two}; {src_hold}",
)

# ZOH checkpoint read as FOH = two-coefficient correction (FIX_foh.md, section 6).
asfoh = {s: path(s, "_asfoh") for s in SEEDS}
src_asfoh = srcs("_asfoh")
pairs = [("44k1_delta_interior", "44k1_delta_2tap_interior")] + [
    (f"{r}k_delta_band48_vs_48k_interior", f"{r}k_delta_2tap_band48_vs_48k_interior")
    for r in RATES
]
gaps = [abs(esr(asfoh[s], a) / esr(two[s], b) - 1) for s in SEEDS for a, b in pairs]
N.add("asfoh_bras", len(gaps), src_asfoh, None)
N.add("asfoh_ecart_rel_max", max(gaps), f"{src_asfoh}; {src_two}", 2, sci(max(gaps), 2))
tol = parsed(f"{SR}/FIX_foh.md", r"\*\*à (10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+) près en relatif\*\*")[0]
N.add("asfoh_tolerance", tol, f"{SR}/FIX_foh.md", 0, sci(tol, 0))
span(
    "asfoh_identity_48k",
    [esr(asfoh[s], "48k_asfoh_identity_esr") for s in SEEDS],
    src_asfoh,
    1,
    sc=True,
)
ident = parsed(f"{SR}/FIX_foh.md", r"Identité à 48 kHz : ESR ≤ (10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)\.")[0]
N.add("asfoh_seuil_identite", ident, f"{SR}/FIX_foh.md", 0, sci(ident, 0))

# The -37 dB floor (predictions_floor.md): resampler, precision, level.
floor = {**load(f"{SR}/floor_925.json"), **load(f"{SR}/floor_736_688.json")}
src_floor = f"{SR}/floor_925.json; {SR}/floor_736_688.json"
fl = [floor[str(s)] for s in SEEDS]
span(
    "floor_v1_v0_hausse_pct",
    [100 * (f["V1_hq_resampler"] / f["V0_default"] - 1) for f in fl],
    src_floor,
    1,
)
N.add(
    "floor_v2_v1_ecart_pct_max",
    max(100 * abs(f["V2_hq_float64"] / f["V1_hq_resampler"] - 1) for f in fl),
    src_floor,
    2,
)
span("floor_hf_share_pct", [100 * f["V4"]["hf_error_share"] for f in fl], src_floor, 0)
span(
    "floor_v3_v2_ratio",
    [f["V3_hq_float64_m20db"] / f["V2_hq_float64"] for f in fl],
    src_floor,
    2,
)
levels = load(f"{SR}/floor_levels.json")
src_levels = f"{SR}/floor_levels.json"
for s in SEEDS:
    for db in ("0", "-20", "-40", "-60"):
        v = levels[str(s)][db]
        N.add(f"floor_esr_s{s}_m{db.lstrip('-')}", v, src_levels, 2, sci(v, 2))
span(
    "floor_m60_m40_ratio",
    [levels[str(s)]["-60"] / levels[str(s)]["-40"] for s in SEEDS],
    src_levels,
    2,
)
span(
    "floor_part_constante_pct",
    [100 * levels[str(s)]["-60"] / levels[str(s)]["0"] for s in SEEDS],
    src_levels,
    1,
)

# Network delay at rate transfer, kappa (T4; predictions_kappa.md).
kj = load(f"{SR}/kappa.json")
src_k = f"{SR}/kappa.json"
ckpts = {f"bm_s{s}": kj["bigmuff"][f"paired_ssm_wavenet_s{s}"] for s in SEEDS}
ckpts["vs100r"] = kj["vs100r"]["vs100r_gc_film_joint_s925_r2"]
for name, c in ckpts.items():
    for r in ("96", "192"):
        N.add(f"kappa_{name}_{r}", c[str(RATES[r])]["kappa"], src_k, 3)
span(
    "kappa_ratio_192_96",
    [c["192000"]["kappa"] / c["96000"]["kappa"] for c in ckpts.values()],
    src_k,
)
KP = f"{SR}/predictions_kappa.md"
r1 = parsed(KP, r"κ\(192\)/κ\(96\) dans \[(\d+,\d+) ; (\d+,\d+)\] → forme")
N.add("kappa_r1_seuil_min", r1[0], KP, 1)
N.add("kappa_r1_seuil_max", r1[1], KP, 2)
readings = kj["readings"]
k_vs = readings["kappa_vs100r"]
indirect = float(
    next(k for k in readings if k.startswith("R2_agrees_with_")).rsplit("_", 1)[1]
)
N.add("kappa_vs100r", k_vs, src_k, 3)
N.add("kappa_vs100r_indirect", indirect, src_k, 2)
N.add("kappa_r2_ecart", abs(k_vs - indirect), src_k, 3)
r2 = parsed(KP, r"\|κ_VS100R − \d+,\d+\| ≤ (\d+,\d+) \(médiane")
N.add("kappa_r2_seuil", r2[0], KP, 1)
N.add("kappa_bm_median", readings["kappa_bigmuff_median"], src_k, 3)
N.add("kappa_r3_ecart", abs(readings["kappa_bigmuff_median"] - k_vs), src_k, 3)
r3 = parsed(KP, r"est à ±(\d+) % du κ du VS100R → propriété de l'architecture")[0] / 100
N.add("kappa_r3_pct", round(100 * r3), KP, None)
N.add("kappa_r3_borne", r3 * k_vs, f"{src_k}; {KP}", 3)
bm = [
    c[f]["kappa"]
    for n, c in ckpts.items()
    if n != "vs100r"
    for f in ("96000", "192000")
]
span("kappa_bm", bm, src_k, 2)
span(
    "kappa_bm_band_sd",
    [
        c[f]["band_sd"]
        for n, c in ckpts.items()
        if n != "vs100r"
        for f in ("96000", "192000")
    ],
    src_k,
)
r4 = parsed(KP, r"≤ (\d+,\d+) échantillon → retard pur")
N.add("kappa_r4_seuil", r4[0], KP, 1)
vs = ckpts["vs100r"]
N.add("kappa_vs100r_band_sd_96", vs["96000"]["band_sd"], src_k, 3)
N.add("kappa_vs100r_band_sd_192", vs["192000"]["band_sd"], src_k, 3)
N.add("kappa_vs100r_retard500_96", vs["96000"]["delay_by_band"]["500"], src_k, 3)
span(
    "kappa_vs100r_retard_autres_96",
    [v for b, v in vs["96000"]["delay_by_band"].items() if b != "500"],
    src_k,
)
span(
    "kappa_retard",
    [c[f]["delay_median"] for c in ckpts.values() for f in ("96000", "192000")],
    src_k,
    2,
)
# ZOH kappa at the three bands predicted for FOH (500 to 2000 Hz), against T1-P1.
span(
    "kappa_zoh_bandes",
    [
        ckpts[f"bm_s{s}"][str(fs)]["delay_by_band"][b] / gauge(fs)
        for s in SEEDS
        for fs in (96000, 192000)
        for b in ("500", "1000", "2000")
    ],
    src_k,
    1,
)

# T1-P1: FOH-trained checkpoints, kappa near zero (kappa_foh.json).
kf = load(f"{SR}/kappa_foh.json")
src_kf = f"{SR}/kappa_foh.json"
fc = [kf["checkpoints"][f"paired_ssm_wavenet_foh_s{s}"] for s in SEEDS]
N.add("fohk_primaire", kf["readings"]["kappa_primary"], src_kf, 3)
N.add("fohk_secondaire", kf["readings"]["kappa_pooled_secondary"], src_kf, 3)
N.add("fohk_verdict", kf["readings"]["verdict"], src_kf, None)
span("fohk_6", [c[f]["kappa_low"] for c in fc for f in ("96000", "192000")], src_kf)
span(
    "fohk_18",
    [
        c[f]["kappa_by_band"][b]
        for c in fc
        for f in ("96000", "192000")
        for b in ("500", "1000", "2000")
    ],
    src_kf,
)
span(
    "fohk_aigu",
    [
        c[f]["kappa_by_band"][b]
        for c in fc
        for f in ("96000", "192000")
        for b in ("4000", "8000")
    ],
    src_kf,
)
span(
    "fohk_allerretour",
    [c[f]["roundtrip_esr"] for c in fc for f in ("96000", "192000")],
    src_kf,
    1,
    sc=True,
)
t1p1 = parsed(
    "THEORY.md",
    r"vérifie \|κ\| ≤ (\d+,\d+) ; réfutée si\s+\|κ\| ≥ (\d+,\d+)\. Aucune prédiction",
)
N.add("fohk_seuil_soutenue", t1p1[0], "THEORY.md", 2)
N.add("fohk_seuil_refutee", t1p1[1], "THEORY.md", 1)

# FOH training, non-inferiority at 48 kHz (predictions_foh_training.md; E-0036).
bench = "demo/nablafx_bench"
d = {}
for s in SEEDS:
    zf, ff = (
        f"{bench}/paired_ssm_wavenet_s{s}.json",
        f"{bench}/paired_ssm_wavenet_foh_s{s}.json",
    )
    z, f = (
        load(zf)["test_last"]["metric/test/esr"],
        load(ff)["test_last"]["metric/test/esr"],
    )
    N.add(f"fohtrain_esr_zoh_s{s}", z, zf, 5)
    N.add(f"fohtrain_esr_foh_s{s}", f, ff, 5)
    d[s] = math.log(z / f)
    N.add(f"fohtrain_d_s{s}", d[s], f"{zf}; {ff}", 3)
src_bench = "; ".join(
    f"{bench}/paired_ssm_wavenet_{v}s{s}.json" for v in ("", "foh_") for s in SEEDS
)
N.add("fohtrain_d_moyen", sum(d.values()) / 3, src_bench, 3)
FT = f"{SR}/predictions_foh_training.md"
margin = parsed(FT, r"marge ln (\d+,\d+) : moyenne des trois d ≥ −ln")[0]
N.add("fohtrain_marge", -math.log(margin), FT, 3)
N.add("fohtrain_d_moyen_sans688", (d[925] + d[736]) / 2, src_bench, 3)
v688 = (
    load(f"{bench}/paired_ssm_wavenet_foh_s688.json")["best_val_loss"],
    load(f"{bench}/paired_ssm_wavenet_s688.json")["best_val_loss"],
)
N.add("fohtrain_val_foh_s688", v688[0], f"{bench}/paired_ssm_wavenet_foh_s688.json", 4)
N.add("fohtrain_val_zoh_s688", v688[1], f"{bench}/paired_ssm_wavenet_s688.json", 4)
plain = {
    s: f"{SR}/sr_transfer_paired_ssm_wavenet_foh_s{s}_last_plain.json" for s in SEEDS
}
src_plain = "; ".join(plain.values())
for s in SEEDS:
    for r in RATES:
        v = esr(plain[s], f"{r}k_delta_band48_vs_48k_interior")
        N.add(f"fohtrain_plain_esr_s{s}_{r}", v, plain[s], 2, sci(v, 2))
span(
    "fohtrain_plain_esr96",
    [esr(plain[s], "96k_delta_band48_vs_48k_interior") for s in SEEDS],
    src_plain,
    1,
    sc=True,
)
span(
    "fohtrain_plain_esr",
    [esr(plain[s], f"{r}k_delta_band48_vs_48k_interior") for s in SEEDS for r in RATES],
    src_plain,
    1,
    sc=True,
)
s96 = parsed(FT, r"modèle 48 kHz du même run ≤ (\d+·10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+) dans la bande")[0]
N.add("fohtrain_seuil_96", s96, FT, 0, sci(s96, 0))
for s in SEEDS:
    N.add(
        f"fohtrain_44k1_ratio_s{s}",
        esr(plain[s], "44k1_delta_interior") / esr(asfoh[s], "44k1_delta_interior"),
        f"{plain[s]}; {asfoh[s]}",
        2 if s == 688 else 3,
    )
s44 = parsed(FT, r"ESR ≤ (\d+,\d+) × celle de la jumelle ZOH relue en FOH")[0]
N.add("fohtrain_seuil_44k1", s44, FT, 2)
lag = load(f"{SR}/foh_lag.json")
l688 = lag["paired_ssm_wavenet_foh_s688"]
N.add("fohtrain_lag0_s688", l688["lags"]["0"], f"{SR}/foh_lag.json", None)
N.add("fohtrain_lag_segments", l688["segments"], f"{SR}/foh_lag.json", None)
N.add("fohtrain_lag_esr_lag0_s688", l688["interior_lag0"], f"{SR}/foh_lag.json", 3)
N.add("fohtrain_lag_esr_best_s688", l688["interior_best_lag"], f"{SR}/foh_lag.json", 3)
N.add(
    "fohtrain_lag_esr_zoh_s688",
    lag["paired_ssm_wavenet_s688"]["interior_best_lag"],
    f"{SR}/foh_lag.json",
    3,
)

# S4-TF-L-16 contrast (P9, H5, H9).
s4 = {
    k: {s: path(s, k, "s4_tf_l_16") for s in SEEDS}
    for k in ("", "_holdtest", "_twotap")
}
s4_up = {
    (s, r): esr(s4["_holdtest"][s], f"{r}k_delta_band48_vs_48k_interior")
    for s in SEEDS
    for r in RATES
}
s4_two = {
    (s, r): esr(s4["_twotap"][s], f"{r}k_delta_2tap_band48_vs_48k_interior")
    for s in SEEDS
    for r in RATES
}
src_s4 = srcs("_holdtest", "_twotap", model="s4_tf_l_16")
span("s4_twotap_red_96", [s4_up[s, "96"] / s4_two[s, "96"] for s in SEEDS], src_s4, 2)
span(
    "s4_twotap_red_192", [s4_up[s, "192"] / s4_two[s, "192"] for s in SEEDS], src_s4, 2
)
span("s4_twotap_esr", list(s4_two.values()), src_s4, 3)
PH = f"{SR}/predictions_hold.md"
N.add(
    "s4_h9_seuil", parsed(PH, r"S4-TF-L-16 : réduction < ×(\d+,\d+) à 96 kHz")[0], PH, 1
)
h5 = parsed(PH, r"n'est pas corrigé\) : réduction entre ×(\d+,\d+) et ×(\d+)\.")
N.add("s4_h5_prevu_min", h5[0], PH, 1)
N.add("s4_h5_prevu_max", round(h5[1]), PH, None)
span(
    "s4_hold_red",
    [
        s4_up[s, r] / esr(s4["_holdtest"][s], f"{r}k_delta_hold_band48_vs_48k_interior")
        for s in SEEDS
        for r in ("96", "192")
    ],
    srcs("_holdtest", model="s4_tf_l_16"),
    2,
)
span(
    "s4_replies_48k_pct",
    [
        100
        * load(s4[""][s])["poles"]["48000"]["folded"]
        / load(s4[""][s])["poles"]["48000"]["total"]
        for s in SEEDS
    ],
    srcs("", model="s4_tf_l_16"),
    1,
)
span(
    "s4_ssm_ratio_96",
    [
        esr(s4[""][s], "96k_delta_vs_around") / esr(last[s], "96k_delta_vs_around")
        for s in SEEDS
    ],
    f"{srcs('', model='s4_tf_l_16')}; {src_last}",
    2,
)
span(
    "s4_naive_delta_ratio",
    [esr(s4[""][s], "44k1_naive") / esr(s4[""][s], "44k1_delta") for s in SEEDS],
    srcs("", model="s4_tf_l_16"),
    1,
)

# Ultrasonic content between two stages (demo/chain_ultrasonic, synthetic systems).
chain = load("demo/chain_ultrasonic/results.json")["summary"]
src_chain = "demo/chain_ultrasonic/results.json"
for exc, short in (
    ("procedural_plucks", "plucks"),
    ("multisine", "multisine"),
    ("colored_noise", "bruit"),
    ("two_tone", "deuxsons"),
):
    N.add(f"chain_{short}_mediane_db", chain[exc]["median_db"], src_chain, 1)
    N.add(f"chain_{short}_pire_db", chain[exc]["max_db"], src_chain, 1)
    N.add(f"chain_{short}_plancher_db", chain[exc]["identity_max_db"], src_chain, 1)

# Two tones: nonlinear pairs (stage 2 not identity) against the instrument floor.
floor2 = chain["two_tone"]["identity_max_db"]
pairs2 = [
    v["db"]
    for k, v in load(src_chain)["pairs"]["two_tone"].items()
    if not k.endswith("->identity")
]
N.add("chain_deuxsons_paires", len(pairs2), src_chain, None)
N.add("chain_deuxsons_sous_plancher", sum(x < floor2 for x in pairs2), src_chain, None)
N.add("chain_deuxsons_depassement_db", max(pairs2) - floor2, src_chain, 2)

# Brightness at 96 and 192 kHz (predictions_brightness.md; readings not decided).
br = load(f"{SR}/brightness.json")
src_br = f"{SR}/brightness.json"
zoh = [br[f"zoh/paired_ssm_wavenet_s{s}"] for s in SEEDS]
foh = [br[f"foh/paired_ssm_wavenet_foh_s{s}"] for s in SEEDS]
N.add("bright_lectures", br["readings"]["R3"], src_br, None)
span("bright_b192_zoh", [c["B_192"] for c in zoh], src_br, 2)
span("bright_b192_foh", [c["B_192"] for c in foh], src_br, 2)
span(
    "bright_i_baisse_db",
    [
        c["sines"]["997_test_rms"]["I_48"] - c["sines"]["997_test_rms"]["I_192"]
        for c in zoh + foh
    ],
    src_br,
    1,
)
span("bright_h192_zoh", [c["sines"]["997_test_rms"]["H_192"] for c in zoh], src_br, 2)


def at16k(c: dict, key: str) -> float:
    return c[key]["16000.0"]


span(
    "bright_16k_zoh",
    [at16k(c, "levels_192") - at16k(c, "levels_48") for c in zoh],
    src_br,
    2,
)
span(
    "bright_16k_foh",
    [at16k(c, "levels_192") - at16k(c, "levels_48") for c in foh],
    src_br,
    2,
)
N.add(
    "bright_16k_48_plus_proche",
    sum(
        abs(at16k(c, "levels_48") - at16k(c, "levels_real"))
        < abs(at16k(c, "levels_192") - at16k(c, "levels_real"))
        for c in zoh + foh
    ),
    src_br,
    None,
)
listened = [zoh[0]["listened"][k] for k in ("10_192", "8_192")]
span("bright_ecoute_esr192", [x["esr_vs_48"] for x in listened], src_br)
span(
    "bright_ecoute_esr192_compense",
    [x["esr_vs_48_delay_compensated"] for x in listened],
    src_br,
    4,
)

N.write(HERE / "numbers.json")
print(f"{len(N.data)} keys -> {HERE / 'numbers.json'}")
