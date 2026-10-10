#!/usr/bin/env python3
"""44.1 kHz variant of the capture v2 source, for the KIPR bench only, product lane.

Separate from reamp_v2_192k.wav and from the Nolly session, which stay unchanged.
Requested by the KIPR worker for its two existing engines (KIPR Native V1 and the Player
reference, 44.1 kHz only, offline renders). Writes:

- datasets/raw/internal/capture_v2/reamp_v2_44k1.wav, 44.1 kHz 24-bit mono (not
  committed: it carries ToneTwist DI material, CC BY-NC);
- demo/capture/v2_44k1/layout.json and PROTOCOL.md.

Same sections, in the same order, as the 192 kHz file. The DI sections are the 192 kHz
ones resampled section by section (every excerpt starts and ends with a 10 ms fade);
markers, silence, pink noise and the old sweeps come from product_valvestate_session and
the measurement signals from product_capture_v2.new_sections, generated at 44.1 kHz.
"""

from __future__ import annotations

import json

import numpy as np
import product_capture_v2 as v2
import product_valvestate_session as vs
import soundfile as sf
from scipy.signal import resample_poly

FS = 44_100
UP, DOWN = 147, 640  # 192 kHz -> 44.1 kHz
WINDOW = ("kaiser", 8.6)  # as configs/data/amp_quality_teacher_v1.yaml
AUDIO = v2.ROOT / "datasets/raw/internal/capture_v2/reamp_v2_44k1.wav"
OUT = v2.ROOT / "demo/capture/v2_44k1"
DI = ("di_train", "di_validation", "di_test")
SEGMENT_S, SEGMENTS = 3, 50


def di_section(name: str, sections: dict) -> np.ndarray:
    s = sections[name]
    x = sf.read(
        v2.AUDIO, dtype="float64", start=s["start"], stop=s["start"] + s["samples"]
    )[0]
    return resample_poly(x, UP, DOWN, window=WINDOW)


def protocol(layout: dict) -> str:
    return "\n".join(
        [
            "# Variante 44,1 kHz du fichier de capture v2 : sous-protocole KIPR",
            "",
            "Voie produit. Généré par `scripts/product_capture_v2_44k1.py` ; ne "
            "pas éditer à la main. Pour le banc KIPR seulement (KIPR Native V1 et "
            "référence Player, 44,1 kHz, rendus hors ligne). Séparée du fichier "
            "192 kHz (`demo/capture/v2/`) et de la séance Nolly, qui ne changent pas.",
            "",
            f"- **Fichier** : `{layout['file']}`, {layout['seconds']} s, 44,1 kHz, "
            f"24 bits, mono, sha256 `{layout['sha256']}`.",
            "- **Sections** : les mêmes que le fichier 192 kHz, dans le même ordre "
            "(`layout.json`). DI rééchantillonnée depuis le 192 kHz (147/640, "
            "Kaiser β = 8,6) ; marqueurs, silence, bruit rose, sinus glissants, "
            "multisinus, deux tons et paliers synthétisés à 44,1 kHz.",
            "- **Identité au bit perdue** : ce fichier n'est identique au bit ni au "
            "fichier VS100R ni au fichier v2 à 192 kHz. Research-memory D-0012 ne "
            "vaut que pour le 192 kHz.",
            "- **Emboîtement conservé** : `di_train` + `di_validation` font 150 s, "
            "soit 50 segments de 3 s (132 300 échantillons). Les extraits de 30 s "
            "(positions 0, 5, …, 45) restent contenus.",
            "- **Indépendance A/B** : `di_train` et `di_validation` viennent des mêmes "
            "cinq enregistrements (fenêtres différentes) ; `di_test` vient de deux "
            "autres (`idmt-bass`, `idmt-gtr4-ib`). Pour deux sources disjointes : "
            "A = `di_train`, B = `di_test`.",
            "- **Borne harmonique** : l'ordre k ne se lit que pour f < 22 050 / k Hz "
            "(ordre 3 : 7 350 Hz ; ordre 9 : 2 450 Hz). Plafonner la bande de 100 à "
            "8 000 Hz par ordre.",
            "- **Retard** : estimer et rapporter un retard fractionnaire résiduel "
            "entre sorties, sans recaler ni corriger les sorties ou les scores. Les "
            "0,8 à 1,05 échantillon observés concernent des modèles VS100R à 48 kHz : "
            "ce n'est pas une plage attendue pour les moteurs KIPR.",
            "",
        ]
    )


def main() -> None:
    lay192 = json.loads(v2.LAYOUT.read_text())
    assert v2.sha256(v2.AUDIO) == lay192["sha256"], "reamp_v2_192k.wav changed"
    by_name = {s["section"]: s for s in lay192["sections"]}
    rng = np.random.default_rng(vs.SEED)
    gap = np.zeros(FS // 2)
    head = [
        ("pad_start", np.zeros(vs.PAD_START_S * FS), {}),
        ("markers_start", vs.markers(FS), {}),
        ("silence", np.zeros(5 * FS), {}),
        (None, gap, {}),
        ("pink_noise_-30dBFS_rms", vs.pink_noise(5, -30, rng, FS), {}),
        (None, gap, {}),
    ]
    for level in (-20, -10, -3):
        head += [(f"sweep_{level}dBFS", vs.sweep(5, level, FS), {}), (None, gap, {})]
    head += [(n, di_section(n, by_name), {"resampled_from": "192 kHz v2"}) for n in DI]
    head.append((None, gap, {}))
    for name, x, params in v2.new_sections(FS):
        assert np.max(np.abs(x)) < 10 ** (-1 / 20), f"{name} peaks above -1 dBFS"
        head += [(name, x, params), (None, gap, {})]
    head += [("markers_end", vs.markers(FS), {}), ("pad_end", np.zeros(6 * FS), {})]
    layout, position = [], 0
    for name, x, params in head:
        if name is not None:
            layout.append(
                {"section": name, "start": position, "samples": len(x)} | params
            )
        position += len(x)
    signal = np.concatenate([x for _, x, _ in head])
    assert np.max(np.abs(signal)) < 1.0
    training = [s for s in layout if s["section"] in ("di_train", "di_validation")]
    assert sum(s["samples"] for s in training) == SEGMENTS * SEGMENT_S * FS
    AUDIO.parent.mkdir(parents=True, exist_ok=True)
    sf.write(AUDIO, v2.to_int(signal), FS, subtype="PCM_24")
    result = {
        "sample_rate": FS,
        "sections": layout,
        "file": str(AUDIO.relative_to(v2.ROOT)),
        "sha256": v2.sha256(AUDIO),
        "seconds": round(len(signal) / FS, 3),
        "training_part_sha256": v2.training_hash(AUDIO, layout),
        "training_part_hash": lay192["training_part_hash"],
        "derived_from": {
            "file": lay192["file"],
            "sha256": lay192["sha256"],
            "di_sections": f"resample_poly(x, {UP}, {DOWN}, window={WINDOW}), "
            "section by section",
        },
        "di_source": lay192["di_source"],
        "notes": [
            "KIPR bench only; separate from the 192 kHz file and the Nolly session.",
            "Not bit-identical to the VS100R file nor to the 192 kHz v2 file: "
            "D-0012 holds for the 192 kHz file only.",
            "di_train + di_validation = 150 s = 50 x 3 s: the 30 s extracts "
            "(positions 0, 5, ..., 45) stay contained.",
            "A/B on disjoint sources: A = di_train, B = di_test (di_validation shares "
            "the di_train recordings).",
            "Harmonic order k readable only for f < 22050 / k Hz.",
            *lay192["notes"][1:],
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "layout.json").write_text(json.dumps(result, indent=1) + "\n")
    (OUT / "PROTOCOL.md").write_text(protocol(result))
    print(f"{result['file']}: {result['seconds']} s, sha256 {result['sha256']}")


if __name__ == "__main__":
    main()
