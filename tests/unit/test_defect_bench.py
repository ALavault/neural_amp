"""Known-defect bench used to calibrate deviation scores."""

from __future__ import annotations

import numpy as np
import pytest

from fssr_nam.product import defect_bench as db

FS = 48_000.0


def test_every_variant_at_reference_value_is_the_reference() -> None:
    x = db.excitation("noise", -20.0, 0.5, 3, FS)
    for defect, (base, name, _) in db.DEFECTS.items():
        if defect == "noise_db":
            continue  # the reference has no noise at all
        _, same = db.variant(defect, getattr(base, name))
        # the redistribution retunes the drive by root finding (xtol 1e-10)
        np.testing.assert_allclose(
            db.apply(base, x, FS), db.apply(same, x, FS), rtol=0, atol=1e-12
        )


def test_redistribution_keeps_thd_and_moves_harmonics() -> None:
    base, changed = db.variant("redistribution_bias", 0.3)
    assert db.thd(changed, db.THD_LEVEL_DB) == pytest.approx(
        db.thd(base, db.THD_LEVEL_DB), rel=1e-6
    )
    t = np.arange(round(0.3 * FS)) / FS
    sine = 10 ** (db.THD_LEVEL_DB / 20) * np.sin(2 * np.pi * 1_000 * t)

    def h2_over_h3(chain: db.Chain) -> float:
        spectrum = np.abs(np.fft.rfft(db.apply(chain, sine, FS)[-4_800:]))
        return spectrum[200] / spectrum[300]

    assert h2_over_h3(changed) > 2 * h2_over_h3(base)


def test_synchronized_sweep_has_integer_f0_lag() -> None:
    _, sweep = db.synchronized_sweep(5.0, FS)
    assert sweep.f0 * sweep.lag == pytest.approx(round(sweep.f0 * sweep.lag), abs=1e-9)


def test_multisine_leaves_detection_lines_empty() -> None:
    x = db.multisine_signal(1.0, FS, seed=4)
    spectrum = np.abs(np.fft.rfft(x))
    excited = spectrum[21:20_000:2]
    assert np.all(spectrum[22:20_000:2] < 1e-9 * excited.max())  # even lines
    assert np.all(excited[3::4] < 1e-9 * excited.max())  # one odd line in four


@pytest.mark.parametrize("family", ["noise", "multisine", "sweep", "two_tone"])
def test_excitation_level_is_exact(family: str) -> None:
    kind = db.LEVEL_KIND[family]
    x = db.excitation(family, -10.0, 0.5, 1, FS)
    level = np.sqrt(np.mean(x**2)) if kind == "rms" else np.max(np.abs(x))
    assert 20 * np.log10(level) == pytest.approx(-10.0, abs=1e-9)
