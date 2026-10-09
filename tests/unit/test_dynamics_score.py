"""Envelope-dynamics score on level steps (ad hoc score E2)."""

from __future__ import annotations

import pytest

from fssr_nam.product import defect_bench as db
from fssr_nam.product.dynamics import dynamic_deviation, step_kinds

FS = 48_000.0


@pytest.fixture(scope="module")
def steps():
    return db.steps_signal(-6.0, 7, FS)


def score(steps, defect: str, severity: float) -> dict:
    x, boundaries = steps
    base, changed = db.variant(defect, severity, base=db.DYNAMIC)
    return dynamic_deviation(
        x, db.apply(base, x, FS), db.apply(changed, x, FS), FS, boundaries
    )


def test_steps_alternate_release_and_attack(steps) -> None:
    x, boundaries = steps
    kinds = step_kinds(x, boundaries, FS)
    assert len(kinds["release"]) == 22 and len(kinds["attack"]) == 21


def test_static_gain_change_scores_zero(steps) -> None:
    out = score(steps, "gain", 1.06)
    assert out["release"]["score_db"] < 1e-9 and out["attack"]["score_db"] < 1e-9


def test_release_change_moves_release_not_attack(steps) -> None:
    small, large = score(steps, "release_ms", 60.0), score(steps, "release_ms", 200.0)
    assert 0 < small["release"]["score_db"] < large["release"]["score_db"]
    assert large["attack"]["score_db"] < large["release"]["score_db"] / 3
