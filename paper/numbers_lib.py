"""Shared helpers of the paper dossiers (paper/p*/): numbers and figures.

Every number a dossier cites is computed by its numbers.py from existing run records
and evaluation outputs, and written to numbers.json as {key: {value, text, source}}:
`text` is what render.py puts in place of {{key}} (French decimal comma), `source` the
repository paths read, separated by "; ". Figures are saved without a creation date so
that a rerun gives the same bytes (paper/check_dossiers.py compares them with git).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def fr(x: float, digits: int = 3) -> str:
    """Fixed-point text with a decimal comma."""
    return f"{x:.{digits}f}".replace(".", ",")


def sci(x: float, digits: int = 1) -> str:
    """Scientific text, e.g. 2,0·10⁻⁴."""
    exponent = math.floor(math.log10(abs(x)))
    mantissa = x / 10**exponent
    return f"{fr(mantissa, digits)}·10{str(exponent).translate(SUPERSCRIPT)}"


class Numbers:
    """Ordered collection written to numbers.json."""

    def __init__(self) -> None:
        self.data: dict[str, dict] = {}

    def add(
        self,
        key: str,
        value: float | int | str,
        source: str,
        digits: int | None = 3,
        text: str | None = None,
    ) -> None:
        if key in self.data:
            raise KeyError(f"duplicate key {key}")
        for path in source.split("; "):
            if not (ROOT / path).exists():
                raise FileNotFoundError(f"{key}: source {path} does not exist")
        if text is None:
            if isinstance(value, int) or digits is None:
                text = str(value)
            else:
                text = fr(float(value), digits)
        self.data[key] = {"value": value, "text": text, "source": source}

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.data, indent=1, ensure_ascii=False) + "\n")


def load(path: str) -> dict | list:
    """A JSON file of the repository."""
    return json.loads((ROOT / path).read_text())


def savefig(fig, path: Path) -> None:
    """PDF without creation date: the same bytes at every run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
