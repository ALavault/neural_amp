#!/usr/bin/env python3
"""Render a dossier's *.src.md into *.md: {{key}} becomes numbers.json[key]["text"].

render.py paper/<dossier>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

KEY = re.compile(r"\{\{\s*([A-Za-z0-9_.-]+)\s*\}\}")


def render(dossier: Path) -> list[Path]:
    numbers = json.loads((dossier / "numbers.json").read_text())
    written = []
    for src in sorted(dossier.glob("*.src.md")):
        text = src.read_text()
        missing = sorted({k for k in KEY.findall(text) if k not in numbers})
        if missing:
            raise SystemExit(f"{src}: keys missing from numbers.json: {missing}")
        target = src.with_name(src.name.replace(".src.md", ".md"))
        header = f"<!-- Généré par paper/render.py depuis {src.name}. -->\n\n"
        target.write_text(header + KEY.sub(lambda m: numbers[m.group(1)]["text"], text))
        written.append(target)
    return written


if __name__ == "__main__":
    for path in render(Path(sys.argv[1])):
        print(path)
