#!/usr/bin/env python3
"""Checks of the paper dossiers paper/p1_alignment, p2_butterfly, p3_sample_rate and
p4_probe_metrics (dossiers without prose; D-PRODUCT-002). Run from anywhere:

    uv run python paper/check_dossiers.py [dossier ...]

Per dossier:
- files: numbers.py, numbers.json, RESULTS.src.md, RESULTS.md, CLAIMS.src.md, CLAIMS.md,
  OUTLINE.md, figures.py, refs.bib, RELATED.md; no .tex file;
- sealed: numbers.py and figures.py name no sealed capture (Blackstar, UA 1176,
  EXTERNAL_REPORT_ONLY) and no KIPR material;
- regenerated: numbers.py, render.py and figures.py rerun, then `git status` of the
  dossier is empty (every file identical to its committed version, nothing untracked);
- numbers: each entry has value, text and existing source paths;
- decimals: no decimal or scientific number in *.src.md outside {{key}};
- no prose: every non-blank line of *.src.md, OUTLINE.md and RELATED.md is a heading,
  a bullet (or its indented continuation), a numbered item, a table row or a comment;
- results: headings Établi, Préliminaire, Retiré, En attente in RESULTS.src.md;
- claims: the label « exploratoire, D-PRODUCT-002 », and in the table whose header
  has « Preuve », each row cites an existing path and a numbers.json key in backticks;
- outline: « à confirmer » and « pages » present; every figures/<name>.pdf it names
  exists; each statement T<n> of its table with « Statut » carries a status found in
  the « Statut : » line of THEORY.md, and the dossier's expected statements are there;
- refs: every refs.bib entry with a DOI resolves at https://doi.org/api/handles/<doi>
  (responseCode 1); an entry without DOI is preceded by a « % CHECK » line; every key
  appears in RELATED.md on a table row marked [V] or [NV].
Prints one row per check and exits 0 only if every check passes.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
DOSSIERS = ("p1_alignment", "p2_butterfly", "p3_sample_rate", "p4_probe_metrics")
EXPECTED_THEORY = {
    "p1_alignment": {"T2"},
    "p2_butterfly": {"T3"},
    "p3_sample_rate": {"T1", "T4"},
    "p4_probe_metrics": set(),
}
REQUIRED = (
    "numbers.py",
    "numbers.json",
    "RESULTS.src.md",
    "RESULTS.md",
    "CLAIMS.src.md",
    "CLAIMS.md",
    "OUTLINE.md",
    "figures.py",
    "refs.bib",
    "RELATED.md",
)
SEALED = re.compile(r"blackstar|ua[ _-]?1176|EXTERNAL_REPORT_ONLY|kipr", re.IGNORECASE)
KEY = re.compile(r"\{\{\s*[A-Za-z0-9_.-]+\s*\}\}")
DECIMAL = re.compile(r"\d[.,]\d|\d[eE][-+]?\d|\d·10|10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]|\d[⁰¹²³⁴⁵⁶⁷⁸⁹]")
ITEM = re.compile(r"^\s*(#{1,6}\s|[-*+]\s|\d+[.)]\s|\||<!--)")
CONTINUATION = re.compile(r"^\s{2,}\S")
DOI = re.compile(r"doi\s*=\s*[{\"]([^}\"]+)[}\"]", re.IGNORECASE)
ENTRY = re.compile(r"^@\w+\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)
_doi_cache: dict[str, bool] = {}


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True)


def resolves(doi: str) -> bool:
    if doi not in _doi_cache:
        ok = False
        for attempt in range(3):
            try:
                url = f"https://doi.org/api/handles/{doi}"
                with urllib.request.urlopen(url, timeout=30) as response:
                    ok = json.load(response).get("responseCode") == 1
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        _doi_cache[doi] = ok
    return _doi_cache[doi]


def paragraphs(path: Path) -> list[int]:
    """Line numbers that are neither an item nor the continuation of one."""
    bad, previous_item, in_comment = [], False, False
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if in_comment:
            in_comment = "-->" not in line
        elif not line.strip():
            previous_item = False
        elif ITEM.match(line):
            previous_item = True
            in_comment = line.lstrip().startswith("<!--") and "-->" not in line
        elif not (previous_item and CONTINUATION.match(line)):
            bad.append(number)
    return bad


def table_rows(text: str, header_word: str) -> list[list[str]]:
    """Rows of the first table whose header contains header_word."""
    rows, inside = [], False
    for line in text.splitlines():
        if not line.startswith("|"):
            if inside:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not inside:
            inside = any(header_word in c for c in cells)
        elif not set("".join(cells)) <= set("-: "):
            rows.append(cells)
    return rows


def theory_status() -> dict[str, str]:
    status, current = {}, None
    for line in (ROOT / "THEORY.md").read_text().splitlines():
        match = re.match(r"^## (T\d+) ", line)
        if match:
            current = match.group(1)
        elif current and line.startswith("Statut :"):
            status[current] = line.removeprefix("Statut :").strip()
            current = None
    return status


def regenerated(name: str) -> tuple[bool, str]:
    steps = [
        run(sys.executable, f"paper/{name}/numbers.py"),
        run(sys.executable, "paper/render.py", f"paper/{name}"),
        run(sys.executable, f"paper/{name}/figures.py"),
    ]
    for step in steps:
        if step.returncode != 0:
            return False, f"{step.args[1]} failed: {step.stderr[-300:]}"
    status = run("git", "status", "--porcelain", "--", f"paper/{name}").stdout.strip()
    return not status, status.replace("\n", "; ") or "identical to committed"


def numbers_ok(numbers: dict) -> tuple[bool, str]:
    bad = [
        k
        for k, v in numbers.items()
        if not {"value", "text", "source"} <= set(v)
        or not all((ROOT / p).exists() for p in v["source"].split("; "))
    ]
    return bool(numbers) and not bad, f"{len(numbers)} keys, bad {bad[:5]}"


def typed_decimals(d: Path) -> tuple[bool, str]:
    typed = [
        f"{src.name}:{n}"
        for src in sorted(d.glob("*.src.md"))
        for n, line in enumerate(src.read_text().splitlines(), 1)
        if DECIMAL.search(KEY.sub("", line))
    ]
    return not typed, ", ".join(typed[:8]) or "none typed"


def no_prose(d: Path) -> tuple[bool, str]:
    files = [*sorted(d.glob("*.src.md")), d / "OUTLINE.md", d / "RELATED.md"]
    prose = [f"{f.name}:{n}" for f in files for n in paragraphs(f)]
    return not prose, ", ".join(prose[:8]) or "items only"


def results_sections(d: Path) -> tuple[bool, str]:
    text = (d / "RESULTS.src.md").read_text()
    heads = ("Établi", "Préliminaire", "Retiré", "En attente")
    lost = [h for h in heads if not re.search(rf"^#+ .*{h}", text, re.MULTILINE)]
    return not lost, f"missing {lost}" if lost else "4/4"


def claims_ok(d: Path, numbers: dict) -> tuple[bool, str]:
    text = (d / "CLAIMS.src.md").read_text()
    rows = table_rows(text, "Preuve")
    unproved = []
    for i, row in enumerate(rows, 1):
        ticks = re.findall(r"`([^`]+)`", " ".join(row))
        if not (
            any((ROOT / t).exists() for t in ticks) and any(t in numbers for t in ticks)
        ):
            unproved.append(i)
    ok = "exploratoire, D-PRODUCT-002" in text and bool(rows) and not unproved
    return ok, f"{len(rows)} rows, unproved rows {unproved}"


def outline_ok(d: Path, name: str) -> tuple[bool, str]:
    text = (d / "OUTLINE.md").read_text()
    figures = sorted(set(re.findall(r"figures/[\w.-]+\.pdf", text)))
    absent = [f for f in figures if not (d / f).exists()]
    theory = theory_status()
    listed = {
        r[0]: r[1]
        for r in table_rows(text, "Statut")
        if re.fullmatch(r"T\d+", r[0]) and len(r) > 1
    }
    wrong = [t for t, s in listed.items() if not s or s not in theory.get(t, "")]
    lacking = sorted(EXPECTED_THEORY[name] - set(listed))
    ok = (
        "à confirmer" in text
        and "pages" in text
        and bool(figures)
        and not (absent or wrong or lacking)
    )
    detail = (
        f"{len(figures)} figures, absent {absent}, statements {sorted(listed)}, "
        f"wrong status {wrong}, lacking {lacking}"
    )
    return ok, detail


def refs_ok(d: Path) -> tuple[bool, str]:
    chunks = re.split(r"(?=^@)", (d / "refs.bib").read_text(), flags=re.MULTILINE)
    related = [
        line for line in (d / "RELATED.md").read_text().splitlines() if line[:1] == "|"
    ]
    entries, dois, failed, unchecked, unrelated = 0, 0, [], [], []
    for i, chunk in enumerate(chunks):
        match = ENTRY.match(chunk)
        if not match:
            continue
        entries += 1
        key = match.group(1)
        doi = DOI.search(chunk)
        if doi:
            dois += 1
            if not resolves(doi.group(1).strip()):
                failed.append(key)
        else:
            before = [x for x in chunks[i - 1].splitlines() if x.strip()] if i else []
            if not before or before[-1].strip() != "% CHECK":
                unchecked.append(key)
        rows = [r for r in related if key in r]
        if not any("[V]" in r or "[NV]" in r for r in rows):
            unrelated.append(key)
    ok = bool(entries) and not (failed or unchecked or unrelated)
    detail = (
        f"{entries} entries, {dois} DOI resolved live, failed {failed}, "
        f"no DOI without % CHECK {unchecked}, not in RELATED {unrelated}"
    )
    return ok, detail


def check(name: str) -> list[tuple[str, bool, str]]:
    d = PAPER / name
    missing = [f for f in REQUIRED if not (d / f).exists()]
    out = [("files", not missing, f"missing {missing}" if missing else "10/10")]
    if missing:
        return out
    tex = [str(p.relative_to(d)) for p in d.rglob("*.tex")]
    out.append(("no .tex", not tex, ", ".join(tex) or "none"))
    scripts = ("numbers.py", "figures.py")
    sealed = [f for f in scripts if SEALED.search((d / f).read_text())]
    out.append(("sealed/KIPR", not sealed, ", ".join(sealed) or "none named"))
    out.append(("regenerated", *regenerated(name)))
    numbers = json.loads((d / "numbers.json").read_text())
    out.append(("numbers", *numbers_ok(numbers)))
    out.append(("decimals", *typed_decimals(d)))
    out.append(("no prose", *no_prose(d)))
    out.append(("results sections", *results_sections(d)))
    out.append(("claims", *claims_ok(d, numbers)))
    out.append(("outline", *outline_ok(d, name)))
    out.append(("refs", *refs_ok(d)))
    return out


def main() -> None:
    names = sys.argv[1:] or list(DOSSIERS)
    all_ok = True
    print(f"{'dossier':<18} {'check':<17} {'':<4} detail")
    for name in names:
        for check_name, ok, detail in check(name):
            all_ok &= ok
            print(f"{name:<18} {check_name:<17} {'PASS' if ok else 'FAIL'} {detail}")
    print("ALL PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
