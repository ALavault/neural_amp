"""Render the review with installed pdfLaTeX and linked bibliography entries."""

import argparse
import json
import re
import subprocess
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def escape(text):
    # Express diacritics as LaTeX accents for the installed T1 fonts.
    text = text.replace('œ', 'oe').replace('Œ', 'Oe').replace('ł', 'l')
    text = text.replace('’', "'").replace('–', '-').replace('—', '-')
    text = text.replace('≥', '>=')
    specials = {
        '\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
        '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
        '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
    }
    accents = {'\u0301': "'", '\u0300': '`', '\u0308': '"',
               '\u0302': '^', '\u0303': '~', '\u030c': 'v',
               '\u030b': 'H', '\u030a': 'r', '\u0327': 'c',
               '\u0304': '=', '\u0306': 'u', '\u0307': '.', '\u0328': 'k'}
    result = []
    for character in text:
        decomposition = unicodedata.normalize('NFD', character)
        if len(decomposition) == 2 and decomposition[1] in accents:
            result.append('\\' + accents[decomposition[1]] + '{' + decomposition[0] + '}')
        else:
            result.append(specials.get(character, character))
    return ''.join(result)


def inline(text):
    parts = re.split(r'(\[@[^\]]+\]|\[[^\]]+\]\(https?://[^)]+\))', text)
    output = []
    for part in parts:
        if part.startswith('[@'):
            keys = [key.strip().lstrip('@') for key in part[1:-1].split(';')]
            output.append(r'\cite{' + ','.join(keys) + '}')
        elif re.fullmatch(r'\[[^\]]+\]\(https?://[^)]+\)', part):
            label, url = re.fullmatch(r'\[([^\]]+)\]\((https?://[^)]+)\)', part).groups()
            output.append(r'\href{' + url + '}{' + escape(label) + '}')
        else:
            output.append(escape(part.replace('`', '')))
    return ''.join(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='review.md')
    parser.add_argument('--output', default='review')
    parser.add_argument('--title', default="Au-dela du score moyen")
    args = parser.parse_args()
    text = (ROOT / args.source).read_text()
    text = re.sub(r'^---\n.*?\n---\n', '', text, flags=re.S)
    corpus = json.loads((ROOT / 'sources/corpus.json').read_text())
    known = {row['key'] for row in corpus}
    used = set(re.findall(r'@([a-z]+\d+[a-z]?)', text))
    if not used <= known:
        raise ValueError(f'Missing references: {used - known}')
    heading = escape(args.title)
    output = [r'''\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=22mm]{geometry}
\usepackage{longtable}
\usepackage[colorlinks=true,urlcolor=blue,citecolor=blue,linkcolor=blue]{hyperref}
\renewcommand{\contentsname}{Sommaire}
\renewcommand{\refname}{Références}
\setlength{\emergencystretch}{4em}
\setlength{\parskip}{5pt}
\setlength{\parindent}{0pt}
\begin{document}
\begin{center}\Large ''' + heading + r'''\\
\large Emulation d'amplificateurs : recherche et audit de validite\\
\normalsize 5 septembre 2026 -- Note exploratoire\end{center}
\tableofcontents
\newpage
''']
    table = False
    for line in text.splitlines():
        if line.startswith('|'):
            cells = [value.strip() for value in line.strip('|').split('|')]
            if all(re.fullmatch(r'[-: ]+', cell) for cell in cells):
                continue
            if not table:
                output.append(r'\begin{longtable}{p{0.06\linewidth}p{0.23\linewidth}p{0.44\linewidth}p{0.15\linewidth}}')
                table = True
            output.append(' & '.join(inline(cell) for cell in cells) + r' \\ \hline')
            continue
        if table:
            output.append(r'\end{longtable}')
            table = False
        if line == '# References':
            break
        if line.startswith('### '):
            output.append(r'\subsubsection{' + inline(line[4:]) + '}')
        elif line.startswith('## '):
            output.append(r'\subsection{' + inline(line[3:]) + '}')
        elif line.startswith('# '):
            output.append(r'\section{' + inline(line[2:]) + '}')
        elif line.startswith('- '):
            output.append(r'\par\noindent\textbullet\ ' + inline(line[2:]))
        else:
            output.append(inline(line))
    if used:
        output.append(r'\begin{thebibliography}{99}')
        for paper in corpus:
            if paper['key'] not in used:
                continue
            reference = f"{paper['authors']}. {paper['title']}. {paper['year']}."
            url = 'https://doi.org/' + paper['doi'] if paper['doi'] else paper['url']
            output.append(r'\bibitem{' + paper['key'] + '}' + escape(reference)
                          + r' \href{' + url + r'}{Source primaire}.')
        output.append(r'\end{thebibliography}')
    output.append(r'\end{document}')
    destination = ROOT / 'output/pdf'
    destination.mkdir(parents=True, exist_ok=True)
    tex = destination / f'{args.output}.tex'
    tex.write_text('\n'.join(output))
    for _ in range(2):
        result = subprocess.run(['pdflatex', '-interaction=nonstopmode',
            '-halt-on-error', tex.name], cwd=destination, capture_output=True,
            text=True, errors='replace')
        if result.returncode:
            raise RuntimeError(result.stdout[-3000:])
    print(destination / f'{args.output}.pdf')


if __name__ == '__main__':
    main()
