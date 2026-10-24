"""
Phase-6 integrity audit: list every number that appears in the paper prose OUTSIDE result macros (\\nv, \\rv, \\sg), so each can be
traced to a design constant, a cited source, or a result file. Equations, tables, labels, citations, references are removed first.
"""
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[2] / "paper"
files = ["abstract", "sec_intro", "sec_related", "sec_method", "sec_setup", "sec_results", "sec_discussion", "sec_threats", "sec_statements"]
hits = []
for f in files:
    t = (PAPER / f"{f}.tex").read_text(encoding="utf-8")
    t = re.sub(r"%.*", "", t)
    t = re.sub(r"\\(nv|rv|sg)(\{[^}]*\})+", " MACRO ", t)                       # result macros
    t = re.sub(r"\\begin\{(equation|algorithmic|figure\*?|table\*?)\}.*?\\end\{\1\}", " ", t, flags=re.S)
    t = re.sub(r"\\(cite|ref|eqref|label|input|includegraphics)(\[[^\]]*\])?\{[^}]*\}", " ", t)
    t = re.sub(r"\$[^$]*\$", " ", t)                                             # inline math ($K{=}10$ etc. listed separately below)
    for m in re.finditer(r"(?<![A-Za-z_\\])(\d[\d,\.]*\d|\d)", t):
        s = max(0, m.start() - 45)
        hits.append((f, m.group(1), re.sub(r"\s+", " ", t[s:m.end() + 35])))
print(f"{len(hits)} numeric literals outside macros/equations/math")
for f, n, ctx in hits:
    print(f"{f:14s} {n:>10s} | {ctx}")
