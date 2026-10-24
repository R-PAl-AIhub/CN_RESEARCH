"""
Originality check: longest verbatim word runs shared between the paper prose and (a) the base paper, (b) the senior paper,
(c) the abstracts of cited works retrieved from Crossref/OpenAlex. Reports every shared run of >= N words.
Usage: python code/plots/originality_check.py [N=7]
"""
import re
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 7


def words(t):
    return re.findall(r"[a-z0-9]+", t.lower())


md = (ROOT / "paper" / "main.md").read_text(encoding="utf-8")
md = md.split("## References")[0]
md = re.sub(r"\$\$.*?\$\$", " ", md, flags=re.S)
paper = words(md)
sources = {"base": (ROOT / "results" / "literature" / "source_text" / "base_paper.txt").read_text(encoding="utf-8"),
           "senior": (ROOT / "results" / "literature" / "source_text" / "senior_paper.txt").read_text(encoding="utf-8")}
for k, v in json.load(open(ROOT / "results" / "literature" / "abstracts.json")).items():
    sources["abstract:" + k] = v
report = []
for name, txt in sources.items():
    src = words(txt)
    grams = {}
    for i in range(len(src) - N + 1):
        grams.setdefault(tuple(src[i:i + N]), i)
    i, found = 0, []
    while i < len(paper) - N + 1:
        if tuple(paper[i:i + N]) in grams:
            j = i + N
            s = grams[tuple(paper[i:i + N])]
            while j < len(paper) and s + (j - i) < len(src) and paper[j] == src[s + (j - i)]:
                j += 1
            found.append(" ".join(paper[i:j]))
            i = j
        else:
            i += 1
    report.append((name, found))
tot = sum(len(f) for _, f in report)
print(f"paper prose words: {len(paper)}; shared verbatim runs of >= {N} words: {tot}")
for name, f in report:
    for x in f:
        print(f"  [{name}] ({len(x.split())} words) {x}")
