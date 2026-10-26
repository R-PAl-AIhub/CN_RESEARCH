"""
paper/main.md : Markdown rendering of the paper generated from the LaTeX sources + paper/numbers.tex (same numbers as the PDF).
Tables are rebuilt from results/tables/*.csv, figures point to figures/*/*.png, equations are kept as LaTeX ($$ ... $$).
"""
import re
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "paper"
NUM = {}
for line in [l for f in ("numbers.tex", "numbers_base.tex") if (P / f).exists() for l in (P / f).read_text(encoding="utf-8").splitlines()]:
    m = re.match(r"\\expandafter\\def\\csname (.+?)\\endcsname\{(.*)\}$", line)
    if m:
        NUM[m.group(1)] = m.group(2)


def braced(s, i):
    """s[i] == '{' -> (content, index after matching '}')."""
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced")


def expand_macros(t):
    def nv(m):
        return NUM.get("num:" + m.group(1), "??" + m.group(1))
    def rv(m):
        return NUM.get("res:" + ":".join(m.groups()), "??rv")
    def sg(m):
        return NUM.get("sig:" + ":".join(m.groups()), "??sg")
    t = re.sub(r"\\nv\{([^}]*)\}", nv, t)
    t = re.sub(r"\\rv\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}", rv, t)
    t = re.sub(r"\\sg\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}", sg, t)
    return t


def expand_inputs(t):
    def rep(m):
        f = P / (m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")
        return expand_inputs(f.read_text(encoding="utf-8"))
    return re.sub(r"\\input\{([^}]+)\}", rep, t)


main = (P / "main.tex").read_text(encoding="utf-8")
body = main[main.index("\\begin{abstract}"):]
body = re.sub(r"%.*", "", body)
body = expand_inputs(body)
body = body.replace("\\bibliographystyle{IEEEtran}", "").replace("\\bibliography{references}", "\\REFS")
body = expand_macros(body)

# ---------- numbering of figures / tables / equations in order of appearance
order = {"fig": [], "tab": [], "eq": []}
for m in re.finditer(r"\\label\{(fig|tab|eq|sec):([^}]+)\}", body):
    k = m.group(1)
    if k in order and m.group(0) not in order[k]:
        order[k].append(m.group(0))
def roman(n):
    r = ""
    for v, s in ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= v:
            r += s
            n -= v
    return r
NUMS = {}
for k, lst in order.items():
    for i, lab in enumerate(lst, 1):
        NUMS[lab[len("\\label{"):-1]] = roman(i) if k == "tab" else str(i)
# number per FLOAT (labels sharing one float get the same number), exactly as LaTeX does
NUMS = {}
nf = nt = ne = 0
for m in re.finditer(r"\\begin\{(figure|table)\*?\}(.*?)\\end\{\1\*?\}|\\begin\{equation\}(.*?)\\end\{equation\}", body, re.S):
    if m.group(1) == "figure":
        nf += 1
        val, blk = str(nf), m.group(2)
    elif m.group(1) == "table":
        nt += 1
        val, blk = roman(nt), m.group(2)
    else:
        ne += 1
        val, blk = str(ne), m.group(3)
    for lab in re.findall(r"\\label\{([^}]+)\}", blk):
        NUMS[lab] = val

FIGFILES = {"fig_bar_f1": "results", "fig_ablation": "results", "fig_selection": "results", "fig_curves": "results", "fig_confusion": "results",
            "fig_sensitivity": "results", "fig_error": "results"}


def figure_block(blk):
    cap = ""
    i = blk.find("\\caption{")
    if i >= 0:
        cap, _ = braced(blk, i + len("\\caption"))
    labs = re.findall(r"\\label\{([^}]+)\}", blk)
    files = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", blk)
    n = NUMS.get(labs[0], "?") if labs else "?"
    imgs = []
    for f in files:
        stem = Path(f).stem
        sub = "results" if stem.startswith("fig_") else "diagrams"
        imgs.append(f"![{stem}](../figures/{sub}/{stem}.png)")
    return "\n\n" + "\n".join(imgs) + f"\n\n*Fig. {n}. {cap}*\n\n"


TABLE_CSV = {"tab:datasets": "table1_datasets", "tab:setup": "table2_setup", "tab:main": "table3_main_f1", "tab:ablation": "table4_ablation",
             "tab:sig": "table5_significance", "tab:cost": "table7_cost", "tab:related": "table8_related"}


def tex_inline(s):
    s = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", s)
    s = re.sub(r"\\(?:emph|textit)\{([^{}]*)\}", r"*\1*", s)
    s = re.sub(r"\\texttt\{([^{}]*)\}", r"`\1`", s)
    return s


def table_block(blk):
    cap, _ = braced(blk, blk.index("\\caption") + len("\\caption"))
    lab = re.findall(r"\\label\{([^}]+)\}", blk)
    n = NUMS.get(lab[0], "?")
    csv = ROOT / "results" / "tables" / (TABLE_CSV.get(lab[0], "") + ".csv")
    md = ""
    if csv.exists():
        df = pd.read_csv(csv).fillna("")
        cols = [tex_inline(re.sub(r"\\cite\{([^}]*)\}", r"[\1]", str(c))) for c in df.columns]
        md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
        for _, r in df.iterrows():
            md += "| " + " | ".join(tex_inline(re.sub(r"\\cite\{([^}]*)\}", r"[\1]", str(v))) for v in r.values) + " |\n"
    return f"\n\n**Table {n}.** {cap}\n\n{md}\n\n"


body = re.sub(r"\\begin\{(figure\*?)\}.*?\\end\{\1\}", lambda m: figure_block(m.group(0)), body, flags=re.S)
body = re.sub(r"\\begin\{(table\*?)\}.*?\\end\{\1\}", lambda m: table_block(m.group(0)), body, flags=re.S)


def alg_block(m):
    blk = m.group(0)
    cap, _ = braced(blk, blk.index("\\caption") + len("\\caption"))
    steps = re.findall(r"\\(?:State|Require|Ensure|For|EndFor|Return)\b(.*)", blk)
    lines = []
    for kind, txt in re.findall(r"\\(State|Require|Ensure|For|EndFor)\b([^\n]*)", blk):
        txt = re.sub(r"\\Comment\{([^}]*)\}", r"   // \1", txt)
        lines.append(f"{kind}: {txt.strip()}" if kind in ("Require", "Ensure") else ("end for" if kind == "EndFor" else ("for " + txt.strip() if kind == "For" else txt.strip())))
    cx = re.search(r"\\textit\{Complexity\.\}(.*?)\\end\{algorithm\}", blk, re.S)
    return f"\n\n**Algorithm 1. {cap}**\n\n```\n" + "\n".join(lines) + "\n```\n" + (("*Complexity.*" + cx.group(1)) if cx else "") + "\n\n"


body = re.sub(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", alg_block, body, flags=re.S)


def eq_block(m):
    inner = m.group(1)
    lab = re.findall(r"\\label\{([^}]+)\}", inner)
    inner = re.sub(r"\\label\{[^}]+\}", "", inner).replace("\\begin{gathered}", "").replace("\\end{gathered}", "")
    n = NUMS.get(lab[0], "") if lab else ""
    return f"\n\n$$\n{inner.strip()}\\tag{{{n}}}\n$$\n\n"


body = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", eq_block, body, flags=re.S)

body = re.sub(r"\\section\*?\{([^}]*)\}\\label\{[^}]*\}", r"\n\n## \1\n\n", body)
body = re.sub(r"\\section\*?\{([^}]*)\}", r"\n\n## \1\n\n", body)
body = re.sub(r"\\subsection\*?\{([^}]*)\}(\\label\{[^}]*\})?", r"\n\n### \1\n\n", body)
body = body.replace("\\begin{abstract}", "## Abstract\n\n").replace("\\end{abstract}", "")
body = re.sub(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", lambda m: "\n\n**Keywords:** " + m.group(1).strip() + "\n", body, flags=re.S)
body = re.sub(r"\\begin\{itemize\}|\\end\{itemize\}|\\begin\{enumerate\}|\\end\{enumerate\}", "\n", body)
body = re.sub(r"\\item\s*", "\n- ", body)
for _ in range(3):
    body = re.sub(r"\\textbf\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"**\1**", body)
    body = re.sub(r"\\(?:emph|textit)\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"*\1*", body)
    body = re.sub(r"\\texttt\{([^{}]*)\}", r"`\1`", body)
body = re.sub(r"\\cite\{([^}]*)\}", lambda m: "[" + "; ".join(k.strip() for k in m.group(1).split(",")) + "]", body)
body = re.sub(r"\\eqref\{([^}]*)\}", lambda m: f"Eq. ({NUMS.get(m.group(1), '?')})", body)
body = re.sub(r"\\ref\{fig:([^}]*)\}", lambda m: f"{NUMS.get('fig:' + m.group(1), '?')}", body)
body = re.sub(r"\\ref\{tab:([^}]*)\}", lambda m: f"{NUMS.get('tab:' + m.group(1), '?')}", body)
body = re.sub(r"Section\s*\\ref\{sec:([^}]*)\}", "the referenced section", body)
body = re.sub(r"\\ref\{sec:([^}]*)\}", "the referenced section", body)
body = re.sub(r"\\label\{[^}]*\}", "", body)
body = re.sub(r"\\url\{([^}]*)\}", r"<\1>", body)
for a, b in (("~", " "), ("\\,", " "), ("---", "—"), ("--", "–"), ("\\%", "%"), ("\\&", "&"), ("\\_", "_"), ("\\#", "#"), ("{[}", "["), ("{]}", "]"),
             ("\\\\", "\n"), ("{\\'a}", "á"), ("``", "“"), ("''", "”"), ("\\noindent", ""), ("\\footnotesize", ""), ("\\scriptsize", ""), ("\\hfill", ""),
             ("{,}", ","), ("\\ ", " "), ("\\paragraph", "")):
    body = body.replace(a, b)
body = re.sub(r"\\IEEEauthor[^\n]*", "", body)
body = body.replace("{\n", "\n").replace("}\n\n\\REFS", "\n\n\\REFS")

# references list from the bib (only cited keys)
cited = []
for m in re.finditer(r"\[([A-Za-z0-9_;\s]+)\]", body):
    for k in m.group(1).split(";"):
        k = k.strip()
        if re.match(r"^[a-z]+\d{4}[a-z]*$", k) and k not in cited:
            cited.append(k)
bib = (P / "references.bib").read_text(encoding="utf-8")
refs = ["\n## References\n"]
for i, k in enumerate(cited, 1):
    m = re.search(r"@\w+\{" + re.escape(k) + r",(.*?)\n\}", bib, re.S)
    if not m:
        continue
    g = lambda f: (re.search(f + r"\s*=\s*\{(.*)\},?\s*$", m.group(1), re.M) or [None, ""])[1]
    au = g("author").replace(" and ", ", ")
    title = re.sub(r"[{}]", "", g("title"))
    ven = re.sub(r"[{}]", "", g("journal") or g("booktitle") or g("howpublished") or g("note"))
    doi = g("doi")
    refs.append(f"{i}. **[{k}]** {au}, “{title},” *{ven}*, {g('year')}." + (f" doi:{doi}" if doi else ""))
body = body.replace("\\REFS", "\n".join(refs))
body = re.sub(r"\n{3,}", "\n\n", body)
title = re.search(r"\\title\{(.*)\}", main).group(1)
out = f"# {title}\n\n*Authors: [Author Name 1], [Author Name 2] — [Department, Institution, City, Country]* (placeholders)\n\n" + body.strip() + "\n"
(P / "main.md").write_text(out, encoding="utf-8")
left = re.findall(r"\\[a-zA-Z]+", out)
print("wrote paper/main.md;", len(out.split()), "words; leftover LaTeX commands:", sorted(set(left))[:20], "; unresolved '??':", out.count("??"))
