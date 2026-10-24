"""
Generate paper/references.bib from VERIFIED metadata (Crossref for DOIs, arXiv API for arXiv ids).
Keys / identifiers come from code/data/verify_refs.py (REFS). The senior paper is the only entry without a DOI: it is the
unpublished manuscript supplied with the project and is added by hand (flagged as such).
Usage: python code/data/make_bib.py [key1,key2,...]   (default: every verified key)
"""
import re
import sys
import time
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
UA = {"User-Agent": "research-pipeline (mailto:ratulsoma2006@gmail.com)"}
ver = json.load(open(ROOT / "results" / "literature" / "citation_verification.json"))
REFS = {k: v for k, v in ver.items()}
OVERRIDE_DOI = {"navia2022": ("doi", "10.1145/3539734")}          # corrected after the first verification pass
ACCEPT = {"yang2021"}                                            # Crossref stores given/family swapped; title+DOI verified manually


def tex(s):
    s = (s or "").replace("&amp;", "&")
    s = re.sub(r"<[^>]+>", "", s)
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")


def authors(lst):
    names = []
    for a in lst:
        fam, giv = a.get("family", ""), a.get("given", "")
        names.append(f"{fam}, {giv}".strip(", "))
    if len(names) > 6:
        names = names[:1] + ["others"]
    return " and ".join(names)


def crossref(doi):
    return requests.get(f"https://api.crossref.org/works/{doi}", headers=UA, timeout=60).json()["message"]


def bib_from_crossref(key, m):
    yr = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
    typ = m.get("type", "")
    title = tex((m.get("title") or [""])[0])
    au = authors(m.get("author", []))
    if key == "yang2021":                     # Crossref stores given/family swapped for this record (OpenAlex: Qin Yang, Masaaki Kondo)
        au = "Yang, Qin and Kondo, Masaaki"
    cont = tex((m.get("container-title") or [""])[0])
    pages = m.get("page") or m.get("article-number") or ""
    f = [f"  author = {{{au}}}", f"  title = {{{{{title}}}}}", f"  year = {{{yr}}}", f"  doi = {{{m['DOI']}}}"]
    if typ == "proceedings-article":
        f.insert(2, f"  booktitle = {{{cont}}}")
        kind = "inproceedings"
    elif typ in ("journal-article",):
        f.insert(2, f"  journal = {{{cont}}}")
        kind = "article"
        if m.get("volume"):
            f.append(f"  volume = {{{m['volume']}}}")
        if m.get("issue"):
            f.append(f"  number = {{{m['issue']}}}")
    else:
        f.insert(2, f"  howpublished = {{{cont}}}")
        kind = "misc"
    if pages:
        f.append(f"  pages = {{{pages}}}")
    return f"@{kind}{{{key},\n" + ",\n".join(f) + "\n}\n"


def bib_from_arxiv(key, aid, meta):
    return (f"@misc{{{key},\n  author = {{{tex(' and '.join(meta['authors']))}}},\n  title = {{{{{tex(meta['title'])}}}}},\n"
            f"  year = {{{meta['year']}}},\n  eprint = {{{aid}}},\n  archivePrefix = {{arXiv}},\n  howpublished = {{arXiv:{aid}}}\n}}\n")


VENUE = {   # arXiv preprints that were later published: cite the peer-reviewed venue (year/venue known from the venues' proceedings)
    "mcmahan2017": ("Proc. 20th Int. Conf. Artificial Intelligence and Statistics (AISTATS)", 2017, "1273--1282"),
    "li2020fedprox": ("Proc. Machine Learning and Systems (MLSys)", 2020, None),
    "arp2022": ("Proc. 31st USENIX Security Symposium", 2022, None),
    "li2022noniid": ("Proc. IEEE 38th Int. Conf. Data Engineering (ICDE)", 2022, None),
}


def bib_inproc_from_arxiv(key, aid, meta):
    bt, yr, pg = VENUE[key]
    f = [f"  author = {{{' and '.join(meta['authors'])}}}", f"  title = {{{{{tex(meta['title'])}}}}}", f"  booktitle = {{{bt}}}", f"  year = {{{yr}}}",
         f"  note = {{arXiv:{aid}}}"]
    if pg:
        f.append(f"  pages = {{{pg}}}")
    return f"@inproceedings{{{key},\n" + ",\n".join(f) + "\n}\n"


def arxiv_full(aid):
    r = requests.get("http://export.arxiv.org/api/query", params={"id_list": aid}, headers=UA, timeout=60).text
    t = " ".join(re.search(r"<entry>.*?<title>(.*?)</title>", r, re.S).group(1).split())
    au = re.findall(r"<name>(.*?)</name>", r)
    au = [f"{a.split()[-1]}, {' '.join(a.split()[:-1])}" for a in au]
    if len(au) > 6:
        au = au[:1] + ["others"]
    return dict(title=t, authors=au, year=int(re.search(r"<published>(\d{4})", r).group(1)))


keys = sys.argv[1].split(",") if len(sys.argv) > 1 else [k for k, v in REFS.items() if v["status"] == "VERIFIED" or k in ACCEPT]
keys = [k for k in keys if k != "senior2026"]
entries = []
for k in keys:
    kind, ident = OVERRIDE_DOI.get(k, (REFS[k]["kind"], REFS[k]["id"]))
    if k in OVERRIDE_DOI or REFS[k]["status"] in ("VERIFIED",) or k in ACCEPT:
        pass
    try:
        if kind == "doi":
            entries.append(bib_from_crossref(k, crossref(ident)))
        else:
            meta = arxiv_full(ident)
            entries.append(bib_inproc_from_arxiv(k, ident, meta) if k in VENUE else bib_from_arxiv(k, ident, meta))
    except Exception as e:
        print("FAILED", k, e)
    time.sleep(0.3)

entries.append("""@unpublished{senior2026,
  author = {Mohamed C P, Haris and Kundu, Anindita},
  title = {{Privacy-Preserving Federated Learning Framework for {DDoS} Attack Detection in {IoT} Networks Using Hybrid Feature Selection}},
  note = {Manuscript submitted to IEEE Access (senior paper supplied with this project; no DOI available)},
  year = {2026}
}
""")
(ROOT / "paper").mkdir(exist_ok=True)
(ROOT / "paper" / "references.bib").write_text("\n".join(entries), encoding="utf-8")
print("wrote", len(entries), "entries")
