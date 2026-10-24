"""
Reference verification (integrity rule 3): every candidate reference is looked up in Crossref (DOI) or the arXiv API
and compared with the expected first-author surname, year and a title fragment. Output: results/literature/citation_verification.json
Usage: python code/data/verify_refs.py
"""
import json
import re
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]
UA = {"User-Agent": "research-pipeline (mailto:ratulsoma2006@gmail.com)"}

# key : (kind, id, first-author surname, year, title fragment)
REFS = {
    "mcmahan2017": ("arxiv", "1602.05629", "McMahan", 2017, "communication-efficient learning of deep networks"),
    "li2020fedprox": ("arxiv", "1812.06127", "Li", 2018, "federated optimization in heterogeneous networks"),
    "kairouz2021": ("doi", "10.1561/2200000083", "Kairouz", 2021, "advances and open problems in federated learning"),
    "hsu2019": ("arxiv", "1909.06335", "Hsu", 2019, "measuring the effects of non-identical data distribution"),
    "li2022noniid": ("arxiv", "2102.02079", "Li", 2021, "federated learning on non-iid data silos"),
    "karunamurthy2025": ("doi", "10.1038/s41598-025-93501-8", "Karunamurthy", 2025, "optimal federated learning-based intrusion detection"),
    "neto2023": ("doi", "10.3390/s23135941", "Neto", 2023, "ciciot2023"),
    "meidan2018": ("doi", "10.1109/MPRV.2018.03367731", "Meidan", 2018, "n-baiot"),
    "vaccari2020": ("doi", "10.3390/s20226578", "Vaccari", 2020, "mqttset"),
    "bouzinis2025": ("doi", "10.1109/TNSM.2025.3564387", "Bouzinis", 2025, "statavg"),
    "nguyen2026": ("doi", "10.1186/s42400-025-00509-8", "Nguyen", 2026, "feature reduction in federated learning"),
    "doriguzzi2024": ("doi", "10.1016/j.cose.2023.103597", "Doriguzzi-Corin", 2024, "flad"),
    "devine2025": ("doi", "10.3390/electronics14061176", "Devine", 2025, "federated machine learning to enable intrusion detection"),
    "agrawal2022": ("doi", "10.1016/j.comcom.2022.09.012", "Agrawal", 2022, "federated learning for intrusion detection system"),
    "yang2021": ("doi", "10.1109/ICECCE52056.2021.9514222", "Yang", 2021, "federated learning-based network intrusion detection with a feature selection"),
    "fdids2025": ("doi", "10.3390/s25144309", "", 2025, "fd-ids"),
    "ambusaidi2016": ("doi", "10.1109/TC.2016.2519914", "Ambusaidi", 2016, "filter-based feature selection"),
    "bonawitz2017": ("doi", "10.1145/3133956.3133982", "Bonawitz", 2017, "practical secure aggregation"),
    "arp2022": ("arxiv", "2010.09470", "Arp", 2020, "dos and don'ts of machine learning in computer security"),
    "kapoor2023": ("doi", "10.1016/j.patter.2023.100804", "Kapoor", 2023, "leakage and the reproducibility crisis"),
    "engelen2021": ("doi", "10.1109/SPW53761.2021.00009", "Engelen", 2021, "troubleshooting an intrusion detection dataset"),
    "rashid2023": ("doi", "10.3390/network3010008", "Rashid", 2023, "federated learning-based approach for improving intrusion detection"),
    "ruzafa2023": ("doi", "10.1109/TII.2021.3126728", "Ruzafa", 2023, "intrusion detection based on privacy-preserving federated learning"),
    "saez2023": ("doi", "10.1016/j.cose.2023.103299", "", 2023, "clustered federated learning architecture"),
    "azimjonov2024": ("doi", "10.1016/j.cose.2023.103598", "Azimjonov", 2024, "lightweight intrusion detection"),
    "navia2022": ("doi", "10.1145/3539734", "Navia", 2022, "budget distributed support vector machine"),
    "cortes1995": ("doi", "10.1007/BF00994018", "Cortes", 1995, "support-vector networks"),
    "buczak2016": ("doi", "10.1109/COMST.2015.2494502", "Buczak", 2016, "survey of data mining and machine learning methods"),
    "chan1983": ("doi", "10.1080/00031305.1983.10483115", "Chan", 1983, "algorithms for computing the sample variance"),
    "fedDWC2025": ("doi", "10.1038/s41598-025-13204-y", "", 2025, "dynamic weight clustered federated learning"),
    "datasetcentric2026": ("doi", "10.1038/s41598-025-32567-w", "", 2026, "dataset-centric evaluation of federated intrusion detection"),
    "breiman2001": ("doi", "10.1023/A:1010933404324", "Breiman", 2001, "random forests"),
    "mothukuri2021": ("doi", "10.1016/j.future.2020.10.007", "Mothukuri", 2021, "survey on security and privacy of federated learning"),
}


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower())


def check_crossref(doi):
    r = requests.get(f"https://api.crossref.org/works/{doi}", headers=UA, timeout=60)
    if r.status_code != 200:
        return None
    m = r.json()["message"]
    yr = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
    return dict(title=(m.get("title") or [""])[0], first_author=(m.get("author") or [{}])[0].get("family", ""), year=yr,
                venue=(m.get("container-title") or [""])[0], authors=[a.get("family", "") for a in m.get("author", [])][:6])


def check_arxiv(aid):
    r = requests.get("http://export.arxiv.org/api/query", params={"id_list": aid}, headers=UA, timeout=60)
    if r.status_code != 200 or "<entry>" not in r.text:
        return None
    t = re.search(r"<entry>.*?<title>(.*?)</title>", r.text, re.S).group(1)
    au = re.findall(r"<name>(.*?)</name>", r.text)
    yr = int(re.search(r"<published>(\d{4})", r.text).group(1))
    return dict(title=" ".join(t.split()), first_author=au[0].split()[-1] if au else "", year=yr, venue="arXiv", authors=[a.split()[-1] for a in au][:6])


out = {}
for key, (kind, ident, fa, yr, frag) in REFS.items():
    for attempt in range(3):
        try:
            meta = check_crossref(ident) if kind == "doi" else check_arxiv(ident)
            break
        except Exception as e:                       # network hiccup -> retry
            meta, err = None, str(e)
            time.sleep(2)
    if meta is None:
        out[key] = dict(status="NOT_FOUND", kind=kind, id=ident)
        print(f"{key:20s} NOT FOUND ({ident})")
        continue
    ok_title = norm(frag) in norm(meta["title"])
    ok_auth = (not fa) or norm(fa) in norm(meta["first_author"]) or any(norm(fa) in norm(a) for a in meta["authors"])
    ok_year = meta["year"] is not None and abs(int(meta["year"]) - yr) <= 1
    out[key] = dict(status="VERIFIED" if (ok_title and ok_auth and ok_year) else "CHECK", kind=kind, id=ident, expected=dict(first_author=fa, year=yr, title_fragment=frag), found=meta,
                    ok=dict(title=ok_title, author=ok_auth, year=ok_year))
    print(f"{key:20s} {out[key]['status']:9s} {meta['first_author']:>16s} {meta['year']} | {meta['title'][:70]}")
    time.sleep(0.3)
(ROOT / "results" / "literature" / "citation_verification.json").write_text(json.dumps(out, indent=1))
print("verified:", sum(v['status'] == 'VERIFIED' for v in out.values()), "of", len(out))
