"""Reproducible literature probe via the OpenAlex API (no key needed). Saves raw hits to results/literature/."""
import sys, json, time, requests, pathlib
OUT = pathlib.Path(__file__).resolve().parents[2] / "results" / "literature"; OUT.mkdir(parents=True, exist_ok=True)
def search(q, since=2021, n=15, tag=None):
    r = requests.get("https://api.openalex.org/works", params={
        "search": q, "filter": f"from_publication_date:{since}-01-01", "per-page": n,
        "select": "id,doi,title,publication_year,primary_location,cited_by_count,authorships"}, timeout=60,
        headers={"User-Agent": "mailto:ratulsoma2006@gmail.com"})
    r.raise_for_status()
    res = r.json()["results"]
    (OUT / f"{tag or q[:40].replace(' ','_')}.json").write_text(json.dumps({"query": q, "since": since, "results": res}, indent=1))
    print(f"\n### {q}")
    for w in res:
        src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
        print(f"- {w['publication_year']} | {w['title'][:110]} | {src} | {w.get('doi')}")
    time.sleep(0.4)
if __name__ == "__main__":
    for i, q in enumerate(sys.argv[1:]):
        search(q, tag=f"q_{int(time.time())}_{i}")
