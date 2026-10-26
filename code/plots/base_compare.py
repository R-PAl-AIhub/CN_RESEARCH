"""
Base-paper comparison tables (Karunamurthy et al. 2025, re-implemented as `base_coa_cnn`).
Reads results/raw/final_*.jsonl (all paper methods) and results/raw/base_*.jsonl (base method, same corpora and partitions).
Writes:
  results/tables/base_scoreboard.csv  test F1 mean/std, failures (F1 < 0.9) and feature count per method and regime
  results/tables/base_tests.csv       paired Wilcoxon tests of the base method against FedStat-FS, FedAvg-MLP and Senior.
                                      Separate family, added post hoc at the supervisor's request; Holm within this family
                                      only (the paper's 54-test family is not changed).
  results/tables/base_cost.csv        cost of the base method (results/base/cost_profile_base.json) next to the paper's
                                      cost profile (results/cost_profile.json)
"""
import json
import glob
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_lib import ROOT, RAW, CONDS, MAIN, load_final, cell, mean_std, paired_tests   # noqa: E402

BASE = "base_coa_cnn"
TAB = ROOT / "results" / "tables"
FAMILY = ["fedstat", "fedstat_mlp", "senior"]          # fixed before the base results were seen (decisions_log D13)


def load_all():
    rows = []
    for f in sorted(glob.glob(str(RAW / "base_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            rows.append(json.loads(line))
    return pd.concat([load_final(), pd.DataFrame(rows)], ignore_index=True)


def main():
    df = load_all()
    board = []
    for m in MAIN + [BASE]:
        for ds, conds in CONDS.items():
            for c in conds:
                f1 = cell(df, ds, c, m)
                nf = cell(df, ds, c, m, "n_feat")
                mu, sd = mean_std(f1)
                board.append(dict(method=m, dataset=ds, cond=c, f1_mean=mu, f1_std=sd, n_runs=len(f1),
                                  n_fail=int((f1 < 0.9).sum()), n_feat_mean=float(np.mean(nf)) if len(nf) else np.nan))
    board = pd.DataFrame(board)
    board.to_csv(TAB / "base_scoreboard.csv", index=False)

    tests = []
    for ds, conds in CONDS.items():
        for c in conds:
            b = cell(df, ds, c, BASE)
            for A in FAMILY:
                a = cell(df, ds, c, A)
                if len(a) == 0 or len(a) != len(b):
                    continue
                tests.append(dict(dataset=ds, cond=c, A=A, B=BASE, **paired_tests(a, b)))
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["p_w_holm_family"] = multipletests(tests["p_w"].fillna(1.0), method="holm")[1]
    tests.to_csv(TAB / "base_tests.csv", index=False)

    cost_f = ROOT / "results" / "base" / "cost_profile_base.json"
    if cost_f.exists():
        prof = json.load(open(ROOT / "results" / "cost_profile.json"))
        prof[BASE] = json.load(open(cost_f))
        cols = ["method", "n_feat", "n_params", "train_sec", "prep_sec", "coa_sec", "infer_ms_per_1k", "comm_model_bytes", "comm_prep_bytes"]
        pd.DataFrame([{k: v.get(k, np.nan) if k != "method" else m for k in cols} for m, v in prof.items()]).to_csv(TAB / "base_cost.csv", index=False)

    write_paper(board, tests)
    piv = board.pivot_table(index="method", columns=["dataset", "cond"], values="f1_mean").reindex(MAIN + [BASE])
    print(piv.round(4).to_string())
    print(tests[["dataset", "cond", "A", "diff", "p_w", "p_w_holm_family"]].round(4).to_string() if len(tests) else "no tests yet")


LABEL = {("ciciot", "iid"): "CIC IID", ("ciciot", "dir1.0"): "CIC Dir(1.0)", ("ciciot", "dir0.3"): "CIC Dir(0.3)",
         ("ciciot", "dir0.1"): "CIC Dir(0.1)", ("nbaiot", "device"): "NBa Device", ("nbaiot", "iid"): "NBa IID"}


def write_paper(board, tests):
    """paper/numbers_base.tex (macros, read via \\nv{base:...}) and paper/tables/table9_base.tex. No number is typed by hand."""
    paper = ROOT / "paper"
    g = lambda m, ds, c, col="f1_mean": float(board[(board.method == m) & (board.dataset == ds) & (board.cond == c)][col].iloc[0])
    mac = {}
    for (ds, c) in LABEL:
        mac[f"base:f1:{ds}:{c}"] = f"{g(BASE, ds, c):.4f}"
    b = board[board.method == BASE]
    mac["base:f1min"], mac["base:f1max"] = f"{b.f1_mean.min():.4f}", f"{b.f1_mean.max():.4f}"
    mac["base:nfail"], mac["base:nruns"] = str(int(b.n_fail.sum())), str(int(b.n_runs.sum()))
    ours = tests[tests.A == "fedstat"].set_index(["dataset", "cond"])["diff"].abs() * 100
    mac["base:gapcic:min"], mac["base:gapcic:max"] = f"{ours.loc['ciciot'].min():.2f}", f"{ours.loc['ciciot'].max():.2f}"
    mac["base:gapnba:min"], mac["base:gapnba:max"] = f"{ours.loc['nbaiot'].min():.1f}", f"{ours.loc['nbaiot'].max():.1f}"
    mac["base:ntests"], mac["base:holm"] = str(len(tests)), f"{tests.p_w_holm_family.max():.3f}"
    nk = {ds: [json.load(open(ROOT / "results" / "base" / f"coa_{ds}_s{s}.json"))["n_keep"] for s in range(10)] for ds in CONDS}
    for ds, short in (("ciciot", "cic"), ("nbaiot", "nba")):
        mac[f"base:nfeat{short}:min"], mac[f"base:nfeat{short}:max"] = str(min(nk[ds])), str(max(nk[ds]))
        mac[f"base:nfeat{short}:mean"] = f"{np.mean(nk[ds]):.1f}"
    diag = dict(json.load(open(ROOT / "results" / "base" / "single_feature_diag.json"))["ciciot"]["top10"])
    for f in ("Number", "Weight", "IAT"):
        mac[f"base:single:{f}"] = f"{diag[f]:.4f}"
    cost = json.load(open(ROOT / "results" / "base" / "cost_profile_base.json"))
    mac["base:params"] = f"{int(cost['n_params']):,}"
    mac["base:train"] = f"{cost['train_sec']:.1f}"
    mac["base:infer"] = f"{cost['infer_ms_per_1k']:.2f}"
    mac["base:trafficKB"] = f"{cost['comm_model_bytes'] / 1024:,.0f}"
    mac["base:coasec"] = f"{cost['coa_sec']:.1f}"
    mac["base:paramratio"] = f"{round(cost['n_params'] / 17):,}"
    with open(paper / "numbers_base.tex", "w", encoding="utf-8") as f:
        f.write("% AUTO-GENERATED by code/plots/base_compare.py - do not edit\n")
        for k, v in mac.items():
            f.write("\\expandafter\\def\\csname num:%s\\endcsname{%s}\n" % (k, v.replace(",", "{,}")))
    rows = []
    for (ds, c), lab in LABEL.items():
        rows.append(f"{lab} & \\textbf{{{g(BASE, ds, c):.4f}}} & {g('fedstat', ds, c):.4f} & {g('fedstat_mlp', ds, c):.4f} & {g('senior', ds, c):.4f} \\\\")
    tex = ("\\begin{table}[t]\n\\centering\n"
           "\\caption{Base paper~\\cite{karunamurthy2025} (Chimp optimisation + federated CNN), re-implemented and run on the same corpora, "
           "partitions and tuning budget: test F1, mean over 10 seeds. Base vs each method: paired Wilcoxon, Holm within this added "
           f"family of {len(tests)} tests, all $p_{{\\mathrm{{Holm}}}}\\le$ {tests.p_w_holm_family.max():.3f}.}}\n"
           "\\label{tab:base}\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lcccc}\n\\toprule\n"
           "Regime & Base (COA+CNN) & FedStat-FS & FedAvg-MLP & Senior \\\\\n\\midrule\n" + "\n".join(rows) +
           "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (paper / "tables" / "table9_base.tex").write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    main()
