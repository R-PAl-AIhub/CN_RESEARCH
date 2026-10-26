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

    piv = board.pivot_table(index="method", columns=["dataset", "cond"], values="f1_mean").reindex(MAIN + [BASE])
    print(piv.round(4).to_string())
    print(tests[["dataset", "cond", "A", "diff", "p_w", "p_w_holm_family"]].round(4).to_string() if len(tests) else "no tests yet")


if __name__ == "__main__":
    main()
