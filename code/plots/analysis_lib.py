"""Shared loaders / statistics for every table and figure script. No number is typed by hand: everything comes from results/raw/*.jsonl."""
import json
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"

CONDS = {"ciciot": ["iid", "dir1.0", "dir0.3", "dir0.1"], "nbaiot": ["device", "iid"]}
COND_LABEL = {"iid": "IID", "dir1.0": "Dir(1.0)", "dir0.3": "Dir(0.3)", "dir0.1": "Dir(0.1)", "device": "Device"}
DS_LABEL = {"ciciot": "CIC-IoT2023", "nbaiot": "N-BaIoT"}
MAIN = ["central_svm", "central_rf", "senior", "senior_noleak", "local_prep_fedavg", "fedstat", "fedstat_prox", "fedstat_mlp"]
ABL = ["fedstat", "abl_localscale", "abl_nosel", "abl_oneshot", "abl_pearson", "abl_lda"]
METRICS = ["f1", "rec", "prec", "acc", "fpr", "mcc"]


def load_final() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(RAW / "final_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_sens() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(RAW / "sens_s*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def cell(df, ds, cond, method, col="f1") -> np.ndarray:
    """Per-seed values sorted by seed (so that arrays from different methods are paired)."""
    d = df[(df.dataset == ds) & (df.cond == cond) & (df.method == method)].sort_values("seed")
    return d[col].to_numpy(dtype=float)


def mean_std(x):
    x = np.asarray(x, float)
    return (x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0) if len(x) else (np.nan, np.nan)


def paired_tests(a, b):
    """Paired comparison of two per-seed arrays: mean difference (a-b), 95% t-CI, paired t p, Wilcoxon p, Cohen's dz."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    if n < 3:
        return dict(n=n, diff=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_t=np.nan, p_w=np.nan, dz=np.nan)
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    p_t = stats.ttest_rel(a, b).pvalue if np.any(d != 0) else 1.0
    try:
        p_w = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p_w = 1.0
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else (np.inf if d.mean() != 0 else 0.0)
    return dict(n=n, diff=d.mean(), ci_lo=d.mean() - tcrit * se, ci_hi=d.mean() + tcrit * se, p_t=float(p_t), p_w=float(p_w), dz=float(dz))


PAIRS = [  # (A, B, purpose)
    ("senior", "senior_noleak", "RQ1 leakage: pooled pre-split vs train-only preprocessing"),
    ("fedstat", "local_prep_fedavg", "RQ2 federated vs naive local preprocessing"),
    ("fedstat", "abl_localscale", "RQ2/A1 global vs local scaling"),
    ("fedstat", "abl_nosel", "A2 with vs without selection"),
    ("fedstat", "abl_oneshot", "A3 iterative vs one-shot"),
    ("fedstat", "central_svm", "RQ3 parity with centralised SVM"),
    ("fedstat", "senior", "vs senior pipeline as released"),
    ("fedstat", "fedstat_prox", "vs FedProx"),
    ("fedstat", "fedstat_mlp", "vs federated MLP"),
]


def significance_table(df) -> pd.DataFrame:
    rows = []
    for ds, conds in CONDS.items():
        for c in conds:
            for A, B, why in PAIRS:
                a, b = cell(df, ds, c, A), cell(df, ds, c, B)
                if len(a) == 0 or len(a) != len(b):
                    continue
                r = paired_tests(a, b)
                rows.append(dict(dataset=ds, cond=c, A=A, B=B, purpose=why, **r))
    t = pd.DataFrame(rows)
    if len(t):
        t["p_w_holm"] = multipletests(t["p_w"].fillna(1.0), method="holm")[1]
        t["p_t_holm"] = multipletests(t["p_t"].fillna(1.0), method="holm")[1]
        # post-hoc sensitivity: Holm within each (A, B) comparison across its client regimes (the primary correction stays global)
        t["p_w_holm_pair"] = t.groupby(["A", "B"])["p_w"].transform(lambda s: multipletests(s.fillna(1.0), method="holm")[1])
    return t


def fmt_p(p):
    return "<0.001" if p < 1e-3 else f"{p:.3f}"
