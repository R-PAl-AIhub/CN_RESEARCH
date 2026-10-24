"""
Dataset access for the experiments (REAL data only).

CIC-IoT2023 : data/processed/ciciot_pool.parquet (built by code/data/build_ciciot_pool.py from the real 46-feature release)
N-BaIoT     : data/processed/nbaiot_pool.parquet (built by code/data/build_nbaiot_pool.py from the real UCI archive)

Both pools are de-duplicated on their feature vectors BEFORE any split, so an identical flow can never sit in two splits.
A "corpus" is a seeded, stratified sample of the de-duplicated pool with a 60/20/20 train/validation/test split.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from . import partition

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"


# ------------------------------------------------------------------------------------------------ pools (cached)
def _dedup(df: pd.DataFrame, feat, name: str):
    n0 = len(df)
    df = df.drop_duplicates(subset=feat, keep="first").reset_index(drop=True)
    return df, {"pool": name, "rows_before": n0, "rows_after": len(df), "duplicates_removed": n0 - len(df)}


def load_pool(name: str):
    """Returns (X float32, y int8, category-string array, group array, feature names, dedup report)."""
    cache = PROC / f"{name}_dedup.npz"
    rep_f = PROC / f"{name}_dedup_report.json"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        import json
        return z["X"], z["y"], z["cat"], z["group"], list(z["feat"]), json.load(open(rep_f))
    import json
    if name == "ciciot":
        df = pd.read_parquet(PROC / "ciciot_pool.parquet")
        feat = [c for c in df.columns if c not in ("Label", "is_attack")]
        df, rep = _dedup(df, feat, name)
        cat = df["Label"].to_numpy(dtype=object)
        group = np.full(len(df), -1)
    elif name == "nbaiot":
        df = pd.read_parquet(PROC / "nbaiot_pool.parquet")
        feat = [c for c in df.columns if c not in ("device", "attack", "is_attack")]
        df, rep = _dedup(df, feat, name)
        cat = df["attack"].to_numpy(dtype=object)
        group = df["device"].to_numpy()
    else:
        raise ValueError(name)
    X, y = df[feat].to_numpy(np.float32), df["is_attack"].to_numpy(np.int8)
    rep["by_class_after"] = {int(k): int(v) for k, v in pd.Series(y).value_counts().items()}
    np.savez_compressed(cache, X=X, y=y, cat=cat, group=group, feat=np.array(feat, dtype=object))
    json.dump(rep, open(rep_f, "w"), indent=1)
    return X, y, cat, group, feat, rep


# ------------------------------------------------------------------------------------------------ corpus
def _stratified_split(strata: np.ndarray, rng, frac=(0.6, 0.2, 0.2)):
    tr, va, te = [], [], []
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        rng.shuffle(idx)
        a = int(np.ceil(frac[0] * len(idx)))
        b = a + int(np.floor(frac[1] * len(idx)))
        if len(idx) <= 3:                       # tiny cell: keep it in training only
            a, b = len(idx), len(idx)
        tr.append(idx[:a]); va.append(idx[a:b]); te.append(idx[b:])
    return np.concatenate(tr), np.concatenate(va), np.concatenate(te)


def build_corpus(name: str, seed: int, n_per_class: int):
    """Seeded stratified sample: n_per_class benign + n_per_class attack rows (N-BaIoT: per device), 60/20/20 split."""
    X, y, cat, group, feat, rep = load_pool(name)
    rng = np.random.RandomState(seed)
    if name == "ciciot":
        keep = np.concatenate([rng.choice(np.where(y == c)[0], size=min(n_per_class, (y == c).sum()), replace=False)
                               for c in (0, 1)])
    else:  # nbaiot: cap per (device, class)
        parts = []
        for g in np.unique(group):
            for c in (0, 1):
                idx = np.where((group == g) & (y == c))[0]
                parts.append(rng.choice(idx, size=min(n_per_class, len(idx)), replace=False))
        keep = np.concatenate(parts)
    keep = np.sort(keep)
    X, y, cat, group = X[keep], y[keep], cat[keep], group[keep]
    cats = sorted(set(cat.tolist()))
    cat_id = np.array([cats.index(c) for c in cat])
    strata = cat_id * 100 + (np.searchsorted(np.unique(group), group) if name == "nbaiot" else 0)
    tr, va, te = _stratified_split(strata, rng)
    return dict(name=name, X=X, y=y, cat=cat_id, cat_names=cats, group=group, feat=feat,
                idx={"train": tr, "val": va, "test": te}, dedup=rep)


def make_clients(corpus: dict, cond: str, seed: int):
    """Returns {split: [index array per client]} (indices into corpus arrays) for a heterogeneity condition.
       cond: 'iid' | 'dir<alpha>' (Dirichlet label/attack-type skew) | 'device' (natural, one client per device)."""
    rng = np.random.RandomState(10_000 + seed)
    out = {}
    if cond == "device":
        devs = np.unique(corpus["group"][corpus["idx"]["train"]])
        for s, ix in corpus["idx"].items():
            out[s] = [ix[corpus["group"][ix] == d] for d in devs]
        return out
    n_cat = len(corpus["cat_names"])
    K = corpus.get("K", 10)
    P = partition.proportions_iid(n_cat, K) if cond == "iid" else \
        partition.proportions_dirichlet(n_cat, K, float(cond.replace("dir", "")), rng)
    for s, ix in corpus["idx"].items():
        parts = partition.split_by_proportions(corpus["cat"][ix], P, rng)
        out[s] = [ix[p] for p in parts]
    return out
