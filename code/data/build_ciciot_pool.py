"""
Stream the 46-feature CIC-IoT2023 parquet files once and build a compact *working pool* of REAL flows:
  - every Benign / DDoS-* row is kept with a seeded Bernoulli probability (P_BENIGN, P_DDOS)
  - the exact class counts of the full release are recorded (integrity check against Neto et al. 2023)
No rows are generated or modified; features are only cast to float32.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "ciciot2023_neto" / "random"
OUT = ROOT / "data" / "processed"; OUT.mkdir(parents=True, exist_ok=True)
SEED, P_BENIGN, P_DDOS = 42, 0.35, 0.0125
rng = np.random.RandomState(SEED)

kept, label_counts = [], {}
t0 = time.time()
for fn in ["train-00000-of-00001.parquet", "test-00000-of-00001.parquet"]:
    pf = pq.ParquetFile(RAW / fn)
    for g in range(pf.num_row_groups):
        df = pf.read_row_group(g).to_pandas()
        for k, v in df["Label"].value_counts().items():
            label_counts[k] = label_counts.get(k, 0) + int(v)
        is_b = df["attack_class"].str.lower().eq("benign")
        is_d = df["attack_class"].str.lower().eq("ddos")
        u = rng.random_sample(len(df))
        m = (is_b & (u < P_BENIGN)) | (is_d & (u < P_DDOS))
        if m.any():
            kept.append(df.loc[m].copy())
    print(f"{fn}: done ({time.time()-t0:.0f}s), kept so far {sum(len(k) for k in kept):,}", flush=True)

pool = pd.concat(kept, ignore_index=True)
feat = [c for c in pool.columns if c not in ("Label", "Label_orig", "attack_class", "label")]
assert len(feat) == 46, len(feat)
pool[feat] = pool[feat].astype("float32")
pool["is_attack"] = pool["attack_class"].str.lower().eq("ddos").astype("int8")
pool = pool[feat + ["Label", "is_attack"]]
pool.to_parquet(OUT / "ciciot_pool.parquet", index=False)
summary = {"seed": SEED, "p_benign": P_BENIGN, "p_ddos": P_DDOS, "n_pool": len(pool),
           "pool_by_label": pool["Label"].value_counts().to_dict(),
           "full_release_label_counts": label_counts, "features": feat}
(OUT / "ciciot_pool_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "features"}, indent=1))
