"""
Build the N-BaIoT working pool from the REAL UCI archive (data/raw/nbaiot/nbaiot.zip).

Source  : https://archive.ics.uci.edu/dataset/442  (doi:10.24432/C5RC8J, CC BY 4.0)
Paper   : Meidan et al., "N-BaIoT - Network-Based Detection of IoT Botnet Attacks Using Deep Autoencoders",
          IEEE Pervasive Computing 17(3):12-22, 2018.
Task    : benign vs DoS/DDoS-style flood traffic. Kept attack files: BASHLITE (gafgyt) combo/junk/tcp/udp and Mirai
          ack/syn/udp/udpplain. The two `scan` files (reconnaissance, not flooding) are excluded to stay inside the
          DDoS scope of this project (decision D6 in decisions_log.md).
Sampling: exact uniform sample WITHOUT replacement per (device, file) via random keys (one pass, chunked), seed 42.
          No row is generated or modified.
"""
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ZIP = ROOT / "data" / "raw" / "nbaiot" / "nbaiot.zip"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
SEED, Q_BENIGN, Q_ATTACK = 42, 12_000, 3_000
FLOOD = {"gafgyt": {"combo", "junk", "tcp", "udp"}, "mirai": {"ack", "syn", "udp", "udpplain"}}
rng = np.random.RandomState(SEED)


def sample_csv(fh, quota):
    """Uniform sample of `quota` rows from a CSV inside the zip (one pass, chunked, random-key selection)."""
    best, total = None, 0
    if True:
        for chunk in pd.read_csv(fh, chunksize=100_000):
            total += len(chunk)
            chunk = chunk.copy()
            chunk["_k"] = rng.random_sample(len(chunk))
            best = chunk if best is None else pd.concat([best, chunk])
            if len(best) > quota:
                best = best.nsmallest(quota, "_k")
    return best.drop(columns="_k"), total


zf = zipfile.ZipFile(ZIP)
EXT = ROOT / "data" / "raw" / "nbaiot" / "extracted"      # attack CSVs live inside nested .rar files in the UCI archive;
                                                          # extracted with Windows bsdtar (see decisions_log D6)
rows, report = [], []
for dev_dir in sorted(p for p in EXT.iterdir() if p.is_dir()):
    device = dev_dir.name
    jobs = [(zf.open(f"{device}/benign_traffic.csv"), "benign_traffic", 0, Q_BENIGN, f"{device}/benign_traffic.csv")]
    for botnet in ("gafgyt", "mirai"):
        for f in sorted((dev_dir / f"{botnet}_attacks").glob("*.csv")):
            if f.stem in FLOOD[botnet]:                       # flood-type attacks only; scan files are never read
                jobs.append((open(f, "rb"), f"{botnet}.{f.stem}", 1, Q_ATTACK, str(f.relative_to(EXT))))
    for fh, attack, is_attack, quota, label in jobs:
        df, total = sample_csv(fh, quota)
        fh.close()
        df["device"], df["attack"], df["is_attack"] = device, attack, np.int8(is_attack)
        rows.append(df)
        report.append({"file": label, "rows_in_file": total, "rows_kept": len(df)})
        print(f"{device:45s} {attack:16s} {total:>9,} rows -> kept {len(df):,}", flush=True)

pool = pd.concat(rows, ignore_index=True)
feat = [c for c in pool.columns if c not in ("device", "attack", "is_attack")]
assert len(feat) == 115, len(feat)
pool[feat] = pool[feat].astype("float32")
pool.to_parquet(OUT / "nbaiot_pool.parquet", index=False)
summ = {"seed": SEED, "quota_benign": Q_BENIGN, "quota_attack_per_file": Q_ATTACK, "n_pool": len(pool),
        "devices": sorted(pool["device"].unique().tolist()), "by_attack": pool["attack"].value_counts().to_dict(),
        "by_device_class": pool.groupby(["device", "is_attack"]).size().unstack().to_dict(), "files": report}
(OUT / "nbaiot_pool_summary.json").write_text(json.dumps(summ, indent=1, default=int))
print("pool", pool.shape, "NaN cells:", int(np.isnan(pool[feat].to_numpy()).sum()))
