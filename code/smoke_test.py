"""Smoke test: every method on CIC-IoT2023 (1 seed), default hyper-parameters, validation split. Not a result."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fedstat import tfcpu
from fedstat import data, methods

ds = sys.argv[1] if len(sys.argv) > 1 else "ciciot"
conds = sys.argv[2].split(",") if len(sys.argv) > 2 else ["iid", "dir0.1"]
tfcpu.set_seed(0)
t0 = time.time()
corp = data.build_corpus(ds, seed=0, n_per_class=100_000 if ds == "ciciot" else 8_000); corp["K"] = 10 if ds == "ciciot" else 9
print(f"corpus {ds}: X {corp['X'].shape}, splits { {k: len(v) for k, v in corp['idx'].items()} }, "
      f"cats {len(corp['cat_names'])}, dedup {corp['dedup']}, build {time.time()-t0:.1f}s", flush=True)
HP = dict(C=10.0, lr=0.1, squared=True, rounds=20, local_epochs=2, mu_p=0.01, n_estimators=50, max_depth=20, l2=1e-3)
for cond in conds:
    cl = data.make_clients(corp, cond, seed=0)
    sizes = [len(a) for a in cl["train"]]
    print(f"\n== {cond}: K={len(sizes)} train sizes {sizes}", flush=True)
    for m in methods.ALL_METHODS:
        hp = dict(HP)
        if m == "fedstat_mlp":
            hp["lr"] = 0.05
        r = methods.run_method(m, corp, cl, hp, seed=0, eval_split="val")
        print(f"{m:18s} F1={r['f1']:.4f} rec={r['rec']:.4f} fpr={r['fpr']:.4f} nfeat={r['n_feat']:3d} "
              f"prep={r['prep_sec']:.1f}s train={r['train_sec']:.1f}s total={r['total_sec']:.1f}s"
              + (f" jac={r['jaccard_vs_pooled']:.2f}" if 'jaccard_vs_pooled' in r else "")
              + (f" skipped={r['n_skipped']}" if r.get('n_skipped') else ""), flush=True)
