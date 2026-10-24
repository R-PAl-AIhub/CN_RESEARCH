"""
Experiment driver.   python code/run_experiments.py {tune|final|sens|all} [--workers N] [--datasets ciciot,nbaiot] [--seeds 10]

tune  : random search (6 trials, identical seed and search space for every method) on the VALIDATION split, seed 0 corpus.
final : tuned hyper-parameters, seeds 0..S-1, evaluated ONCE on the TEST split.  (ablations inherit the tuned `fedstat` hp)
sens  : hyper-parameter sensitivity (tau, LDA top-k, rounds) on the VALIDATION split (test is never touched here).
Outputs: results/tuning/*.json, results/raw/*.jsonl (one JSON object per run).  All tasks are resumable (skip if output exists).
"""
import os
import sys
import json
import time
from pathlib import Path
from multiprocessing import get_context

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
RES = ROOT / "results"
(RES / "tuning").mkdir(parents=True, exist_ok=True)
(RES / "raw").mkdir(parents=True, exist_ok=True)

CONDS = {"ciciot": ["iid", "dir1.0", "dir0.3", "dir0.1"], "nbaiot": ["device", "iid"]}
NPC = {"ciciot": 100_000, "nbaiot": 8_000}
KCLIENTS = {"ciciot": 10, "nbaiot": 9}
N_TRIALS = 6
SVM_FAMILY = ("central_svm", "fedstat", "fedstat_prox", "local_prep_fedavg", "abl_nosel", "abl_pearson", "abl_lda")
TUNE_SEEDS = (0, 1, 2)          # hyper-parameters are chosen on the mean VALIDATION F1 over these corpora (robustness to one lucky seed)


def _np(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def sample_hp(method, trial):
    """Same random-search stream for all methods (seed 1234): trial t draws the same (lr, C, loss) everywhere."""
    import numpy as np
    rng = np.random.RandomState(1234 + trial)
    lr = float(10 ** rng.uniform(np.log10(0.005), np.log10(0.3)))
    C = float(10 ** rng.uniform(np.log10(0.3), 2.0))
    sq = bool(rng.randint(2))
    mu = float(10 ** rng.uniform(-3, 0))
    l2 = float(10 ** rng.uniform(-5, -2))
    if method in SVM_FAMILY:
        hp = dict(lr=lr, C=C, squared=sq, rounds=20, local_epochs=2, epochs=20)
        if method == "fedstat_prox":
            hp["mu_p"] = mu
        return hp
    if method in ("senior", "senior_noleak"):
        return dict(C=C, squared=sq)
    if method == "fedstat_mlp":
        return dict(lr=lr, l2=l2, rounds=20, mlp_epochs=1)
    if method == "central_rf":
        combos = [(50, 10), (50, 20), (50, 30), (100, 10), (100, 20), (100, 30)]      # exhaustive grid of N_TRIALS points
        n, d = combos[trial]
        return dict(n_estimators=n, max_depth=d)
    raise ValueError(method)


def _init_worker():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)


def tune_task(args):
    ds, cond = args
    out = RES / "tuning" / f"{ds}_{cond}.json"
    if out.exists():
        return f"skip {out.name}"
    from fedstat import tfcpu, data, methods
    import numpy as np
    tfcpu.set_seed(0)
    setups = []
    for sd in TUNE_SEEDS:
        corp = data.build_corpus(ds, seed=sd, n_per_class=NPC[ds])
        corp["K"] = KCLIENTS[ds]
        setups.append((sd, corp, data.make_clients(corp, cond, seed=sd)))
    res = {}
    for m in methods.TUNED + methods.TUNED_EXTRA:
        trials = []
        for t in range(N_TRIALS):
            hp = sample_hp(m, t)
            f1s, secs = [], 0.0
            for sd, corp, cl in setups:
                r = methods.run_method(m, corp, cl, hp, seed=sd, eval_split="val")
                f1s.append(r["f1"]); secs += r["total_sec"]
            trials.append(dict(hp=hp, val_f1=float(np.mean(f1s)), val_f1_per_seed=f1s, val_f1_min=float(np.min(f1s)), sec=secs))
        best = max(trials, key=lambda z: z["val_f1"])
        res[m] = dict(best_hp=best["hp"], best_val_f1=best["val_f1"], trials=trials)
        print(f"[tune {ds}/{cond}] {m}: best mean val F1 {best['val_f1']:.4f} (min over seeds {best['val_f1_min']:.4f}) hp={best['hp']}", flush=True)
    json.dump(res, open(out, "w"), indent=1, default=_np)
    return f"done {out.name}"


def final_task(args):
    ds, seed, want_conds = args
    out = RES / "raw" / f"final_{ds}_s{seed}.jsonl"
    if out.exists():
        return f"skip {out.name}"
    from fedstat import tfcpu, data, methods
    tfcpu.set_seed(seed)
    corp = data.build_corpus(ds, seed=seed, n_per_class=NPC[ds])
    corp["K"] = KCLIENTS[ds]
    lines = []
    for cond in CONDS[ds]:
        tuned = json.load(open(RES / "tuning" / f"{ds}_{cond}.json"))
        cl = data.make_clients(corp, cond, seed=seed)
        for m in methods.ALL_METHODS:
            hp = tuned[m]["best_hp"] if m in tuned else tuned["fedstat"]["best_hp"]      # A1/A3 inherit the tuned FedStat-FS hp
            r = methods.run_method(m, corp, cl, hp, seed=seed, eval_split="test", want_curves=(m == "fedstat" and seed == 0))
            r.update(dataset=ds, cond=cond, seed=seed, hp=hp, n_train=int(sum(len(a) for a in cl["train"])),
                     client_train_sizes=[int(len(a)) for a in cl["train"]])
            lines.append(r)
            print(f"[final {ds}/{cond}/s{seed}] {m:18s} test F1={r['f1']:.4f}", flush=True)
    with open(out, "w") as f:
        for r in lines:
            f.write(json.dumps(r, default=_np) + "\n")
    return f"done {out.name}"


def sens_task(args):
    seed, = args
    out = RES / "raw" / f"sens_s{seed}.jsonl"
    if out.exists():
        return f"skip {out.name}"
    from fedstat import tfcpu, data, methods
    ds, cond = "ciciot", "dir0.3"
    tfcpu.set_seed(seed)
    corp = data.build_corpus(ds, seed=seed, n_per_class=NPC[ds])
    corp["K"] = KCLIENTS[ds]
    tuned = json.load(open(RES / "tuning" / f"{ds}_{cond}.json"))
    cl = data.make_clients(corp, cond, seed=seed)
    lines = []
    grid = [("tau", v) for v in (0.1, 0.2, 0.3, 0.4, 0.5)] + [("lda_top", v) for v in (5, 10, 15, 20, 25, 30)] + \
           [("rounds", v) for v in (1, 2, 5, 10, 20, 40)]
    for m in ("fedstat", "local_prep_fedavg"):
        for name, v in grid:
            hp = dict(tuned[m]["best_hp"])
            hp[name] = v
            r = methods.run_method(m, corp, cl, hp, seed=seed, eval_split="val")
            lines.append(dict(method=m, param=name, value=v, seed=seed, f1=r["f1"], rec=r["rec"], fpr=r["fpr"], mcc=r["mcc"],
                              n_feat=r["n_feat"], acc=r["acc"]))
            print(f"[sens s{seed}] {m} {name}={v}: val F1={r['f1']:.4f} nfeat={r['n_feat']}", flush=True)
    with open(out, "w") as f:
        for r in lines:
            f.write(json.dumps(r, default=_np) + "\n")
    return f"done {out.name}"


def run_pool(fn, tasks, workers):
    ctx = get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        for msg in pool.imap_unordered(fn, tasks):
            print("TASK:", msg, flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    dss = sys.argv[sys.argv.index("--datasets") + 1].split(",") if "--datasets" in sys.argv else ["ciciot", "nbaiot"]
    S = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 10
    t0 = time.time()
    if stage in ("tune", "all"):
        run_pool(tune_task, [(d, c) for d in dss for c in CONDS[d]], workers)
    if stage in ("final", "all"):
        run_pool(final_task, [(d, s, None) for d in dss for s in range(S)], workers)
    if stage in ("sens", "all") and "ciciot" in dss:
        run_pool(sens_task, [(s,) for s in range(min(S, 5))], workers)
    print(f"stage {stage} finished in {(time.time()-t0)/60:.1f} min", flush=True)
