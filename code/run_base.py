"""
Base-paper comparison (Karunamurthy et al., Sci. Rep. 2025): Chimp-optimisation feature selection + federated CNN,
re-implemented and run under exactly the same protocol as code/run_experiments.py (same corpora, same partitions,
same 6-trial random-search stream on the validation split of seeds 0-2, test split used once for seeds 0-9).

  python code/run_base.py smoke            one validation run per dataset (timing)
  python code/run_base.py coa   [--workers N]   COA subset per (dataset, seed), pooled TRAINING rows only
  python code/run_base.py tune  [--workers N]
  python code/run_base.py final [--workers N]
Outputs: results/base/coa_*.json, results/tuning/base_*.json, results/raw/base_*.jsonl (all resumable).
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import CONDS, NPC, KCLIENTS, N_TRIALS, TUNE_SEEDS, RES, _np, run_pool   # noqa: E402

METHOD = "base_coa_cnn"
(RES / "base").mkdir(parents=True, exist_ok=True)


def sample_hp(trial):
    """Same random stream as every other method (seed 1234 + trial): the first draw is mapped from the shared
    learning-rate range [0.005, 0.3] onto the Adam range [1e-4, 1e-2] on the same log scale."""
    import numpy as np
    rng = np.random.RandomState(1234 + trial)
    lo, hi = np.log10(0.005), np.log10(0.3)
    u = (rng.uniform(lo, hi) - lo) / (hi - lo)
    return dict(lr=float(10 ** (-4 + 2 * u)), rounds=20, cnn_epochs=1, batch_size=1000)


def _setup(ds, seed, cond):
    from fedstat import data
    import numpy as np
    corp = data.build_corpus(ds, seed=seed, n_per_class=NPC[ds])
    corp["K"] = KCLIENTS[ds]
    cl = data.make_clients(corp, cond, seed=seed)
    union = np.sort(np.concatenate(cl["train"]))
    assert np.array_equal(union, np.sort(corp["idx"]["train"])), "client train shards must equal the pooled training rows"
    return corp, cl


def _keep(ds, seed):
    return json.load(open(RES / "base" / f"coa_{ds}_s{seed}.json"))["keep"]


def coa_task(args):
    ds, seed = args
    out = RES / "base" / f"coa_{ds}_s{seed}.json"
    if out.exists():
        return f"skip {out.name}"
    from fedstat import tfcpu, data, coa
    tfcpu.set_seed(seed)
    corp = data.build_corpus(ds, seed=seed, n_per_class=NPC[ds])
    tr = corp["idx"]["train"]
    keep, info = coa.select(corp["X"][tr], corp["y"][tr], seed=seed)
    info.update(keep=[int(i) for i in keep], n_keep=int(len(keep)), features=[corp["feat"][i] for i in keep])
    json.dump(info, open(out, "w"), indent=1, default=_np)
    return f"done {out.name}: {len(keep)} of {info['n_usable']} features, holdout acc {info['holdout_acc']:.4f}, {info['sec']:.0f}s"


def tune_task(args):
    ds, cond = args
    out = RES / "tuning" / f"base_{ds}_{cond}.json"
    if out.exists():
        return f"skip {out.name}"
    from fedstat import tfcpu, methods
    import numpy as np
    tfcpu.set_seed(0)
    setups = [(sd, *_setup(ds, sd, cond), _keep(ds, sd)) for sd in TUNE_SEEDS]
    trials = []
    for t in range(N_TRIALS):
        hp = sample_hp(t)
        f1s, secs = [], 0.0
        for sd, corp, cl, keep in setups:
            r = methods.run_method(METHOD, corp, cl, dict(hp, coa_keep=keep), seed=sd, eval_split="val")
            f1s.append(r["f1"]); secs += r["total_sec"]
        trials.append(dict(hp=hp, val_f1=float(np.mean(f1s)), val_f1_per_seed=f1s, sec=secs))
        print(f"[tune base {ds}/{cond}] trial {t} lr={hp['lr']:.5f} mean val F1={np.mean(f1s):.4f}", flush=True)
    best = max(trials, key=lambda z: z["val_f1"])
    json.dump(dict(best_hp=best["hp"], best_val_f1=best["val_f1"], trials=trials), open(out, "w"), indent=1, default=_np)
    return f"done {out.name}: best lr {best['hp']['lr']:.5f}, val F1 {best['val_f1']:.4f}"


def final_task(args):
    ds, seed = args
    out = RES / "raw" / f"base_{ds}_s{seed}.jsonl"
    if out.exists():
        return f"skip {out.name}"
    from fedstat import tfcpu, methods
    tfcpu.set_seed(seed)
    ref = {}
    for line in open(RES / "raw" / f"final_{ds}_s{seed}.jsonl"):
        r = json.loads(line)
        if r["method"] == "fedstat":
            ref[r["cond"]] = r
    keep = _keep(ds, seed)
    lines = []
    for cond in CONDS[ds]:
        corp, cl = _setup(ds, seed, cond)
        sizes = [int(len(a)) for a in cl["train"]]
        n_train = int(sum(sizes))
        assert n_train == ref[cond]["n_train"] and sizes == ref[cond]["client_train_sizes"], \
            f"partition differs from the stored final run ({ds}/{cond}/s{seed})"
        hp = json.load(open(RES / "tuning" / f"base_{ds}_{cond}.json"))["best_hp"]
        r = methods.run_method(METHOD, corp, cl, dict(hp, coa_keep=keep), seed=seed, eval_split="test")
        r.update(dataset=ds, cond=cond, seed=seed, hp=hp, n_train=n_train, client_train_sizes=sizes)
        lines.append(r)
        print(f"[final base {ds}/{cond}/s{seed}] test F1={r['f1']:.4f} nfeat={r['n_feat']} train={r['train_sec']:.0f}s", flush=True)
    with open(out, "w") as f:
        for r in lines:
            f.write(json.dumps(r, default=_np) + "\n")
    return f"done {out.name}"


def smoke():
    from fedstat import tfcpu, coa, methods
    tfcpu.set_seed(0)
    for ds, cond in (("ciciot", "iid"), ("nbaiot", "device")):
        corp, cl = _setup(ds, 0, cond)
        tr = corp["idx"]["train"]
        keep, info = coa.select(corp["X"][tr], corp["y"][tr], seed=0)
        print(f"[smoke {ds}] COA: {len(keep)} of {info['n_usable']} features, holdout acc {info['holdout_acc']:.4f}, "
              f"{info['evals']} evals, {info['sec']:.0f}s", flush=True)
        r = methods.run_method(METHOD, corp, cl, dict(sample_hp(0), coa_keep=keep), seed=0, eval_split="val")
        print(f"[smoke {ds}/{cond}] val F1={r['f1']:.4f} params={r['n_params']} train={r['train_sec']:.0f}s "
              f"infer={r['infer_ms_per_1k']:.2f}ms/1k", flush=True)


def cost_child():
    """Same protocol as code/run_cost.py: CIC-IoT2023, seed 0, IID clients, validation split, tuned hp, cold process."""
    import threading
    import psutil
    from fedstat import tfcpu, methods
    tfcpu.set_seed(0)
    corp, cl = _setup("ciciot", 0, "iid")
    hp = json.load(open(RES / "tuning" / "base_ciciot_iid.json"))["best_hp"]
    proc = psutil.Process(); base = proc.memory_info().rss; peak = [base]; stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss); time.sleep(0.01)
    th = threading.Thread(target=sampler, daemon=True); th.start()
    r = methods.run_method(METHOD, corp, cl, dict(hp, coa_keep=_keep("ciciot", 0)), seed=0, eval_split="val")
    stop.set(); th.join()
    print("RESULT " + json.dumps(dict(train_sec=r["train_sec"], prep_sec=r["prep_sec"], infer_ms_per_1k=r["infer_ms_per_1k"],
                                      n_params=r["n_params"], n_feat=r["n_feat"], comm_model_bytes=r["comm_model_bytes"],
                                      comm_prep_bytes=r["comm_prep_bytes"], peak_incremental_mb=(peak[0] - base) / 2 ** 20, f1_val=r["f1"])))


def cost():
    import os
    import subprocess
    import statistics
    reps = []
    for _ in range(3):
        p = subprocess.run([sys.executable, __file__, "cost-child"], capture_output=True, text=True,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if line:
            reps.append(json.loads(line[0][7:]))
        else:
            print("cost child failed", p.stderr[-400:])
    out = {k: statistics.median(r[k] for r in reps) for k in reps[0]}
    out.update(method=METHOD, repeats=len(reps), coa_sec=json.load(open(RES / "base" / "coa_ciciot_s0.json"))["sec"])
    json.dump(out, open(RES / "base" / "cost_profile_base.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "cost-child":
        cost_child(); sys.exit(0)
    if stage == "cost":
        cost(); sys.exit(0)
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    t0 = time.time()
    if stage == "smoke":
        smoke()
    if stage in ("coa", "all"):
        run_pool(coa_task, [(d, s) for d in CONDS for s in range(10)], workers)
    if stage in ("tune", "all"):
        run_pool(tune_task, [(d, c) for d in CONDS for c in CONDS[d]], workers)
    if stage in ("final", "all"):
        run_pool(final_task, [(d, s) for d in CONDS for s in range(10)], workers)
    print(f"stage {stage} finished in {(time.time()-t0)/60:.1f} min", flush=True)
