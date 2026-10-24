"""
Computational-cost profile (CPU only). Each method runs COLD in a FRESH subprocess (3 repeats, medians reported) on CIC-IoT2023 (seed 0, IID clients, validation
split, tuned hyper-parameters); a sampling thread records the incremental peak resident memory of the whole run.
Output: results/cost_profile.json   (used by code/plots/make_tables.py -> Table VII)
Usage : python code/run_cost.py            (driver)     |   python code/run_cost.py --child <method>   (internal)
"""
import os
import sys
import json
import time
import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
METHODS = ["central_svm", "central_rf", "senior", "fedstat", "fedstat_prox", "fedstat_mlp"]

if "--child" in sys.argv:
    method = sys.argv[sys.argv.index("--child") + 1]
    import psutil
    from fedstat import tfcpu, data, methods
    tfcpu.set_seed(0)
    corp = data.build_corpus("ciciot", seed=0, n_per_class=100_000)
    corp["K"] = 10
    cl = data.make_clients(corp, "iid", seed=0)
    hp = json.load(open(ROOT / "results" / "tuning" / "ciciot_iid.json"))[method]["best_hp"]
    proc = psutil.Process()
    base = proc.memory_info().rss
    peak = [base]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.01)
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    r = methods.run_method(method, corp, cl, hp, seed=0, eval_split="val")
    stop.set()
    th.join()
    print("RESULT " + json.dumps(dict(method=method, train_sec=r["train_sec"], prep_sec=r["prep_sec"], total_sec=r["total_sec"],
                                      infer_ms_per_1k=r["infer_ms_per_1k"], n_params=r["n_params"], n_feat=r["n_feat"],
                                      comm_model_bytes=r["comm_model_bytes"], comm_prep_bytes=r["comm_prep_bytes"],
                                      peak_incremental_mb=(peak[0] - base) / 2 ** 20, rss_mb=peak[0] / 2 ** 20, f1_val=r["f1"])))
    sys.exit(0)

import statistics
REPEATS = 3
out = {}
for m in METHODS:
    reps = []
    for _ in range(REPEATS):                              # cold run in a fresh process, repeated; medians are reported
        p = subprocess.run([sys.executable, __file__, "--child", m], capture_output=True, text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(m, "FAILED", p.stderr[-400:])
            continue
        reps.append(json.loads(line[0][7:]))
    if not reps:
        continue
    out[m] = {k: (statistics.median([r[k] for r in reps]) if isinstance(reps[0][k], (int, float)) else reps[0][k]) for k in reps[0]}
    out[m]["repeats"] = len(reps)
    print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[m].items()}, flush=True)
(ROOT / "results" / "cost_profile.json").write_text(json.dumps(out, indent=1))
