"""
Phase 1: verify that the TF ports reproduce the senior (scikit-learn) results on the senior's own preprocessed arrays
(code/senior_repro/outputs/data/*.npy, produced by the unchanged senior code on real CIC-IoT2023 flows).
  (a) TF linear SVM (hinge and squared hinge) centralised  vs sklearn LinearSVC
  (b) TF federated linear SVM (5 workers, IID): one-shot average (what the senior code does) and iterative FedAvg
  (c) Keras MLP 17-100-50-1                                 vs sklearn MLPClassifier
Usage: python code/phase1_tf_port_check.py [--mlp]   (--mlp re-trains the Keras MLP, ~2 min; otherwise cached)
"""
import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fedstat import tfcpu
from fedstat.models import LinearSVM, SVMTrainer, make_mlp
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
OUTJ = ROOT / "results" / "phase1_tf_port_check.json"
D = ROOT / "code" / "senior_repro" / "outputs" / "data"
Xtr, Xte = np.load(D / "X_train.npy").astype(np.float32), np.load(D / "X_test.npy").astype(np.float32)
ytr, yte = np.load(D / "y_train.npy").astype(np.int8), np.load(D / "y_test.npy").astype(np.int8)
tfcpu.set_seed(42)
LAM = 1.0 / (1.0 * len(Xtr))   # lambda = 1/(C n) maps sklearn LinearSVC(C=1) onto the mean-loss objective
d = Xtr.shape[1]


def m(y, p):
    return dict(acc=accuracy_score(y, p), prec=precision_score(y, p), rec=recall_score(y, p), f1=f1_score(y, p))


prev = json.load(open(OUTJ)) if OUTJ.exists() else {}
out = {}

# (a) centralised TF SVMs
for name, sq in (("tf_svm_central_hinge", False), ("tf_svm_central_sqhinge", True)):
    t = time.time()
    mod = LinearSVM(d)
    tr = SVMTrainer(mod)
    tr.fit(Xtr, ytr, epochs=20, batch_size=1024, lr=0.1, lr_end=0.002, lam=LAM, squared=sq, rng=np.random.RandomState(42))
    out[name] = {**m(yte, tr.predict(Xte)), "sec": time.time() - t}

# (b) federated, 5 IID workers
idx = np.random.RandomState(42).permutation(len(Xtr))
parts = np.array_split(idx, 5)


def fed(rounds, local_epochs, warm, sq, lr0=0.1, decay=0.0):
    g = np.zeros(d + 1)
    t0 = time.time()
    for r in range(rounds):
        ps, ns = [], []
        lr = lr0 / (1.0 + decay * r)
        for k, ix in enumerate(parts):
            mk = LinearSVM(d)
            trk = SVMTrainer(mk)
            if warm:
                mk.set_params(g)
            trk.fit(Xtr[ix], ytr[ix], epochs=local_epochs, batch_size=1024, lr=lr, lam=LAM, squared=sq,
                    lr_end=(0.002 if not warm else None), rng=np.random.RandomState(100 * r + k))
            ps.append(mk.get_params())
            ns.append(len(ix))
        g = np.average(np.stack(ps), axis=0, weights=np.array(ns, float))
    mg = LinearSVM(d)
    mg.set_params(g)
    return {**m(yte, SVMTrainer(mg).predict(Xte)), "sec": time.time() - t0}


# what the senior code does: 3 "rounds" that each retrain from scratch (cold start) -> identical to one-shot averaging
out["tf_fed5_oneshot_avg_sqhinge"] = fed(rounds=1, local_epochs=20, warm=False, sq=True)
# what Algorithm 2 of the senior paper describes: iterative FedAvg with warm start from the global model
out["tf_fed5_fedavg_warm_sqhinge"] = fed(rounds=20, local_epochs=2, warm=True, sq=True, lr0=0.1, decay=0.3)
out["tf_fed5_fedavg_warm_hinge"] = fed(rounds=20, local_epochs=2, warm=True, sq=False, lr0=0.1, decay=0.3)

# (c) Keras MLP (senior ANN: 17-100-50-1, batch 32, 10 epochs)
if "--mlp" in sys.argv or "keras_mlp_central" not in prev:
    t = time.time()
    mlp = make_mlp(d)
    mlp.fit(Xtr, ytr, batch_size=32, epochs=10, verbose=0)
    out["keras_mlp_central"] = {**m(yte, (mlp.predict(Xte, batch_size=8192, verbose=0)[:, 0] > 0.5).astype(int)),
                                "sec": time.time() - t}
else:
    out["keras_mlp_central"] = prev["keras_mlp_central"]

# sklearn reference numbers from the unchanged senior run
ref = json.load(open(ROOT / "code" / "senior_repro" / "outputs" / "results" / "experiment_results.json"))
keys = ("accuracy", "precision", "recall", "f1_score")
out["senior_sklearn_reference"] = {
    "central_svm": {k: ref["centralized_svm"][k] for k in keys},
    "fed5": {k: ref["federated_svm"]["5"][k] for k in keys},
    "ann": {k: ref["comparative_models"]["ANN"][k] for k in keys}}
OUTJ.write_text(json.dumps(out, indent=1, default=float))
for k, v in out.items():
    print(k, json.dumps(v, default=lambda x: round(float(x), 4)))
