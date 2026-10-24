"""
Diagnostic for the N-BaIoT selection failures: is the Pearson-LDA *intersection* subset linearly insufficient?
Exact solvers (sklearn LinearSVC dual=False, LogisticRegression) are fitted on the pooled TRAINING split only and scored on
the test split of the same corpus (no tuning on it; it only characterises a failure already visible on the training split).
Seeds 5, 6 and 9 fail with the intersection subset in the main experiment; seed 0 does not.
Output: results/diag_selection_failure.json
"""
import sys
import json
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
warnings.filterwarnings("ignore")
from fedstat import tfcpu, data, prep as P
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


def f1(y, p):
    tp = ((p == 1) & (y == 1)).sum()
    return float(2 * tp / (p.sum() + y.sum() + 1e-12))


out = {}
for seed in (0, 5, 6, 9):
    corp = data.build_corpus("nbaiot", seed=seed, n_per_class=8000)
    corp["K"] = 9
    cl = data.make_clients(corp, "iid", seed=seed)
    X, y = corp["X"], corp["y"]
    tr = np.concatenate([X[i] for i in cl["train"]]); ytr = np.concatenate([y[i] for i in cl["train"]])
    te = np.concatenate([X[i] for i in cl["test"]]); yte = np.concatenate([y[i] for i in cl["test"]])
    for mode in ("intersection", "pearson", "none"):
        pre = P.fit_pooled(tr, ytr, mode=mode)
        Ztr, Zte = pre.transform(tr), pre.transform(te)
        svm = LinearSVC(C=1.0, dual=False, max_iter=5000).fit(Ztr, ytr)
        lr = LogisticRegression(C=10, max_iter=2000).fit(Ztr, ytr)
        out[f"{seed}:{mode}"] = dict(seed=seed, mode=mode, n_feat=int(len(pre.keep)), svm_train_f1=f1(ytr, svm.predict(Ztr)),
                                     svm_test_f1=f1(yte, svm.predict(Zte)), logreg_test_f1=f1(yte, lr.predict(Zte)))
        print(out[f"{seed}:{mode}"], flush=True)
(ROOT / "results" / "diag_selection_failure.json").write_text(json.dumps(out, indent=1))
