"""
Method registry. Every method: fit on the clients' TRAINING shards only, evaluate on the clients' `eval_split` shards
(validation during tuning, test exactly once for the final report).

  central_svm        pooled train-only preprocessing + centralised TF linear SVM        (privacy-violating upper bound)
  central_rf         pooled train-only preprocessing + random forest (sklearn)          (accuracy ceiling, not federated)
  senior             SENIOR PIPELINE AS RELEASED: preprocessing fitted on ALL rows (train+val+test, i.e. before the split),
                     independent sklearn LinearSVC per client, one size-weighted average
  senior_noleak      same, but preprocessing fitted on the training rows only         (isolates the leakage, RQ1)
  local_prep_fedavg  naive federated: local variance/Pearson/LDA per client + majority vote, per-client scalers, FedAvg
  fedstat            PROPOSED: federated-statistics preprocessing (exact) + iterative FedAvg (TF linear SVM)
  fedstat_prox       proposed preprocessing + FedProx local objective
  fedstat_mlp        proposed preprocessing + FedAvg with a Keras MLP 17-100-50-1
  abl_localscale     ablation A1: federated selection but per-client scalers
  abl_nosel          ablation A2: no feature selection (variance filter only), global scaler   (StatAvg-like)
  abl_oneshot        ablation A3: federated preprocessing but one-shot cold-start averaging
  abl_pearson/abl_lda ablation A4: Pearson-only / LDA-only instead of the intersection
  base_coa_cnn       BASE PAPER (Karunamurthy et al. 2025), re-implemented: Chimp-optimisation wrapper selection on pooled
                     training rows + pooled scaler + FedAvg of a 1-D CNN (run by code/run_base.py, not part of ALL_METHODS)
"""
import time
import numpy as np
from . import prep as P
from . import fl
from .metrics import scores, jaccard

TUNED = ["central_svm", "central_rf", "senior", "senior_noleak", "local_prep_fedavg", "fedstat", "fedstat_prox", "fedstat_mlp"]
TUNED_EXTRA = ["abl_nosel", "abl_pearson", "abl_lda"]          # ablations whose feature count differs from FedStat-FS: tuned on their own
ABLATIONS = ["abl_localscale", "abl_nosel", "abl_oneshot", "abl_pearson", "abl_lda"]     # A1/A3 inherit the tuned `fedstat` hp
ALL_METHODS = TUNED + ABLATIONS

TAU_DEFAULT, VAR_THR, LDA_TOP_DEFAULT = 0.2, 0.01, 20      # senior settings, shared by every method


def _shards(corpus, idx_list):
    X, y = corpus["X"], corpus["y"]
    return [X[i] for i in idx_list], [y[i] for i in idx_list]


def _per_category(corpus, ev_idx, pred_list):
    """Recall per attack category and false-positive rate on benign, over the evaluation shards."""
    cat = np.concatenate([corpus["cat"][i] for i in ev_idx])
    y = np.concatenate([corpus["y"][i] for i in ev_idx])
    p = np.concatenate(pred_list)
    out = {}
    for c, name in enumerate(corpus["cat_names"]):
        m = cat == c
        if m.sum() == 0:
            continue
        out[name] = dict(n=int(m.sum()), err=float((p[m] != y[m]).mean()), pos_rate=float(p[m].mean()))
    return out


def run_method(method, corpus, clients, hp, seed, eval_split="val", want_curves=False):
    t_all = time.time()
    tr_idx, ev_idx = clients["train"], clients[eval_split]
    Xtr, ytr = _shards(corpus, tr_idx)
    Xev, yev = _shards(corpus, ev_idx)
    n_train = int(sum(len(a) for a in Xtr))
    lam = 1.0 / (hp.get("C", 1.0) * n_train)
    rounds, E = hp.get("rounds", 20), hp.get("local_epochs", 2)
    info = {"method": method}
    TAU, LDA_TOP = hp.get("tau", TAU_DEFAULT), hp.get("lda_top", LDA_TOP_DEFAULT)      # sensitivity study overrides these
    Xcat, ycat = np.concatenate(Xtr), np.concatenate(ytr)

    # ------------------------------------------------------------------ preprocessing (differs per method)
    t0 = time.time()
    local_scalers = None
    comm_prep = 0
    if method in ("central_svm", "central_rf", "senior_noleak"):
        pre = P.fit_pooled(Xcat, ycat, TAU, VAR_THR, LDA_TOP)
    elif method == "senior":                       # LEAK: fitted on train + val + test rows together
        allX = np.concatenate([corpus["X"][corpus["idx"][s]] for s in ("train", "val", "test")])
        ally = np.concatenate([corpus["y"][corpus["idx"][s]] for s in ("train", "val", "test")])
        pre = P.fit_pooled(allX, ally, TAU, VAR_THR, LDA_TOP)
    elif method == "base_coa_cnn":                 # base paper: COA subset (pooled, train-only, chosen by run_base.py) + pooled scaler
        keep = np.asarray(hp["coa_keep"], dtype=int)
        Z = Xcat[:, keep].astype(np.float64)
        pre = P.Prep(keep, Z.mean(0), P._safe_std(Z.var(0)))
    elif method == "local_prep_fedavg":
        pre = P.fit_local_vote(list(zip(Xtr, ytr)), TAU, VAR_THR, LDA_TOP)
        comm_prep = len(Xtr) * 8 * Xcat.shape[1]                         # one vote vector per client (bytes, generous)
    else:
        mode = {"abl_nosel": "none", "abl_pearson": "pearson", "abl_lda": "lda"}.get(method, "intersection")
        pre = P.fit_federated(list(zip(Xtr, ytr)), TAU, VAR_THR, LDA_TOP, mode)
        comm_prep = len(Xtr) * P.bytes_uploaded(Xcat.shape[1])
        if method == "abl_localscale":
            local_scalers = []
            for X_k in Xtr:
                Z = X_k[:, pre.keep].astype(np.float64)
                local_scalers.append((Z.mean(0), P._safe_std(Z.var(0))))
    info["prep_sec"] = time.time() - t0
    info["n_feat"] = int(len(pre.keep)); info["keep"] = [int(i) for i in pre.keep]
    info["comm_prep_bytes"] = int(comm_prep)
    if method == "fedstat":                        # RQ1 fidelity: compare with the pooled train-only selection
        ref = P.fit_pooled(Xcat, ycat, TAU, VAR_THR, LDA_TOP)
        info["jaccard_vs_pooled"] = jaccard(pre.keep, ref.keep)
        info["max_abs_mean_diff"] = float(np.abs(pre.mean - ref.mean).max()) if len(pre.keep) == len(ref.keep) and (pre.keep == ref.keep).all() else None

    def tf_client(X, k, split_shards=False):
        if method == "local_prep_fedavg":
            return P.local_scaler_transform(pre, X, k)
        if local_scalers is not None:
            m, s = local_scalers[k]
            return ((X[:, pre.keep].astype(np.float64) - m) / s).astype(np.float32)
        return pre.transform(X)

    Ztr = [tf_client(X, k) for k, X in enumerate(Xtr)]
    Zev = [tf_client(X, k) for k, X in enumerate(Xev)]

    # ------------------------------------------------------------------ training
    t0 = time.time()
    if method == "central_svm":
        theta, ex = fl.central_svm(np.concatenate(Ztr), ycat, epochs=hp.get("epochs", 20), lr=hp["lr"], lam=lam,
                                   squared=hp["squared"], seed=seed)
        predict = lambda Z: fl.svm_predict(theta, Z); n_params = len(theta)
    elif method == "central_rf":
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"], n_jobs=4, random_state=seed)
        rf.fit(np.concatenate(Ztr), ycat)
        ex = dict(sec=time.time() - t0); predict = rf.predict
        n_params = int(sum(t.tree_.node_count for t in rf.estimators_))
    elif method in ("senior", "senior_noleak"):
        theta, ex = fl.senior_sklearn(Ztr, ytr, C=hp["C"], loss="squared_hinge" if hp["squared"] else "hinge", seed=seed)
        predict = lambda Z: fl.svm_predict(theta, Z); n_params = len(theta)
    elif method == "abl_oneshot":
        theta, ex = fl.oneshot_svm(Ztr, ytr, epochs=hp.get("epochs", 20), lr=hp["lr"], lam=lam, squared=hp["squared"], seed=seed)
        predict = lambda Z: fl.svm_predict(theta, Z); n_params = len(theta)
    elif method == "base_coa_cnn":
        model, ex = fl.fedavg_cnn(Ztr, ytr, rounds=rounds, local_epochs=hp.get("cnn_epochs", 1), lr=hp["lr"],
                                  batch_size=hp.get("batch_size", 1000), seed=seed)
        predict = lambda Z: fl.cnn_predict(model, Z); n_params = ex["n_params"]
    elif method == "fedstat_mlp":
        model, ex = fl.fedavg_mlp(Ztr, ytr, rounds=rounds, local_epochs=hp.get("mlp_epochs", 1), lr=hp["lr"], l2=hp.get("l2", 1e-3), seed=seed)
        predict = lambda Z: fl.mlp_predict(model, Z); n_params = ex["n_params"]
    else:                                              # all FedAvg/FedProx linear variants
        val_fn = None
        if want_curves and local_scalers is None and method != "local_prep_fedavg":
            Zv, yv = np.concatenate(Zev), np.concatenate(yev)
            def val_fn(g):
                f = Zv.astype(np.float64) @ g[:-1] + g[-1]
                pm = 2.0 * yv - 1.0
                return {"val_loss": float(np.mean(np.maximum(0, 1 - pm * f))), "val_acc": float(((f > 0) == (yv == 1)).mean())}
        theta, ex = fl.fedavg_svm(Ztr, ytr, rounds=rounds, local_epochs=E, lr=hp["lr"], lam=lam,
                                  mu_p=hp.get("mu_p", 0.0) if method == "fedstat_prox" else 0.0,
                                  squared=hp["squared"], seed=seed, val_fn=val_fn)
        predict = lambda Z: fl.svm_predict(theta, Z); n_params = len(theta)
        info["curves"] = ex["hist"]
    info["train_sec"] = ex["sec"]; info["comm_model_bytes"] = int(ex.get("comm_bytes", 0)); info["n_params"] = int(n_params)
    for k in ("n_used", "n_skipped"):
        if k in ex:
            info[k] = ex[k]

    # ------------------------------------------------------------------ evaluation
    t0 = time.perf_counter()
    preds = [predict(Z) if len(Z) else np.zeros(0, np.int8) for Z in Zev]
    info["infer_ms_per_1k"] = (time.perf_counter() - t0) / max(1, sum(len(z) for z in Zev)) * 1e6     # ms per 1000 flows
    yv_all, pv_all = np.concatenate(yev), np.concatenate(preds)
    info.update(scores(yv_all, pv_all))
    info["per_cat"] = _per_category(corpus, ev_idx, preds)
    info["total_sec"] = time.time() - t_all
    return info
