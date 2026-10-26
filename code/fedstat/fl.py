"""
Federated / centralised training loops (TensorFlow for every neural or hinge-loss model; sklearn only for the senior
LinearSVC re-run and the RF ceiling, which are not deep learning).
"""
import time
from typing import List, Optional, Callable
import numpy as np
from . import tfcpu  # noqa: F401
import tensorflow as tf
from .models import LinearSVM, SVMTrainer

_CACHE = {}


def _trainer(d: int):
    """One reusable model+trainer per input width (avoids re-tracing tf.function for every client and round)."""
    if d not in _CACHE:
        m = LinearSVM(d)
        _CACHE[d] = (m, SVMTrainer(m))
    return _CACHE[d]


def _wavg(params: List[np.ndarray], sizes: List[int]) -> np.ndarray:
    w = np.asarray(sizes, dtype=np.float64)
    return np.average(np.stack(params), axis=0, weights=w / w.sum())


def fedavg_svm(Xc, yc, rounds=20, local_epochs=2, lr=0.1, lam=1e-5, mu_p=0.0, squared=True, batch_size=64,
               decay=1.0, seed=0, val_fn: Optional[Callable] = None, val_every=1):
    """Iterative FedAvg (mu_p=0) / FedProx (mu_p>0) with warm start from the global model (Eq. 11-12)."""
    d = Xc[0].shape[1]
    model, tr = _trainer(d)
    g = np.zeros(d + 1)
    hist, active = [], [k for k in range(len(Xc)) if len(Xc[k]) > 0]
    t0 = time.time()
    for r in range(rounds):
        ps, ns, losses = [], [], []
        for k in active:
            model.set_params(g)
            losses.append(tr.fit(Xc[k], yc[k], epochs=local_epochs, batch_size=batch_size, lr=lr / (1 + decay * r), lam=lam,
                                 mu=mu_p, global_params=g, squared=squared, rng=np.random.RandomState(seed * 100003 + r * 131 + k)))
            ps.append(model.get_params()); ns.append(len(Xc[k]))
        g = _wavg(ps, ns)
        rec = {"round": r + 1, "train_loss": float(np.mean(losses))}
        if val_fn is not None and ((r + 1) % val_every == 0 or r == rounds - 1):
            rec.update(val_fn(g))
        hist.append(rec)
    return g, dict(hist=hist, sec=time.time() - t0, comm_bytes=rounds * len(active) * 2 * 4 * (d + 1))


def oneshot_svm(Xc, yc, epochs=20, lr=0.1, lam=1e-5, squared=True, batch_size=64, seed=0):
    """Cold-start local training + a single size-weighted average (what the senior code effectively does), in TensorFlow."""
    d = Xc[0].shape[1]
    model, tr = _trainer(d)
    ps, ns = [], []
    t0 = time.time()
    for k in range(len(Xc)):
        if len(Xc[k]) == 0 or len(np.unique(yc[k])) < 2:
            continue                                  # a single-class client cannot fit a discriminative binary SVM
        model.set_params(np.zeros(d + 1))
        tr.fit(Xc[k], yc[k], epochs=epochs, batch_size=batch_size, lr=lr, lr_end=lr * 0.02, lam=lam, squared=squared,
               rng=np.random.RandomState(seed * 100003 + k))
        ps.append(model.get_params()); ns.append(len(Xc[k]))
    g = _wavg(ps, ns) if ps else np.zeros(d + 1)
    return g, dict(sec=time.time() - t0, comm_bytes=len(ps) * 2 * 4 * (d + 1), n_used=len(ps), n_skipped=len(Xc) - len(ps))


def senior_sklearn(Xc, yc, C=1.0, loss="squared_hinge", max_iter=1000, seed=42):
    """The senior Algorithm 2 as implemented: independent LinearSVC per client + size-weighted average (3 identical rounds
    collapse to one). Single-class clients cannot be fitted by LinearSVC and are skipped."""
    from sklearn.svm import LinearSVC
    import warnings
    d = Xc[0].shape[1]
    ps, ns = [], []
    t0 = time.time()
    for k in range(len(Xc)):
        if len(Xc[k]) == 0 or len(np.unique(yc[k])) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LinearSVC(C=C, loss=loss, max_iter=max_iter, random_state=seed, dual=True).fit(Xc[k], yc[k])
        ps.append(np.concatenate([m.coef_[0], m.intercept_])); ns.append(len(Xc[k]))
    g = _wavg(ps, ns) if ps else np.zeros(d + 1)
    return g, dict(sec=time.time() - t0, comm_bytes=len(ps) * 3 * 4 * (d + 1), n_used=len(ps), n_skipped=len(Xc) - len(ps))


def central_svm(X, y, epochs=20, lr=0.1, lam=1e-5, squared=True, batch_size=64, seed=0):
    d = X.shape[1]
    model, tr = _trainer(d)
    model.set_params(np.zeros(d + 1))
    t0 = time.time()
    tr.fit(X, y, epochs=epochs, batch_size=batch_size, lr=lr, lr_end=lr * 0.02, lam=lam, squared=squared,
           rng=np.random.RandomState(seed))
    return model.get_params(), dict(sec=time.time() - t0)


def svm_predict(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return ((X.astype(np.float64) @ theta[:-1] + theta[-1]) > 0).astype(np.int8)


# ----------------------------------------------------------------------------------------------- neural (Keras)
def _mlp(d, hidden=(100, 50), l2=1e-3):
    reg = tf.keras.regularizers.l2(l2 / 2.0)
    layers = [tf.keras.layers.Input((d,))]
    layers += [tf.keras.layers.Dense(h, activation="relu", kernel_regularizer=reg) for h in hidden]
    layers += [tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=reg)]
    return tf.keras.Sequential(layers)


def fedavg_mlp(Xc, yc, rounds=20, local_epochs=1, lr=0.05, l2=1e-3, batch_size=128, decay=1.0, seed=0):
    """FedAvg with a Keras MLP 17-100-50-1 (plain SGD locally; weights averaged layer-wise by client size)."""
    tf.keras.utils.set_random_seed(seed)
    d = Xc[0].shape[1]
    model = _mlp(d, l2=l2)
    opt = tf.keras.optimizers.SGD(lr)
    model.compile(optimizer=opt, loss="binary_crossentropy")
    g = model.get_weights()
    active = [k for k in range(len(Xc)) if len(Xc[k]) > 0]
    t0 = time.time()
    for r in range(rounds):
        opt.learning_rate.assign(lr / (1 + decay * r))
        ws, ns = [], []
        for k in active:
            model.set_weights(g)
            model.fit(Xc[k], yc[k], epochs=local_epochs, batch_size=batch_size, shuffle=True, verbose=0)
            ws.append(model.get_weights()); ns.append(len(Xc[k]))
        w = np.asarray(ns, float) / np.sum(ns)
        g = [sum(wi * layer[j] for wi, layer in zip(w, ws)) for j in range(len(g))]
    model.set_weights(g)
    nparam = int(sum(np.prod(a.shape) for a in g))
    return model, dict(sec=time.time() - t0, comm_bytes=rounds * len(active) * 2 * 4 * nparam, n_params=nparam)


def mlp_predict(model, X):
    return (model.predict(X, batch_size=16384, verbose=0)[:, 0] > 0.5).astype(np.int8)


# ----------------------------------------------------------------------------------------------- base paper: FL-CNN
def _cnn(d, filters=(32, 64, 128), dropout=0.5):
    """Base-paper CNN (Karunamurthy et al. 2025, Table 2: filters 32/64/128, kernel 3, ReLU, dropout 0.5, softmax).
    Tabular flow features are treated as a 1-D sequence (Conv1D); each convolution is followed by max-pooling."""
    layers = [tf.keras.layers.Input((d, 1))]
    for f in filters:
        layers += [tf.keras.layers.Conv1D(f, 3, padding="same", activation="relu"),
                   tf.keras.layers.MaxPooling1D(2, padding="same")]
    layers += [tf.keras.layers.Flatten(), tf.keras.layers.Dropout(dropout), tf.keras.layers.Dense(2, activation="softmax")]
    return tf.keras.Sequential(layers)


def fedavg_cnn(Xc, yc, rounds=20, local_epochs=1, lr=1e-3, batch_size=1000, seed=0):
    """FedAvg of the base-paper CNN: every round each client starts from the global weights, trains with a fresh Adam
    optimiser, and the server averages the weights by client size."""
    tf.keras.utils.set_random_seed(seed)
    d = Xc[0].shape[1]
    model = _cnn(d)
    opt = tf.keras.optimizers.Adam(lr)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy")      # compiled once: no re-tracing per client
    g = model.get_weights()
    active = [k for k in range(len(Xc)) if len(Xc[k]) > 0]
    t0 = time.time()
    for r in range(rounds):
        ws, ns = [], []
        for k in active:
            model.set_weights(g)
            if opt.built:                                  # fresh Adam state for every client and round
                opt.iterations.assign(0)
                for v in list(opt._momentums) + list(opt._velocities):
                    v.assign(tf.zeros_like(v))
            model.fit(Xc[k][:, :, None], yc[k], epochs=local_epochs, batch_size=batch_size, shuffle=True, verbose=0)
            ws.append(model.get_weights()); ns.append(len(Xc[k]))
        w = np.asarray(ns, float) / np.sum(ns)
        g = [sum(wi * layer[j] for wi, layer in zip(w, ws)) for j in range(len(g))]
    model.set_weights(g)
    nparam = int(sum(np.prod(a.shape) for a in g))
    return model, dict(sec=time.time() - t0, comm_bytes=rounds * len(active) * 2 * 4 * nparam, n_params=nparam)


def cnn_predict(model, X):
    return np.argmax(model.predict(X[:, :, None], batch_size=16384, verbose=0), axis=1).astype(np.int8)
