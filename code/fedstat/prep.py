"""
Federated preprocessing from additive sufficient statistics (the proposed FedStat-FS layer) + the pooled and
per-client reference implementations it is compared with.

Everything the server needs for (i) the variance filter, (ii) Pearson selection, (iii) LDA selection,
(iv) their intersection and (v) global standardisation is a function of per-class counts, means and centred
second moments.  Clients therefore upload only additive statistics (no rows).  With secure aggregation the
server would only ever see the sums; here the sum is computed in the clear because we simulate one machine.

Protocol (2 tiny rounds, exact w.r.t. the pooled computation):
  round A: client k -> (n_k0, n_k1, S1_k0, S1_k1)                           # counts and per-class feature sums
           server   -> global mean mu, class means mu_0, mu_1
  round B: client k -> M_kc = sum_{x in class c} (x - mu)(x - mu)^T          # centred around the GLOBAL mean
           server   -> global covariance, within-class scatter S_W, Pearson, LDA direction, scaler
Centring around the global mean (instead of raw second moments) avoids catastrophic cancellation for heavy-tailed
network features such as byte counts.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


# ------------------------------------------------------------------------------------------------- statistics
@dataclass
class RoundA:                       # client -> server, first upload
    n: np.ndarray                   # (2,)   class counts
    s1: np.ndarray                  # (2, d) per-class feature sums


@dataclass
class RoundB:                       # client -> server, second upload
    m2: np.ndarray                  # (2, d, d) per-class centred second-moment matrices around the global mean


def client_round_a(X: np.ndarray, y: np.ndarray) -> RoundA:
    X = X.astype(np.float64)
    n = np.array([(y == 0).sum(), (y == 1).sum()], dtype=np.float64)
    s1 = np.stack([X[y == 0].sum(0), X[y == 1].sum(0)])
    return RoundA(n, s1)


def client_round_b(X: np.ndarray, y: np.ndarray, mu: np.ndarray) -> RoundB:
    Xc = X.astype(np.float64) - mu
    m2 = np.stack([Xc[y == 0].T @ Xc[y == 0], Xc[y == 1].T @ Xc[y == 1]])
    return RoundB(m2)


def bytes_uploaded(d: int) -> int:
    """float64 payload of both rounds for one client (symmetric matrices could halve round B)."""
    return 8 * (2 + 2 * d + 2 * d * d)


# ------------------------------------------------------------------------------------------------- selection
@dataclass
class Prep:
    """A fitted preprocessing pipeline: kept feature indices + affine standardisation of those features."""
    keep: np.ndarray                        # indices into the original feature vector
    mean: np.ndarray                        # (len(keep),)
    std: np.ndarray                         # (len(keep),)
    info: Dict = field(default_factory=dict)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X[:, self.keep].astype(np.float64) - self.mean) / self.std).astype(np.float32)


def _safe_std(v):
    s = np.sqrt(v)
    return np.where(s > 0, s, 1.0)


def select_from_moments(n, mu_c, cov_total, sw, d, tau=0.2, var_thr=0.01, lda_top=20, mode="intersection",
                        mean_all=None, var_all_ddof1=None) -> Dict:
    """Shared arithmetic: variance filter -> Pearson(|rho|>tau) -> LDA(top-k |coef| on standardised features)."""
    n0, n1 = n
    N = n0 + n1
    keep1 = np.where(var_all_ddof1 >= var_thr)[0]                            # step 1: variance filter (raw scale, ddof=1)
    # Pearson correlation of each feature with the binary label
    ybar = n1 / N
    cov_xy = (mu_c[1] - mean_all) * ybar                                     # cov(x,y) = (mu_1 - mu) * p1  (population)
    var_y = ybar * (1 - ybar)
    sd_x = np.sqrt(np.maximum(np.diag(cov_total), 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = cov_xy / (sd_x * np.sqrt(var_y))
    rho = np.nan_to_num(rho)
    pear = keep1[np.abs(rho[keep1]) > tau]
    # LDA on the standardised, variance-filtered features:  w_z = sigma * S_W^{-1} (mu_1 - mu_0)
    sub = keep1
    sig = _safe_std(np.diag(cov_total)[sub])
    sw_z = sw[np.ix_(sub, sub)] / np.outer(sig, sig)
    dmu_z = (mu_c[1] - mu_c[0])[sub] / sig
    w_z = np.linalg.pinv(sw_z, rcond=1e-10, hermitian=True) @ dmu_z
    order = np.argsort(-np.abs(w_z))
    lda = sub[order[:lda_top]]
    if mode == "intersection":
        final = np.array(sorted(set(pear.tolist()) & set(lda.tolist())), dtype=int)
    elif mode == "pearson":
        final = np.array(sorted(pear.tolist()), dtype=int)
    elif mode == "lda":
        final = np.array(sorted(lda.tolist()), dtype=int)
    elif mode == "none":
        final = keep1
    else:
        raise ValueError(mode)
    return dict(keep=final, keep_var=keep1, pearson=pear, lda=lda, rho=rho, lda_coef=dict(zip(sub.tolist(), w_z.tolist())))


def fit_federated(client_data: List, tau=0.2, var_thr=0.01, lda_top=20, mode="intersection") -> Prep:
    """
    Proposed FedStat-FS. `client_data` is a list of (X_k, y_k) that never leave their client: only RoundA/RoundB
    objects are passed to the server code below.
    """
    d = client_data[0][0].shape[1]
    A = [client_round_a(X, y) for X, y in client_data]                    # -- client side --
    n = sum(a.n for a in A)                                               # -- server side: sums only --
    s1 = sum(a.s1 for a in A)
    mu_c = s1 / n[:, None]
    mean_all = s1.sum(0) / n.sum()
    B = [client_round_b(X, y, mean_all) for X, y in client_data]          # -- client side (uses broadcast mean) --
    m2 = sum(b.m2 for b in B)                                             # -- server side --
    N = n.sum()
    cov_total = (m2.sum(0)) / N                                           # population covariance (ddof = 0) around mu
    # within-class scatter S_W = sum_c [ M_c - n_c (mu_c - mu)(mu_c - mu)^T ]
    sw = sum(m2[c] - n[c] * np.outer(mu_c[c] - mean_all, mu_c[c] - mean_all) for c in (0, 1))
    var_ddof1 = np.diag(m2.sum(0)) / (N - 1)
    sel = select_from_moments(n, mu_c, cov_total, sw, d, tau, var_thr, lda_top, mode, mean_all, var_ddof1)
    keep = sel["keep"]
    mean = mean_all[keep]
    std = _safe_std(np.diag(cov_total)[keep])                            # ddof=0, same as sklearn StandardScaler
    return Prep(keep, mean, std, dict(kind="federated", n_keep_var=len(sel["keep_var"]), n_pearson=len(sel["pearson"]),
                                      n_lda=len(sel["lda"]), sel=sel, bytes_per_client=bytes_uploaded(d)))


def fit_pooled(X: np.ndarray, y: np.ndarray, tau=0.2, var_thr=0.01, lda_top=20, mode="intersection") -> Prep:
    """Pooled (centralised) reference: identical arithmetic on the concatenated data - what the senior code does."""
    X = X.astype(np.float64)
    d = X.shape[1]
    n = np.array([(y == 0).sum(), (y == 1).sum()], float)
    mu_c = np.stack([X[y == 0].mean(0), X[y == 1].mean(0)])
    mean_all = X.mean(0)
    Xc = X - mean_all
    m2 = np.stack([Xc[y == 0].T @ Xc[y == 0], Xc[y == 1].T @ Xc[y == 1]])
    cov_total = m2.sum(0) / len(X)
    sw = sum(m2[c] - n[c] * np.outer(mu_c[c] - mean_all, mu_c[c] - mean_all) for c in (0, 1))
    sel = select_from_moments(n, mu_c, cov_total, sw, d, tau, var_thr, lda_top, mode, mean_all, X.var(0, ddof=1))
    keep = sel["keep"]
    return Prep(keep, mean_all[keep], _safe_std(np.diag(cov_total)[keep]),
                dict(kind="pooled", n_keep_var=len(sel["keep_var"]), n_pearson=len(sel["pearson"]), n_lda=len(sel["lda"]), sel=sel))


def fit_local_vote(client_data: List, tau=0.2, var_thr=0.01, lda_top=20, mode="intersection", vote_frac=0.5) -> Prep:
    """
    Naive federated baseline WITHOUT sharing statistics: every client runs the senior pipeline on its own rows
    (own variance filter, Pearson, LDA, own scaler).  Clients upload only a 0/1 vote vector per feature; the
    features chosen by at least `vote_frac` of the clients are kept.  Each client then standardises with its OWN
    scaler, so the returned Prep carries per-client scalers in info["client_scalers"].
    """
    d = client_data[0][0].shape[1]
    votes = np.zeros(d)
    n_valid = 0
    for X, y in client_data:
        if len(np.unique(y)) < 2:                                          # single-class client cannot run LDA/Pearson
            continue
        n_valid += 1
        p = fit_pooled(X, y, tau, var_thr, lda_top, mode)                   # same code, but on ONE client's rows
        votes[p.keep] += 1
    keep = np.where(votes >= vote_frac * max(n_valid, 1))[0]
    if len(keep) == 0:                                                     # degenerate vote: fall back to top-5 voted
        keep = np.argsort(-votes)[:5]
    scalers = []
    for X, y in client_data:                                               # every client standardises with its OWN stats
        Xk = X[:, keep].astype(np.float64)
        scalers.append((Xk.mean(0), _safe_std(Xk.var(0))))
    dummy_mean, dummy_std = np.zeros(len(keep)), np.ones(len(keep))
    return Prep(keep, dummy_mean, dummy_std, dict(kind="local_vote", client_scalers=scalers, votes=votes,
                                                  n_valid_voters=n_valid))


def local_scaler_transform(prep: Prep, X: np.ndarray, client_id: int) -> np.ndarray:
    m, s = prep.info["client_scalers"][client_id]
    return ((X[:, prep.keep].astype(np.float64) - m) / s).astype(np.float32)
