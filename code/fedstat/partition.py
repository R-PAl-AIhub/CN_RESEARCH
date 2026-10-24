"""
Client partitioners. A partition is a proportion matrix P (categories x clients); every split (train / val / test)
is divided among the clients with the SAME proportions, so each client's held-out shards follow its own training
distribution (the deployment scenario of a gateway that keeps seeing its own kind of traffic).
"""
import numpy as np


def proportions_iid(n_cat: int, K: int) -> np.ndarray:
    return np.full((n_cat, K), 1.0 / K)


def proportions_dirichlet(n_cat: int, K: int, alpha: float, rng: np.random.RandomState, min_mass=0.02) -> np.ndarray:
    """Hsu et al. (2019) label-skew: for every category c draw p_c ~ Dir(alpha * 1_K). Re-draw until every client
    receives at least `min_mass` of one category so that no client is empty."""
    for _ in range(200):
        P = rng.dirichlet(alpha * np.ones(K), size=n_cat)
        if (P.max(axis=0) >= min_mass).all():
            return P
    return P


def split_by_proportions(cat: np.ndarray, P: np.ndarray, rng: np.random.RandomState):
    """Assign every row (with category id cat[i]) to a client according to P[cat[i]]. Returns list of index arrays."""
    K = P.shape[1]
    owner = np.empty(len(cat), dtype=int)
    for c in range(P.shape[0]):
        idx = np.where(cat == c)[0]
        rng.shuffle(idx)
        if len(idx) == 0:
            continue
        # deterministic proportional cut (largest-remainder) rather than i.i.d. sampling -> smoother partitions
        cuts = np.floor(np.cumsum(P[c]) * len(idx)).astype(int)
        cuts[-1] = len(idx)
        start = 0
        for k in range(K):
            owner[idx[start:cuts[k]]] = k
            start = cuts[k]
    return [np.where(owner == k)[0] for k in range(K)]


def by_group(group: np.ndarray, K=None):
    """Natural partition (e.g. N-BaIoT device id)."""
    ids = np.unique(group)
    return [np.where(group == g)[0] for g in ids]
