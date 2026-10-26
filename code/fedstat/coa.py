"""
Binary Chimp Optimization Algorithm (COA) wrapper feature selection, re-implemented for the base paper
(Karunamurthy et al., Sci. Rep. 2025, Eqs. 1-15). The base paper does not give the population size, iterations,
alpha, the classifier inside the fitness, or the binarisation; the choices below are ours and are logged in
decisions_log.md (D13).

  * position x_i in [0,1]^D, feature j selected if x_ij > 0.5 (at least one feature is always kept)
  * leaders = the 4 best solutions found so far (attacker, barrier, chaser, driver), Eqs. 5-13
  * f decays linearly 2.5 -> 0, a = 2 f r1 - f, c = 2 r2, m = logistic-map chaotic vector, Eqs. 1-4
  * with probability 0.5 a chimp is replaced by the chaotic vector, Eq. 14
  * fitness (maximised) = alpha * A + (1 - alpha) * log10(N / n), Eq. 15, A = accuracy of a 5-NN classifier
    fitted on a stratified subsample of the TRAINING rows and scored on a disjoint holdout of TRAINING rows.
    Validation and test rows are never seen.
"""
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def _holdout(X, y, rng, n_fit=4000, n_hold=2000):
    idx = rng.permutation(len(y))
    fit, hold = [], []
    for c in (0, 1):
        ic = idx[y[idx] == c]
        k_fit = min(len(ic) // 2, n_fit // 2)
        k_hold = min(len(ic) - k_fit, n_hold // 2)
        fit.append(ic[:k_fit]); hold.append(ic[k_fit:k_fit + k_hold])
    return np.concatenate(fit), np.concatenate(hold)


def select(X, y, seed, pop=10, iters=20, alpha=0.99):
    """X, y: pooled TRAINING rows only. Returns (keep indices into X columns, info dict)."""
    t0 = time.time()
    rng = np.random.RandomState(seed)
    X = X.astype(np.float64)
    sd = X.std(0)
    usable = np.where(sd > 0)[0]                          # constant columns cannot be standardised
    Xu = (X[:, usable] - X[:, usable].mean(0)) / sd[usable]
    D = len(usable)
    fit_i, hold_i = _holdout(Xu, y, rng)
    cache = {}

    def fitness(mask):
        key = mask.tobytes()
        if key not in cache:
            cols = np.where(mask)[0]
            knn = KNeighborsClassifier(n_neighbors=5, n_jobs=1).fit(Xu[np.ix_(fit_i, cols)], y[fit_i])
            acc = float((knn.predict(Xu[np.ix_(hold_i, cols)]) == y[hold_i]).mean())
            cache[key] = (alpha * acc + (1 - alpha) * np.log10(D / len(cols)), acc)
        return cache[key]

    def to_mask(x):
        m = x > 0.5
        if not m.any():
            m[np.argmax(x)] = True
        return m

    P = rng.uniform(0, 1, (pop, D))
    chaos = rng.uniform(0.05, 0.95, D)
    best = []                                            # (fitness, acc, position) of the best solutions so far

    def remember(x):
        fv, acc = fitness(to_mask(x))
        best.append((fv, acc, x.copy()))
        best.sort(key=lambda z: -z[0])
        del best[4:]
        return fv

    for i in range(pop):
        remember(P[i])
    curve = [best[0][0]]
    for t in range(iters):
        f = 2.5 - 2.5 * t / iters
        for i in range(pop):
            chaos = 4.0 * chaos * (1.0 - chaos)          # logistic map, m in Eq. 1
            if rng.rand() < 0.5:
                parts = []
                for lead in best[:4]:
                    r1, r2 = rng.rand(D), rng.rand(D)
                    a, c = 2 * f * r1 - f, 2 * r2
                    d = np.abs(c * lead[2] - chaos * P[i])
                    parts.append(lead[2] - a * d)
                P[i] = np.clip(np.mean(parts, axis=0), 0, 1)
            else:
                P[i] = chaos.copy()                      # Eq. 14, chaotic value
            remember(P[i])
        curve.append(best[0][0])
    keep = usable[np.where(to_mask(best[0][2]))[0]]
    return keep, dict(fitness=best[0][0], holdout_acc=best[0][1], n_usable=int(D), curve=curve,
                      evals=len(cache), sec=time.time() - t0, alpha=alpha, pop=pop, iters=iters)
