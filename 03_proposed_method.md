# 03 - Proposed method: FedStat-FS (federated-statistics preprocessing for federated linear IDS)

## 1. Technique borrowed from the same domain
Screening of *other sub-areas of federated IDS* for techniques that solve a related problem but have not been applied to G5
(privacy-consistent hybrid-filter selection + scaling for a linear detector):

| Candidate (same domain) | Related problem it solved | Applied to G5? | Feasible on CPU? | Pick? |
|---|---|---|---|---|
| **StatAvg** - clients share feature statistics, server aggregates them, global statistics redistributed for uniform normalisation (Bouzinis et al., IEEE TNSM 2025) | Non-IID feature distributions in FL-IDS (deep models) | **No** - normalisation only; no selection, no linear/hybrid-filter pipeline, no leakage study | yes (d floats/client) | **YES - first feasible fit** |
| Federated chi-square selection with GRU (Fed-FeRe, Cybersecurity 2026) | Communication/compute reduction | Different selector and model | GRU heavier | later |
| Prototype sharing (PROTEAN 2025), knowledge distillation (FD-IDS 2025) | Non-IID label skew (deep models) | Not about preprocessing | heavier | no |
| Clustered FL (Saez-de-Camara et al. 2023) | Heterogeneous IoT clients | Not about preprocessing | yes | no |

**Core idea.** StatAvg's principle - *share sufficient statistics, not rows* - is extended from normalisation to the **whole preprocessing stage** of the senior
pipeline. Variance filter, Pearson correlation, the LDA direction and the standardiser are all functions of per-class counts, means and (centred) second moments.
Those are **additive across clients**, so the server can reproduce the pooled result *exactly* from two small uploads per client, and never sees a traffic record.
The downstream detector is then trained with *genuine* iterative FedAvg (warm start; senior code only does one-shot averaging) on the aligned, globally standardised
features. Optional FedProx term as an optimiser-level comparison.

**Why it should work.** (i) Exactness: Pearson and LDA depend on the data only through moments, hence federated == pooled (verified numerically: identical 17-feature
set as the senior sklearn code, mean/std differences ~1e-9/1e-15, also under strongly skewed partitions).
(ii) Under non-IID data, local scalers and local selections disagree (each client sees a different slice of the feature distribution); averaging weight vectors learned in
inconsistent coordinate systems degrades the global model, which shared statistics prevent.
(iii) Removing the pooled, pre-split fit removes test-set leakage and the data-locality violation.

**What changes in the senior code.**
| Senior component | Replacement | File |
|---|---|---|
| `FeatureSelector.fit_transform(df)` on pooled, pre-split data | `fit_federated(client_data)` (statistics only) fitted on training partitions only | `code/fedstat/prep.py` |
| `StandardScaler` on pooled data | global mean/std from the same statistics | `prep.py` |
| `FederatedSVM.fit`: cold-start `LinearSVC` per round, never uses global model | warm-start FedAvg with local SGD (hinge/squared hinge), optional prox term | `code/fedstat/fl.py`, `models.py` |
| IID random partition only | Dirichlet label/attack-type skew + natural device split | `code/fedstat/partition.py` |
| Test set inside preprocessing | strict train / validation / test; test used once | `code/fedstat/data.py`, `run_experiments.py` |

**Risks.** (R1) Because the method is exact, it *cannot* beat pooled train-only preprocessing; its value is privacy-consistency at no accuracy cost plus robustness versus naive local
preprocessing - if local preprocessing turns out equally good, we will report that. (R2) The uploaded d x d matrices reveal second-order statistics; secure aggregation (Bonawitz et al. 2017) or
DP noise would be needed in deployment (not implemented; stated as limitation). (R3) `pinv` of the scatter matrix is sensitive to near-collinear features (mitigated by the variance filter and rcond).
(R4) Linear detectors may underfit compared with MLP/RF.

## 2. Mathematical model (numbering of this file; the paper merges Eqs. 9-10 into one and folds the metrics into one equation, so its numbers from (9) on shift by one - see `05_visuals_index.md`)

**Setting.** K clients; client k holds D_k = {(x_i, y_i)}, x_i in R^d, y_i in {0,1} (0 benign, 1 DDoS), n_k = |D_k|, n = sum_k n_k.

**Eq. (1) - problem formulation.**  min_theta F(theta) = sum_{k=1}^{K} (n_k/n) F_k(theta),  F_k(theta) = (1/n_k) sum_{i in D_k} l(y_i, f_theta(phi(x_i))) + (lambda/2)||w||^2,
with theta = (w, b), f_theta(z) = w^T z + b, and preprocessing map phi(x) = (x_F - m) / s (feature subset F, mean m, std s).
*Variables:* l = loss (Eq. 11); lambda = L2 weight; F = selected feature indices; m, s = global mean/std of x_F. *Code:* `fl.py::fedavg`, `prep.py::Prep.transform`.

**Eq. (2) - round-A statistics (client -> server).**  n_kc = sum_{i in D_k} 1[y_i = c],  s_kc = sum_{i in D_k, y_i = c} x_i,  c in {0,1}.
**Eq. (3) - server aggregation.**  n_c = sum_k n_kc,  mu_c = (sum_k s_kc)/n_c,  mu = (n_0 mu_0 + n_1 mu_1)/n,  p_1 = n_1/n.
*Meaning:* class counts and class means of the *pooled* training data, recovered from sums. *Code:* `prep.py::client_round_a`, `fit_federated`.

**Eq. (4) - round-B statistics (client -> server, after mu is broadcast).**  M_kc = sum_{i in D_k, y_i = c} (x_i - mu)(x_i - mu)^T,  M_c = sum_k M_kc.
*Meaning:* centred second-moment matrices around the *global* mean (numerically stable for heavy-tailed byte/packet features). *Code:* `client_round_b`.

**Eq. (5) - covariance and variance filter.**  Sigma = (M_0 + M_1)/n,  sigma_j^2 = Sigma_jj;  v_j = (M_0 + M_1)_jj/(n - 1);  keep F_1 = {j : v_j >= nu} (nu = 0.01).
**Eq. (6) - Pearson correlation with the label.**  rho_j = p_1 (mu_{1j} - mu_j) / (sigma_j sqrt(p_1 (1 - p_1))),  F_P = {j in F_1 : |rho_j| > tau} (tau = 0.2).
*Derivation:* cov(x_j, y) = p_1 (mu_{1j} - mu_j), var(y) = p_1 (1 - p_1). *Code:* `prep.py::select_from_moments`.

**Eq. (7) - within-class scatter.**  S_W = sum_{c in {0,1}} [ M_c - n_c (mu_c - mu)(mu_c - mu)^T ].
**Eq. (8) - LDA direction on standardised features.**  w_LDA = D S_W^{-1} (mu_1 - mu_0), D = diag(sigma_1..sigma_d), restricted to F_1; F_L = indices of the top-k (k = 20) values of |w_LDA|.
*Meaning:* the coefficient of the Fisher discriminant for standardised inputs (identical ranking to sklearn `LinearDiscriminantAnalysis.coef_` on standardised data); S_W is inverted with a Moore-Penrose pseudo-inverse. *Code:* `select_from_moments` (`pinv`).

**Eq. (9) - hybrid selection (intersection).**  F = F_P intersect F_L.   **Eq. (10) - global standardiser.**  m_j = mu_j, s_j = sigma_j for j in F (ddof = 0, as sklearn `StandardScaler`).
*Code:* `select_from_moments(mode="intersection")`, `fit_federated`.

**Eq. (11) - detector loss and local step.**  l_hinge = max(0, 1 - y~ f), l_sq = max(0, 1 - y~ f)^2 with y~ = 2y - 1; local objective with proximal term:
H_k(theta; theta^t) = (1/n_k) sum_i l(...) + (lambda/2)||w||^2 + (mu_p/2)||theta - theta^t||^2 (mu_p = 0 gives FedAvg, > 0 FedProx); mini-batch SGD  theta <- theta - eta grad H_k.
**Eq. (12) - aggregation.**  theta^{t+1} = sum_k (n_k/n) theta_k^{t+1}   (FedAvg).   *Code:* `models.py::SVMTrainer.fit`, `fl.py::fedavg`.

**Evaluation metrics (Eq. 13-19).**  Accuracy = (TP+TN)/(TP+TN+FP+FN);  Precision = TP/(TP+FP);  Recall = TP/(TP+FN);  F1 = 2 P R/(P+R);  FPR = FP/(FP+TN);
MCC = (TP TN - FP FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN));  selection agreement J(F_a, F_b) = |F_a intersect F_b| / |F_a union F_b| (Jaccard).  *Code:* `metrics.py`.

**Communication accounting.**  statistics upload per client = 8(2 + 2d + 2d^2) bytes (float64, both rounds; symmetry would halve the d^2 term); model exchange per round per client = 2 * 4 * (|F| + 1) bytes (float32, up + down).

## 3. Algorithm 1 - FedStat-FS + federated linear detector

```
Input : clients k=1..K holding local training data D_k (never transmitted); thresholds nu, tau, top-k; rounds T; local epochs E;
        step size eta; lambda; proximal weight mu_p (0 for FedAvg)
Output: global detector theta_T and preprocessing (F, m, s)

 -- Phase 1: federated preprocessing (2 communication rounds, once) --
 1: each client k : send (n_kc, s_kc)                                      # Eq. (2)
 2: server        : n_c, mu_c, mu, p_1  <- sums; broadcast mu               # Eq. (3)
 3: each client k : send M_kc computed around mu                           # Eq. (4)
 4: server        : Sigma, v, rho, S_W; F_1, F_P, F_L                       # Eq. (5)-(8)
 5: server        : F <- F_P ∩ F_L ; (m, s) <- (mu_F, sigma_F); broadcast (F, m, s)   # Eq. (9)-(10)
 -- Phase 2: federated training (T rounds) --
 6: theta^0 <- 0
 7: for t = 0..T-1:
 8:    each client k (parallel): z_i <- (x_i[F]-m)/s ; theta_k <- theta^t ; run E epochs of SGD on H_k(.;theta^t)   # Eq. (11)
 9:    server: theta^{t+1} <- sum_k (n_k/n) theta_k                                                             # Eq. (12)
10: return theta_T, (F, m, s)
```
**Complexity.** Phase 1: client O(n_k d^2) time, O(d^2) memory, 2 uploads of O(d^2) numbers; server O(K d^2 + d^3) (pseudo-inverse). Phase 2 per round: client O(E n_k |F|), server O(K |F|);
model size O(|F|). Total extra communication of the proposed layer: one-off 8(2+2d+2d^2) B per client as implemented, i.e. full (non-symmetric) matrices in float64: d = 46: 33.8 kB; d = 115: 208.5 kB (exploiting symmetry would roughly halve this; not implemented). *Correction (Phase 6): an earlier draft of this file quoted 17.6 kB / 107 kB, which were the symmetric-halved values, not what the code uploads.*

## 4. Ablations planned (each isolates one new component)
A1 local scalers instead of global scaling (federated selection kept) - A2 no selection (all variance-filtered features, global scaling; StatAvg-like) - A3 one-shot averaging instead of iterative rounds (federated preprocessing kept) -
A4 Pearson-only / LDA-only instead of the intersection - plus baselines: senior pipeline as released (pooled pre-split preprocessing, cold-start local LinearSVC, one-shot average), the same without leakage, naive local preprocessing + FedAvg,
FedProx, federated MLP (Keras), centralised SVM and centralised random forest.

## 5. As implemented (final state after Phases 4-6)
- Code: `code/fedstat/prep.py` (Eqs. 2-10), `models.py` (Eq. 11; fused closed-form epoch, verified equal to autodiff in `code/tests/test_gradients.py`), `fl.py` (Eq. 12), `methods.py` (all compared pipelines), `metrics.py`.
- Final federated budget shared by every linear method: T = 20, E = 2, batch 64, step decay lr/(1+t) (decision D7 in `decisions_log.md`); MLP: E = 1, batch 128.
- Result of the risks listed in section 1: R1 materialised (exact => cannot beat pooled train-only preprocessing; the benefit is versus naive local preprocessing); R4 materialised (linear detector 0.14-0.47 F1 points below the federated MLP on CIC-IoT2023).
- New finding not anticipated in the plan: the Pearson-LDA *intersection* itself is unstable on N-BaIoT (see `04_experiments_log.md` and `results/diag_selection_failure.json`).
