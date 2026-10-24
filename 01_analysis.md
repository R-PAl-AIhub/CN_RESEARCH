# 01 - Analysis of the base paper, senior paper and senior codebase

## LOCKED RESEARCH DOMAIN

> **Federated-learning-based intrusion detection (DDoS/flood detection) for IoT networks using flow-level features and
> feature selection, with lightweight (linear) detectors.**

Everything downstream (gaps, ideas, datasets, citations) stays inside this domain: federated intrusion detection for IoT.

---

## 1. Base paper

**Karunamurthy, Vijayan, Kshirsagar, Tan - "An optimal federated learning-based intrusion detection for IoT environment",
*Scientific Reports* 15:8696 (2025), doi:10.1038/s41598-025-93501-8.**

| Item | Content |
|---|---|
| Problem | Centralised ML-IDS needs pooled traffic (privacy, data-breach risk); FL-IDS should also cope with high-dimensional inputs. |
| Method | Federated CNN classifier on each IoT gateway + **Chimp Optimisation Algorithm (COA)** wrapper feature selection (fitness = alpha*accuracy + (1-alpha)*log10(N/n)); FedAvg-style aggregation (Eq. 16-17); global model pushed back to clients. |
| Dataset | MQTTset (Vaccari et al. 2020, Kaggle): 6 classes (legitimate + Flooding DoS, MQTT publish flood, SlowITe, malformed data, brute force). Balanced, 70/30 split. 34 -> 17 features. |
| Metrics | Accuracy, precision, recall, F1 per class and macro-averaged. |
| Results | FL-CNN: accuracy 0.9559, macro-F1 0.9404, vs RF 0.9160, DT 0.9159, NN 0.9045, MLP 0.9038, GB 0.8842, NB 0.6439. |
| Stated limitations | Poisoned/malicious local updates hurt the global model; the optimiser adds computational cost on constrained devices. |
| Observations (ours) | Baseline rows in Table 10 carry a citation to a different, earlier paper (ref. [29]), so it is unclear whether the baselines were re-run on the same split; no non-IID study, no significance tests, a single 70/30 split, no code-availability statement; the notation of Eq. 17 reuses the per-sample symbol f_i for the global objective. |

## 2. Senior paper

**Haris Mohamed C P and A. Kundu - "Privacy-Preserving Federated Learning Framework for DDoS Attack Detection in IoT Networks Using
Hybrid Feature Selection", IEEE Access (manuscript dated 2026, DOI placeholder).**

| Item | Content |
|---|---|
| Problem | Binary DDoS detection with FL on IoT gateways; wants a *lightweight, interpretable* federated model that matches centralised accuracy. |
| Method | (i) **Hybrid filter selection**: variance filter (<0.01) -> Pearson |rho|>0.2 -> LDA top-20 -> **intersection** (46 -> 17 features). (ii) **Federated linear SVM**: each worker trains `LinearSVC`, uploads (w, b, n_k); server takes the size-weighted average (Eq. 5); R=3 rounds; N in {3,5,10}; IID partitions. |
| Dataset | CIC-IoT2023 (Neto et al. 2023), 500,000 flows (51% benign / 49% DDoS), 80/20 split, 46 features. |
| Metrics | Accuracy, precision, recall, F1, FPR/FNR, training time, memory/node, communication bytes. |
| Results | Federated SVM F1 0.9881, acc 0.9884 (= centralised); RF F1 0.9995; ANN 0.9982; Isolation Forest 0.4173. |
| Stated limitations | Controlled-lab data, binary task, **IID assumption**, **static feature selection**, homogeneous workers. |

**How the senior paper extends the base paper.** Same domain and the same "feature selection + federated aggregation" recipe, but:
COA wrapper selection -> cheap two-filter intersection; federated CNN -> federated linear SVM (interpretable, 18 floats/round);
MQTTset multi-class -> CIC-IoT2023 binary DDoS; adds resource metrics (time/memory/communication) and a centralised upper bound.

## 3. Senior codebase (`Senior_code/`, scikit-learn; 5 modules)

Structure: `config.py` (all constants) -> `data_preprocessing.py` (`DataLoader`, `FeatureSelector` = Algorithm 1) ->
`federated_learning.py` (`FederatedSVM` = Algorithm 2, `CentralizedModels`, `evaluate_model`) -> `main_experiment.py` -> `visualization.py`.

### 3.1 Reproduction (executed on CPU inside `./.venv`)

* Data: real CIC-IoT2023 flows (46-feature release; row counts verified against the published release, see `data/processed/ciciot_pool_summary.json`).
  A uniformly random 255,000 benign + 245,000 DDoS sample (seed 42) was written to CSV and consumed by the **unchanged** senior scripts
  (`code/senior_repro/`, only `DATASET_PATH` differs; diff-verified). Log: `results/senior_repro_run1.log`; outputs: `code/senior_repro/outputs/`.

| Quantity | Senior paper | Reproduced (our run, real data) |
|---|---|---|
| Features 46 -> after variance / Pearson / LDA / final | 46 -> 40 / 35 / 20 / **17** | 46 -> 34 / 24 / 20 / **17** |
| 10-fold CV F1 (feature-selection check) | 0.985-0.991 | 0.9964 (code's own range check 0.90-0.97 prints a warning) |
| Central SVM acc / prec / rec / F1 | .9884 / .9965 / .9798 / .9881 | .9964 / .9963 / .9964 / .9963 |
| Fed. SVM (5 workers) acc / prec / rec / F1 | .9884 / .9965 / .9798 / .9881 | .9966 / .9967 / .9964 / .9966 |
| Fed = Central parity (N = 3, 5, 10) | yes | yes (F1 .9965 / .9966 / .9966) |
| Random Forest F1 | .9995 | .9984 |
| ANN F1 | .9982 | .9987 |
| Isolation Forest F1 | .4173 | .1974 |
| FPR / FNR (fed. SVM, N=5) | 0.34% / 2.02% | 0.32% / 0.36% |

**Verdict.** Qualitative claims reproduce (parity, scale invariance, RF/ANN slightly above SVM, Isolation Forest fails, 46->17 features).
Quantitatively our SVM F1 is **higher** than the paper's (0.9966 vs 0.9881, recall 0.996 vs 0.980); the gap is attributable to the sample
(the senior loader keeps the *first* rows it meets file-by-file, which is not a random draw; its file order is not published) and is
within what a different DDoS-type mix can explain. We treat 0.9966 as the reproduced baseline; we do not "tune" toward the paper's number.

### 3.2 TensorFlow port (verified against sklearn on the senior's own preprocessed arrays)

The senior code is scikit-learn (SVM, RF, IF, MLP); PyTorch is *not* used (it appears only inside a `try/except` in the seeding helper).
Deep-learning parts are ported to TensorFlow/Keras (`code/fedstat/models.py`); results in `results/phase1_tf_port_check.json`:

| Model | sklearn (senior) F1 | TensorFlow port F1 | |Delta| |
|---|---|---|---|
| Centralised linear SVM | 0.9963 | 0.9966 (squared hinge) / 0.9969 (hinge) | <= 0.0006 |
| Federated SVM N=5, one-shot average | 0.9966 | 0.9965 | 0.0001 |
| Federated SVM N=5, iterative warm-start FedAvg | (not implemented in senior code) | 0.9964 | - |
| ANN 17-100-50-1 | 0.9987 | 0.9977 (Keras) | 0.0010 |

Mapping used: `LinearSVC(C=1)` <-> mean-loss objective with lambda = 1/(C n); an initial lambda=1e-4 under-fit (F1 0.9918) and was corrected
(logged in `decisions_log.md` D4).

### 3.3 Code-versus-paper mismatches and defects (each verified by reading the code and/or the run log)

| # | Finding | Evidence | Consequence |
|---|---|---|---|
| M1 | **Feature selection and standardisation are fitted on the entire dataset before the train/test split** (`selector.fit_transform(df)` line 372, `train_test_split` line 379). | `data_preprocessing.py` | Test-set statistics leak into selection and scaling; also **contradicts the privacy premise** (Alg. 1 needs all raw rows in one place). |
| M2 | **"Rounds" are not iterative.** Every round builds a fresh `LinearSVC` per worker (cold start) on the same partition; `global_model` is overwritten but never sent back. | `federated_learning.py` l. 99, 126; run log: identical worker F1 in rounds 1, 2, 3 | Algorithm 2 (download M_global, refine) is not what is executed; result = one-shot parameter averaging. The claim "F1 gain between round 3 and 4 < 0.001" is trivially 0. |
| M3 | Isolation Forest, RF and ANN are trained **centrally**, although the paper says they run "in the same federated setting". | `CentralizedModels.*` | Table 4 is not a federated comparison. |
| M4 | Paper: "Loss: Hinge". Code: sklearn default `squared_hinge`. | grep, no `loss=` argument | Objective differs from Eq. 4. |
| M5 | Paper: "PyTorch coordinates distributed execution". Code: no torch usage. | grep | Software description inaccurate. |
| M6 | `ANN_ALPHA = 0.001 # Learning rate` is passed as sklearn `alpha` (L2 penalty), not the learning rate. | `config.py` l. 113, `federated_learning.py` | Hyper-parameter mislabelled. |
| M7 | CV range check (0.90 < F1 < 0.97) contradicts the paper's reported CV F1 (0.985-0.991); the check prints a warning on real data. | run log | Inconsistent acceptance criterion. |
| M8 | Feature counts along the pipeline (40/35) do not reproduce (34/24); only the final 17 does. | run log | Intermediate numbers in Table 2 are sample-dependent. |
| M9 | Paper claims "Byzantine resilience" from size-weighting, and that the SVM landscape is "non-convex". Neither is true: size weighting gives a large-data attacker *more* influence; the SVM primal is convex. | Sec. III-A/B | Unsupported/incorrect claims (we do not repeat them). |
| M10 | Sequential first-N sampling in `_load_cic_iot2023` (per-file, per-chunk `sample(needed)`) is not a random draw of the 46 M flows and can over-represent the first attack types in file order. | `data_preprocessing.py` l. 89-149 | Explains part of the metric gap; sample bias. |
| M11 | Synthetic-data demo mode (`USE_REAL_DATASET=False`) exists. | `config.py` | **Not used anywhere in this project** (rule: no synthetic data). |
| M12 | 7.7% of flows in a random CIC-IoT2023 sample are exact duplicates (37,793 of 37,799 removable duplicates are DDoS); the random split of the senior code lets identical rows fall in both train and test. | our audit of `ciciot_pool.parquet` | Optimistic test metrics (quantified in Phase 4). |

## 4. What this analysis implies for the research plan
1. The senior pipeline's preprocessing is **not privacy-consistent** (M1) and is **leaky** (M1, M12): both matter for a paper whose selling point is privacy.
2. The "federated" training is one-shot averaging of independently trained linear SVMs (M2): plausible under IID data, untested under any heterogeneity.
3. All evidence is from one dataset and one IID split; base and senior papers share the same open questions (non-IID data, static features, malicious clients).
   These feed Phase 2 (`02_research_gaps.md`).
