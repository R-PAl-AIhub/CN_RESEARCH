# 02 - Research gaps (locked domain: federated intrusion/DDoS detection for IoT, flow features, lightweight detectors)

Method: (1) gaps collected in the order they surfaced while reading the two papers and auditing the code; (2) each tested, in that
order, against criteria (a)-(e); (3) the first gap passing all five is selected. Literature evidence was gathered with
reproducible probes (`code/data/lit_search.py`; raw hits in `results/literature/`), web searches, and reading the abstracts/papers named below.
Evidence limits: full texts of some MDPI/Elsevier papers were not accessible (abstracts only); OpenAlex free-text search is fuzzy (hit counts are indicative only), and paywalled full texts were judged from
abstracts; "no existing paper found" always means *within these searches as of 2026-09-21*.

Criteria: (a) inside the locked domain; (b) truly open (no existing paper solves it - evidence cited); (c) >= 2 real public datasets;
(d) CPU-feasible; (e) buildable by extending the senior codebase.

## Gap list (discovery order) and screening

| # | Gap (limitation / open problem inside the domain) | Where it came from | (a) | (b) truly open? | (c) | (d) | (e) | Decision |
|---|---|---|---|---|---|---|---|---|
| G1 | **Non-IID clients** for the federated linear-SVM IDS (senior limitation 3, base paper "heterogeneous IoT") | Senior Sec. VI-B/F, future work 2 | yes | **No.** Crowded: FD-IDS (Sensors 2025), PROTEAN (2025), Sentinel (2025), FedDWC clustered FL for IoT DDoS (Sci. Rep. 2025), dataset-centric non-IID evaluation on CIC-IoT2023/Edge-IIoTset/TII-SSRC-23 (Sci. Rep. 2026); OpenAlex "non-IID federated intrusion detection IoT": 2,422 hits since 2021. Not left as a *general* gap. | - | - | - | **Skipped** (fails b). Its *preprocessing-layer* facet is kept inside G5. |
| G2 | Binary detection only; multi-class attack typing | Senior limitation 2, future work 1 | yes | **No.** Multi-class FL-IDS is standard (MQTTset 6 classes in the base paper, Edge-IIoTset, CIC-IoT2023 33 classes). | - | - | - | **Skipped** (fails b). |
| G3 | Malicious/poisoned clients and Byzantine aggregation | Base paper conclusion; senior future work 3 | yes | **No.** Personalised-FL IDS poisoning & defence (FGCS 2023) and many robust-aggregation works. | - | - | - | **Skipped** (fails b). |
| G4 | Static feature selection (no adaptation under attack drift) | Senior limitation 4 | yes | Partly studied (incremental FL-IDS under evolving threats, arXiv 2603.10776; Fed-FeRe adaptive thresholds); an *evaluation on real drifting traffic* would need timestamped multi-day data that neither candidate dataset provides in usable form. | 1 | yes | yes | **Skipped** (fails c: no second dataset with realistic temporal structure that we can obtain). |
| G5 | **Privacy-inconsistent, leaky preprocessing in federated linear IDS pipelines.** The senior pipeline computes the variance filter, Pearson, LDA and the scaler on the *pooled* data *before the train/test split* (M1), so (i) raw traffic must be centralised for preprocessing - contradicting the privacy claim - and (ii) test statistics leak. A privacy-consistent alternative is missing, and it is unknown how naive local substitutes behave when clients are heterogeneous. | Code audit (M1, M2, M12) | yes | **Yes, at this granularity.** Closest works: (i) **StatAvg** (IEEE TNSM 2025) shares client feature statistics for *global normalisation* of deep FL-IDS - no feature selection, no linear/hybrid-filter pipeline, no leakage analysis; (ii) **Fed-FeRe** (Cybersecurity 2026) integrates chi-square selection with a GRU - different selector, deep model, no exactness/non-IID-preprocessing study; (iii) Yang & Kondo (ICECCE 2021) greedy per-attack selection with multiple global models; (iv) general privacy-preserving federated feature selection/correlation (Fed-chi2 arXiv 2021; DP feature selection, Eng. Appl. AI 2026) - not IDS, not LDA; (v) **Devine et al. (Electronics 2025)** report a federated SVM for IoT DDoS with "a novel approach to data preparation for federated learning" - the publisher blocks automated full-text access, so only the abstract was read; it does not say whether preparation statistics are aggregated federatively, so we cannot exclude a partial overlap and word every novelty claim as *to the best of our knowledge*. Note that this paper also contradicts the senior paper's statement that federated SVM had never been applied to real IoT traffic. Probes `G_fedfs`, `G_fedscaler`, `G_sufficient`, `G_linear` returned no work that derives *hybrid Pearson-and-LDA selection + global standardisation from additive statistics for a linear FL-IDS* or that measures pooled-preprocessing leakage against a federated alternative. | **2** (CIC-IoT2023, N-BaIoT) | yes | yes (extends `FeatureSelector`, `FederatedSVM`) | **SELECTED** |

Why earlier gaps were skipped: G1-G3 fail criterion (b) (solved or saturated in the peer-reviewed/preprint literature cited above); G4 fails (c).
G5 is the first gap that passes all of (a)-(e).

### Datasets for G5 (real, public, citable)
| Dataset | Source / licence / version | Why suitable |
|---|---|---|
| CIC-IoT2023 | https://www.unb.ca/cic/datasets/iotdataset-2023.html ; Neto et al., *Sensors* 23(13):5941, doi:10.3390/s23135941 ; obtained through the HF mirror `lacg030175/CIC-IoT-2023-neto-full` (declared CC-BY-4.0; row counts verified against the release: 46,686,579 flows, 1,098,195 benign, 33,984,560 DDoS) | Same dataset as the senior paper; 46 features; 11 DDoS types enable attack-type-skewed clients. |
| N-BaIoT | https://archive.ics.uci.edu/dataset/442 ; doi:10.24432/C5RC8J ; CC BY 4.0 ; Meidan et al., *IEEE Pervasive Comput.* 17(3), 2018 | Real traffic from 9 commercial IoT devices infected by Mirai/BASHLITE: a *natural* federated split (one client per device), 115 features. |

## Selected gap (final wording)
> Federated linear IDS pipelines still fit their preprocessing (feature selection and scaling) on pooled, pre-split data. No prior work
> shows how to obtain the *same* hybrid Pearson-and-LDA subset and global standardisation from client-side additive statistics, what the pooled
> pre-split protocol leaks, or how naive local preprocessing behaves when clients are non-IID.

## Research questions
* **RQ1 (fidelity and leakage).** Can Pearson-LDA-intersection selection and standardisation be computed *exactly* from additive client statistics, and how much optimism does the pooled, pre-split preprocessing of the senior pipeline add relative to a leakage-free (train-only) protocol?
* **RQ2 (heterogeneity).** As client data become more non-IID (Dirichlet label/attack-type skew on CIC-IoT2023; natural device split on N-BaIoT), how much do naive local preprocessing pipelines lose, compared with federated-statistics preprocessing, for federated linear detectors?
* **RQ3 (cost and competitiveness).** What communication and compute overhead does the statistics exchange add, and does the resulting lightweight detector stay competitive with a federated MLP, a proximal (FedProx) optimiser and a centralised random forest?
