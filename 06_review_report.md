# 06 - Self-review and integrity report

Procedure: `academic-paper-reviewer` (full mode) was applied as a five-seat, role-separated panel plus an editorial synthesis, run on the draft
`paper/main.tex` (IEEE conference, 8 pages). The panel is simulated by the same model that wrote the paper, so it is **not independent**
(the skill itself says role separation is not a claim of independent error processes). Its value is a structured adversarial pass; the
hard checks in Part C are deterministic scripts, not opinion.

## A. Panel reports (condensed) and disposition of every finding

### Seat 1 - Journal-Fit Reviewer
* **F1.1 Novelty is thin.** The proposed layer is an exact re-computation of pooled preprocessing from moments; StatAvg already shares means/variances. *Valid.* The ablation A2 (no selection, global scaler) is essentially StatAvg and matches or beats FedStat-FS. **Fixed:** title changed from a method-centred claim to "Exact Federated Preprocessing ... What Shared Statistics Buy"; the Discussion now states that exactness is an algebraic identity, that A2 is StatAvg-like, and that the added value over StatAvg is the audited exact selection *whose benefit was not shown*.
* **F1.2 Fit / significance for the venue.** Two testbed datasets, binary task. *Accepted as limitation* (Threats/Limitations).
* Strengths noted: honest negative findings; reproducibility package; audit of a released pipeline.

### Seat 2 - Methodology Reviewer
* **F2.1 Holm over 54 tests makes the headline contrast (Local-prep vs FedStat-FS, Dir(0.1)) "not significant" although every seed favours FedStat-FS (raw p = floor 0.002).** *Valid concern in both directions.* **Fixed transparently:** the primary correction stays global (pre-specified), and a **post-hoc, labelled** within-comparison Holm (six regimes) is reported next to it (0.012); the text says the effect is significant only under the less conservative family. No p-value was altered.
* **F2.2 Tuning on validation splits of seeds 0-2 while test corpora of other seeds share the pool.** *Valid, small.* Stated in Threats (Internal). The test split of a corpus is never used to tune or select.
* **F2.3 First-pass tuning on a single seed produced diverging step sizes; ablations inherited a mismatched lr.** *Valid and already found by us before submission.* **Fixed:** protocol v2 (3-seed tuning, wider lr range, separate tuning for ablations with other feature counts); first-pass results archived in `results/archive_v1/`, logged as D8; the paper uses v2 only.
* **F2.4 N-BaIoT means/stds are dominated by failed seeds.** *Valid.* Failures are counted explicitly (F1 < 0.9) and diagnosed with exact solvers on the **training** split (`results/diag_selection_failure.json`); medians and failure counts are in the CSV/macros.
* **F2.5 Timing/memory noise.** *Valid.* Cost table uses cold runs in fresh processes, median of three; RAM column flagged as noisy in the caption.
* **F2.6 10 seeds limit Wilcoxon resolution (floor 0.002).** Stated in the paper.

### Seat 3 - Domain Reviewer
* **F3.1 Baseline coverage:** no FedBN/StatAvg *code* comparison. *Partly accepted:* A2 reproduces StatAvg's mechanism (global mean/std) inside the same pipeline; FedBN is not applicable to a linear model. Stated.
* **F3.2 A citation supported more than it says** ("lightweight design suited to constrained gateways" attached to Cortes and Vapnik). **Fixed:** the citation now supports only "linear SVM".
* **F3.3 Devine et al. (2025) claims a federated-SVM data-preparation approach; full text could not be accessed.** *Accepted:* recorded in `02_research_gaps.md`; novelty is worded "to the best of our knowledge".
* **F3.4 Literature numbers.** Only numbers stated in cited abstracts/papers are quoted (Table I "Reported result"), flagged as not comparable.

### Seat 4 - Perspective Reviewer
* **F4.1 "Privacy-consistent" is structural, not measured;** the d x d second-moment upload reveals class-conditional covariances. *Valid.* Stated in Discussion and Threats (construct); secure aggregation / DP are named as future work, not claimed.
* **F4.2 Deployment relevance:** statistics upload (33.8 KB/client for d = 46, 208.5 KB for d = 115) exceeds the whole model traffic of the linear detector; grows as d^2. Reported in RQ3, not hidden.
* **F4.3 Practical guidance** for practitioners (shared standardiser; Pearson-only or none unless a sufficiency check passes; train-only fits; de-duplicate) added to the Discussion.

### Seat 5 - Devil's Advocate
* **DA1 "Exactness verified 60/60" is a tautology.** *Agreed;* reframed as an implementation/numerical check (Discussion).
* **DA2 "Naive local preprocessing" is a strawman.** *Partly:* per-client scaling is common in practice, and the decisive comparison against a shared-statistics standardiser is A2 (StatAvg-like); both are reported. The finding that *selection* does not help is itself the counter-argument the paper makes against its own method.
* **DA3 "No leakage optimism" could be an artefact of a weak linear model / large sample.** *Possible;* stated as "no measurable optimism here", not as a general claim.
* **DA4 The selection-instability finding rests on N-BaIoT and 10 seeds.** *Possible;* the failure is reproduced with exact solvers, but the mechanism is not tested and the paper no longer speculates about it (the mechanism sentence was removed to keep every statement evidence-backed).

### Editorial synthesis
Decision (simulated): **Major revision -> after the fixes above, "accept with minor changes"** in a workshop/conference context. Remaining
weaknesses that cannot be fixed inside this study: only two datasets/binary task; privacy not measured; the proposed selection brings no measured benefit;
linear model below MLP by 0.14-0.47 F1 points on CIC-IoT2023.

## B. Two-stage revision log
Stage 1 fixes: F1.1, F2.1, F2.3 (before review), F3.2, DA1, F4.3. Stage 2 (re-check after edits): paper recompiled (8 pages), macros/tables/figures regenerated, visuals index and numeric audit re-run - no regression.

## C. Integrity pass (deterministic checks and evidence)

| Check | Result | Evidence |
|---|---|---|
| Every citation exists, authors/year correct | **21 of 23 verified automatically** against Crossref/arXiv (title fragment + first author + year). Exceptions: `yang2021` verified manually by title/DOI (Crossref stores given/family names swapped; corrected to Qin Yang and Masaaki Kondo, cross-checked in OpenAlex); `senior2026` is the unpublished manuscript supplied with the task and cannot be looked up (flagged in the bib) | `results/literature/citation_verification.json`, `code/data/verify_refs.py`, `code/data/make_bib.py` |
| Citation count 20-30 | **23** unique keys cited (22 verified + the senior manuscript); `paper/references.bib` holds exactly these | `grep \cite` count; 34 -> 23 entries after pruning |
| Citations support their claims | Reviewed against abstracts/titles; one over-attribution fixed (F3.2); Devine full text inaccessible (recorded) | `02_research_gaps.md`, `results/literature/abstracts.json` |
| Every number in the paper matches result files | All experimental numbers are LaTeX macros generated from `results/` (`paper/numbers.tex`, 2.2k definitions); remaining literals audited one by one: design constants, dataset facts, cited numbers; the derived ones (feature counts, MLP gap) were converted to macros | `results/numeric_audit.txt`, `code/plots/make_macros.py` |
| Figures/tables/equations numbered, captioned, referenced, matched to code, listed in the index | **All checks passed** (this pass found and fixed 5 unreferenced equations, 3 unreferenced figures) | `05_visuals_index.md`, `code/plots/make_visuals_index.py` |
| Every RQ explicitly answered | RQ1-RQ3 have italic "Answer:" sentences in Results and a summary in Discussion | `paper/sec_results.tex` |
| Baselines tuned fairly; test never used for tuning | Same random-search budget (6 trials, same stream, 3 validation corpora) for every tuned method; RF grid of 6; ablations A1/A3 inherit, A2/A4 tuned; `tune` uses `eval_split="val"`, `final` uses `"test"` once. One caveat (pool overlap across seeds) disclosed | `code/run_experiments.py`, `results/tuning/` |
| >= 2 datasets | CIC-IoT2023 and N-BaIoT | Tables II/IV |
| Inside the locked domain | Federated IDS/DDoS detection for IoT throughout | `01_analysis.md`, `02_research_gaps.md` |
| No synthetic/fake data | Only real releases; the senior code's synthetic mode was never used; partitions are of real rows | `decisions_log.md` D3-D6, `code/senior_repro/config.py` (USE_REAL_DATASET=True) |
| Originality (no copied sentences) | No verbatim run of >= 7 words against the base paper, senior paper and 8 cited abstracts; two 6-word matches are boilerplate ("the authors declare no competing interests") and a quoted term | `results/originality_check.txt` |
| Venue template and page limit; required statements | IEEEtran conference class, **8 pages** including references; data/code availability, ethics, conflict of interest, funding, CRediT, AI-use disclosure present; author/affiliation/funding are placeholders | `paper/main.pdf` |
| Runs on CPU in local `./.venv`, TensorFlow for deep learning, fresh venv reproduces | CUDA disabled in every script; all DL code Keras/TF; sklearn only for the senior LinearSVC re-run and the RF ceiling. Fresh-venv result: **passed** - a venv built from `requirements.txt` imports TF 2.21.0, passes the gradient unit test and reproduces the smoke run (a first attempt at a deeper folder failed only because of Windows MAX_PATH; see decisions D12) | `results/logs/fresh_venv_check.log` |

### Known residual issues (not hidden)
1. The Pearson-LDA intersection is the senior paper's selection rule; its instability on N-BaIoT is reported, not repaired (a repair would be a post-hoc method change).
2. `senior2026` cannot be machine-verified; the base paper's citation of its own baselines ([29]) was not chased.
3. The reviewer panel is self-simulated; an external human review is still needed before submission.
4. Fonts: the offline TeX bundle lacks the IEEE Times fonts; TeX Gyre Termes (a Times clone) is used, with the official `IEEEtran` class. A submission build with the venue's own TeX Live may shift line breaks slightly.
