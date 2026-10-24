# Decisions Log

Autonomous pipeline run. Started 2026-09-21. All decisions made without user input, per the run instructions.

## D0. Environment (native Windows, CPU only)
- OS: Windows 11 Home Single Language, build 10.0.26200
- CPU: AMD Ryzen 7 7435HS (8 cores / 16 threads)
- RAM: 23.7 GB
- Python: 3.11.9 (via `py -3.11`, from `D:\SOFTWARES\python.exe`); system default Python is 3.14, so the launcher was used
- venv: `./.venv` inside project root (verified: `where python` lists the .venv interpreter first)
- TensorFlow: 2.21.0 (Keras 3.15.1), CPU-only wheel; verified with a 64x64 matmul on CPU. Other pins in `requirements.txt` (scikit-learn 1.9.1, NumPy 2.4.6, SciPy 1.17.1, pandas 3.0.6).
- Timeline note: the whole run happened on 2026-09-21 in one session; the run instructions asked for no user questions and none were asked.
- CUDA_VISIBLE_DEVICES=-1 forced in every script; no GPU packages installed.

## D1. Skills found and used
Found (installed as the `academic-research-skills` plugin v3.21.1; names match by purpose):
- `academic-pipeline`  -> orchestration (read SKILL.md)
- `deep-research`      -> literature review / gap finding / fact-check
- `academic-paper`     -> drafting
- `academic-paper-reviewer` -> self-review
Decision: the `academic-pipeline` skill mandates user-confirmation checkpoints after every stage. The run
instructions explicitly forbid pausing to ask the user, so the checkpoints are treated as auto-approved and
logged here instead. All other standards of the skills (integrity checks, two-stage review, citation
verification) are followed.

## D2. Dataset access (initial probe)
- CIC-IoT2023 official page: https://www.unb.ca/cic/datasets/iotdataset-2023.html . The direct directory
  `cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/CSV/` now returns a generic UNB web page (no listing), so it
  cannot be scripted directly.
- An ungated public HuggingFace mirror of the original per-attack CSV files exists:
  https://huggingface.co/datasets/baalajimaestro/DDoS-CICIoT2023 (not gated, csv format). Will use it as
  transport only, and record both the original source and the mirror. Integrity of the mirror is checked
  against the official label names / feature count in Phase 4.

## D3. CIC-IoT2023 version (schema mismatch found and resolved)
- First mirror tried (`baalajimaestro/DDoS-CICIoT2023`, per-pcap CSVs) has only **39** features and no label column
  (label = file name). The senior paper/code use the **46-feature** release with a label column, so that mirror was
  DISCARDED (files deleted, downloader removed).
- Adopted: `lacg030175/CIC-IoT-2023-neto-full` (HF, declared CC-BY-4.0; derived from CIC's official 169-file release,
  46 features + `Label`). Integrity verified in `data/processed/ciciot_pool_summary.json`: 46,686,579 rows;
  Benign 1,098,195; DDoS 33,984,560 - identical to the counts published by Neto et al. (Sensors 2023).
- Original source recorded everywhere: https://www.unb.ca/cic/datasets/iotdataset-2023.html (CIC, UNB).
- `data/processed/ciciot_pool.parquet` = seeded Bernoulli sample (P_benign=0.35, P_ddos=0.0125, seed 42) of REAL rows,
  809,836 rows, float32 cast only. No row is generated or edited.

## D4. Phase-1 reproduction protocol
- The senior code is scikit-learn (not PyTorch; torch only appears inside a try/except in the seeding helper).
  Decision: the unchanged senior scripts are run as the "as-released" baseline (copy in `code/senior_repro/`,
  only `DATASET_PATH` edited, diff verified). Deep-learning pieces (ANN) and the federated linear SVM used for the
  proposed method are re-implemented in TensorFlow/Keras (`code/fedstat/`), verified against sklearn in
  `results/phase1_tf_port_check.json`.
- The senior loader reads CSVs sequentially and keeps the first rows found; the exact file order used by the senior is
  unknown, so a uniformly random 255k benign + 245k DDoS sample (seed 42) of the real pool was written to CSV and fed
  to the unchanged loader.
- sklearn LinearSVC(C=1) <-> mean-hinge SGD objective mapping: lambda = 1/(C*n) (fixed after a first TF run with
  lambda=1e-4 under-fit: F1 0.9918 vs 0.9963).

## D5. Second dataset
- N-BaIoT (UCI id 442, doi 10.24432/C5RC8J, CC BY 4.0; Meidan et al., IEEE Pervasive Computing 2018): real traffic of 9
  commercial IoT devices infected by Mirai/BASHLITE. Chosen because the 9 devices give a *natural* federated
  partition. The UCI server truncates the transfer at ~490 MB and has no range support, so download is retried in full.

## D6. N-BaIoT specifics
- UCI archive (1,772,922,927 bytes, sha256 64929678b081d8e579a8d7c488cf11cc588403f282d5fb065b4156edbd55de9b) holds `benign_traffic.csv` per device,
  but the attack files are nested `.rar` archives. A first pool build silently contained benign rows only (108k rows, 0 attacks) and was
  discarded; attack CSVs were then extracted with Windows `bsdtar` (`data/raw/nbaiot/extracted/`).
- Task = benign vs flood-type traffic. Kept: BASHLITE combo/junk/tcp/udp and Mirai ack/syn/udp/udpplain; the two `scan` classes are excluded
  (reconnaissance, outside the DDoS scope). Ennio Doorbell and Samsung Webcam have no Mirai files in the release (dataset design).
- Pool = exact uniform sample without replacement per (device, file): 12,000 benign / 3,000 per attack file, seed 42.

## D7. Federated-training budget (found by test on validation data; documented before final runs)
- First tuning pass (batch 1024, decay 0.3) left the federated SVM ~0.6 F1-points below the centralised SVM even under IID data. A controlled
  check on the validation split showed this was an *optimisation-budget* effect (number of SGD steps), not a preprocessing effect:
  batch 64 / 128 or 5 local epochs recover centralised accuracy. All federated linear methods therefore share: T=20 rounds, E=2 local epochs,
  batch 64, step decay lr/(1+r), with lr, C and loss tuned per method by identical random search. Tuning files from the first pass were deleted and re-run.
- Local epochs were fused into one `tf.function` with closed-form mini-batch gradients (equality with autodiff verified in `code/tests/test_gradients.py`).

## D8. First-pass results archived; tuning protocol made robust (documented BEFORE the re-run)
- First final pass (`results/archive_v1/`, kept as evidence) exposed two protocol weaknesses on N-BaIoT:
  (i) hyper-parameters tuned on ONE seed included aggressive step sizes (lr 0.26, squared hinge) that *diverged* on other corpora (|theta| ~ 4e11);
  the ablation `abl_nosel` (111 features) inherited an lr tuned for 13 features and collapsed on every device-split seed;
  (ii) for some seeds the Pearson-LDA *intersection* subset (19 features) is genuinely linearly insufficient (exact LinearSVC and LogisticRegression
  on the pooled TRAIN split reach only F1 = 0.63 with it, vs 0.9996 with Pearson-only or all 111 features) - this is a property of the senior's
  selection rule, reproduced exactly by the federated implementation, not a federated bug (exploratory diagnostics were in a scratch folder, later deleted; the reported diagnostic is `code/diagnostics_selection_failure.py` -> `results/diag_selection_failure.json`).
- Fix (applies to ALL methods equally): hyper-parameters are now chosen by the mean validation F1 over 3 corpora (seeds 0-2), lr range widened to
  [0.005, 0.3], and ablations whose feature count differs from FedStat-FS (no selection, Pearson-only, LDA-only) are tuned separately.
  The test split is still never used for any decision. The selection instability is kept and REPORTED, not hidden.
- No result of the first pass is used in the paper except to describe this decision.

## D9. Final experimental design (v2, the one reported)
- Datasets: CIC-IoT2023 (200k-flow corpora: 100k benign + 100k DDoS, 60/20/20; K=10; IID, Dirichlet 1.0/0.3/0.1) and N-BaIoT (<=8k per device and class; K=9 devices + IID control).
- Methods: 8 tuned (Central SVM/RF, Senior, Senior no-leak, Local-prep FedAvg, FedStat-FS, +FedProx, FedAvg-MLP) + ablations A1-A4. 10 seeds, test split used once per run.
- Tuning: random search, 6 trials, identical stream for every method, objective = mean validation F1 over corpora of seeds 0-2 (D8). Ablations A2/A4 tuned separately, A1/A3 inherit FedStat-FS.
- Statistics: paired Wilcoxon + 95% t-interval, Holm over all 54 tests (primary, fixed before seeing p-values) with a **post-hoc, labelled** per-comparison Holm as sensitivity (Phase 6, finding F2.1). No p-value or metric was edited.
- The senior claims were not "tuned toward": the reproduced senior F1 (0.9966) is reported as is, above the paper's 0.9881.

## D10. Paper production decisions
- No LaTeX distribution was installed; a project-local `tools/tectonic.exe` (self-contained XeTeX) builds the paper with the official `IEEEtran` class. Two Tectonic processes must never run concurrently (cache lock -> spurious "font not loadable" errors).
- The offline bundle has no Times/bold Latin Modern; `fontspec` + TeX Gyre Termes (a Times clone) restores bold and IEEE uppercase headings. `amssymb/amsfonts` and `\mathbf` were dropped (fonts unavailable offline).
- The first draft was 12 pages; brought to the 8-page limit by (in order): redrawing diagrams at print size (fonts kept >= 4.6 pt), merging the two literature tables into one, compacting figures/tables, cutting ~35% of prose, pruning the reference list from 33 to 23 (all uncited entries removed, incl. two that had been planned), setting the reference/statement sections in `\scriptsize`.
- Numbers in the paper are LaTeX macros generated from result files (`paper/numbers.tex`), never typed; `paper/main.md` is generated from the same sources.
- Bug found and fixed while doing this: `references.bib` initially printed "Y. Qin and M. Kondo" (Crossref swapped given/family names); corrected to Qin Yang / Masaaki Kondo.
- LaTeX was never written through shell heredocs after early mishaps (backslash sequences became control characters); the Write/Edit tools were used instead.

## D11. Phase 6 review-driven changes
Title re-framed (less method-centric), Discussion now states the algebraic-identity/StatAvg-equivalence caveat, one over-attributed citation fixed, five unreferenced equations and three figures referenced, post-hoc Holm sensitivity added. See `06_review_report.md`.

## D12. Reproducibility check
A fresh venv created from `requirements.txt` at a *deep* path failed on Windows (MAX_PATH 260: TensorFlow ships very long include paths; the project path `D:\RATUL\VIT\THIRD YEAR\COMPUTER NETWORKS\RESEARCH\My Research\` is already long). This is an environment limit, not a dependency problem; the check was repeated with a shorter in-project venv folder (`.vc`), result in `results/logs/fresh_venv_check.log`. README documents the caveat (enable Windows long paths or use a short project path).

### D12 result (fresh-environment reproduction)
`.vc` (fresh venv created from `requirements.txt`, Python 3.11.9): `pip install -r requirements.txt` succeeded; TensorFlow 2.21.0 / scikit-learn 1.9.1 / NumPy 2.4.6 import;
`code/tests/test_gradients.py` PASS; `code/smoke_test.py ciciot iid` ran end-to-end on CPU (Central SVM val F1 0.9958, FedStat-FS 0.9933, selection Jaccard 1.00).
Log: `results/logs/fresh_venv_check.log`. The temporary `.vc` folder was deleted afterwards.
