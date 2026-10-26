# FedStat-FS: privacy-consistent feature selection and scaling for federated linear DDoS detection in IoT

Autonomous research pipeline: base paper (Karunamurthy et al., *Sci. Rep.* 2025) -> senior paper/code (federated SVM with Pearson-LDA feature
selection, CIC-IoT2023) -> audit, gap, method, experiments on **real public data**, figures/tables and an IEEE conference paper.
Everything runs on **CPU only**, natively on **Windows**, inside `./.venv`, and every deep-learning component uses **TensorFlow/Keras**.

## Layout
| Path | Content |
|---|---|
| `01_analysis.md` ... `06_review_report.md`, `decisions_log.md` | pipeline documents (analysis, gaps, method, experiments log, visuals index, review, all decisions) |
| `code/fedstat/` | library: `prep.py` (federated statistics preprocessing), `models.py`/`fl.py` (TensorFlow linear SVM, FedAvg/FedProx, Keras MLP), `methods.py`, `data.py`, `partition.py`, `metrics.py` |
| `code/run_experiments.py` | tuning (validation only) -> final runs (test once) -> sensitivity |
| `code/plots/` | diagrams, tables, figures, LaTeX number macros (no hand-typed numbers) |
| `code/data/` | dataset download / pool building / literature probe / reference verification / bib generation |
| `code/senior_repro/` | the **unchanged** senior code (only `DATASET_PATH` edited) and its outputs on real data |
| `results/` | `raw/*.jsonl` (one JSON per run), `tuning/`, `tables/`, `literature/`, logs, `archive_v1/` (first pass, kept as evidence) |
| `figures/diagrams`, `figures/results` | PDF + SVG + 300-DPI PNG |
| `paper/` | `main.tex`, `main.pdf`, `main.md`, `references.bib` |
| `tools/tectonic.exe` | self-contained LaTeX engine (local, no system install) |

## Reproduce (Windows, Git Bash)
```bash
py -3.11 -m venv ./.venv                      # Python 3.10/3.11
source ./.venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt               # exact pinned versions (CPU TensorFlow 2.21)
python code/run_all.py                        # downloads real data (~3.5 GB), runs everything, builds tables/figures/paper
```
`code/run_all.py` is resumable (each step skips finished work). N-BaIoT attack files sit in nested `.rar` archives; extraction uses the
Windows-bundled `C:\Windows\System32\tar.exe` (bsdtar). Expect roughly 2-3 hours on an 8-core CPU for the full grid (10 seeds x 2 datasets).

## Troubleshooting (Windows)
* `pip install` of TensorFlow can fail with `OSError: [Errno 2] ... venv\Lib\site-packages\tensorflow\include\external\...` when the venv path is long (Windows MAX_PATH = 260). Either enable Windows long paths (https://pip.pypa.io/warnings/enable-long-paths) or keep the project/venv path short (a venv folder named `.venv` at this project's root works; a deeper folder did not).
* Build the paper with `tools/tectonic.exe main.tex` inside `paper/`, one process at a time.

## Data (real, public; nothing synthetic is used anywhere)
* **CIC-IoT2023** (Neto et al., *Sensors* 2023, doi:10.3390/s23135941). Official page https://www.unb.ca/cic/datasets/iotdataset-2023.html; downloaded
  through the ungated HF mirror `lacg030175/CIC-IoT-2023-neto-full` (46 features, declared CC-BY-4.0); row counts verified against the release.
* **N-BaIoT** (Meidan et al., *IEEE Pervasive Comput.* 2018, doi:10.24432/C5RC8J, CC BY 4.0), UCI id 442.

## Integrity notes
* The test split is used once per final run; all tuning uses validation splits only (3 corpora, 6 random-search trials, identical for every method).
* Every number in the paper is generated (`paper/numbers.tex`) from `results/`; references were verified against Crossref/arXiv (`results/literature/citation_verification.json`).
* The senior code's synthetic-data mode is never used.

## Base-paper comparison (added later)
Re-implementation of Karunamurthy et al. (2025): Chimp-optimisation feature selection + federated CNN, run under the same protocol
(choices for settings the paper does not report: `decisions_log.md`, D13).
```
python code/run_base.py coa   --workers 4     # feature subset per dataset and seed (training rows only)
python code/run_base.py tune  --workers 4     # same 6-trial budget as every other method
python code/run_base.py final --workers 4     # 10 test seeds, same partitions as results/raw/final_*.jsonl
python code/run_base.py cost
python code/plots/base_compare.py             # results/tables/base_*.csv, paper/numbers_base.tex, paper/tables/table9_base.tex
```
