"""
Tables I-VIII (Phase 4.5-C): CSV in results/tables/ + LaTeX (booktabs) in results/tables/*.tex.
Every experimental number is computed from results/raw/*.jsonl, results/tuning/*.json, results/cost_profile.json.
Literature numbers (Tables VI, VIII) are quoted from the cited sources and marked as such.
"""
import sys
import json
import platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_lib import (ROOT, CONDS, COND_LABEL, DS_LABEL, MAIN, ABL, load_final, cell, mean_std, significance_table, fmt_p)
from style import TAB, NAMES

TEX = TAB


def esc(s):
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")


def write(name, df, caption, label, colfmt, star=False, size="\\scriptsize", note=None, raw=False):
    df.to_csv(TAB / f"{name}.csv", index=False)
    body = ["\\begin{table*}[t]" if star else "\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
            "\\setlength{\\tabcolsep}{2.4pt}",
            f"\\begin{{tabular}}{{{colfmt}}}", "\\toprule"]
    body.append(" & ".join(df.columns) + " \\\\")
    body.append("\\midrule")
    for _, r in df.iterrows():
        body.append(" & ".join(str(v) if raw else esc(v) for v in r.values) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    if note:
        body.append(f"\\\\[1pt]{{\\scriptsize {note}}}")
    body.append("\\end{table*}" if star else "\\end{table}")
    (TEX / f"{name}.tex").write_text("\n".join(body), encoding="utf-8")


def bold_best(vals, higher=True):
    """vals: list of (mean, std) -> list of latex strings, best mean in bold."""
    means = [m for m, _ in vals]
    best = np.nanmax(means) if higher else np.nanmin(means)
    out = []
    for m, s in vals:
        t = f"{m:.4f}\\,$\\pm$\\,{s:.4f}"
        out.append(f"\\textbf{{{t}}}" if np.isclose(m, best, atol=5e-5) else t)
    return out


def main():
    df = load_final()
    n_seeds = df.seed.nunique()
    (ROOT / "results" / "tables").mkdir(exist_ok=True, parents=True)

    # ------------------------------------------------------------------------------------------ Table I datasets
    from fedstat import data
    rows = []
    for ds, npc, K in (("ciciot", 100_000, 10), ("nbaiot", 8_000, 9)):
        c = data.build_corpus(ds, seed=0, n_per_class=npc)
        rep = c["dedup"]
        pool_before = rep["rows_before"]
        y = c["y"]
        sp = {k: len(v) for k, v in c["idx"].items()}
        rows.append({
            "Dataset": DS_LABEL[ds],
            "Source / licence": "unb.ca/cic (mirror: HF lacg030175/CIC-IoT-2023-neto-full; CC-BY-4.0) \\cite{neto2023}" if ds == "ciciot"
            else "UCI id 442, doi:10.24432/C5RC8J; CC BY 4.0 \\cite{meidan2018}",
            "Task / classes (release size)": "benign vs 12 DDoS types (46.7M flows)" if ds == "ciciot" else "benign vs 8 flood files (7.06M flows, 9 devices)",
            "Features": len(c["feat"]), "Pool rows": f"{pool_before:,}", "After de-dup": f"{rep['rows_after']:,}",
            "Corpus (seed 0)": f"{len(y):,} ({(y == 1).mean() * 100:.0f}\\% attack)",
            "Train/val/test": f"{sp['train']:,}/{sp['val']:,}/{sp['test']:,}", "Clients": f"{K}"})
    t1 = pd.DataFrame(rows)
    write("table1_datasets", t1, "Datasets (real, public). The pool is a seeded random sample of the release; exact duplicates are removed before any split; the corpus is a seeded stratified sample split 60/20/20.",
          "tab:datasets", "p{1.0cm}p{3.6cm}p{2.6cm}p{0.7cm}p{0.9cm}p{0.9cm}p{1.6cm}p{2.5cm}p{0.5cm}", star=True, raw=True)

    # ------------------------------------------------------------------------------------------ Table II setup
    import tensorflow as tf, sklearn, scipy
    tuning = {}
    for ds, cs in CONDS.items():
        for c in cs:
            tuning[(ds, c)] = json.load(open(ROOT / "results" / "tuning" / f"{ds}_{c}.json"))
    envrows = [("Platform", "Windows 11 Home (build 10.0.26200), native; AMD Ryzen 7 7435HS (8 cores / 16 threads); 23.7 GB RAM; CPU only"),
               ("Software", f"Python {platform.python_version()}, TensorFlow {tf.__version__} / Keras {tf.keras.__version__}, scikit-learn {sklearn.__version__}, NumPy {np.__version__}, SciPy {scipy.__version__}"),
               ("Splits, seeds", f"60/20/20 stratified; {n_seeds} seeds (corpus sample, split, partition, initialisation)"),
               ("Preprocessing", "variance $\\ge$ 0.01; Pearson $|\\rho|>0.2$; LDA top-20; intersection; standardisation"),
               ("FL protocol", "T = 20 rounds, E = 2 local epochs, batch 64, plain SGD, lr/(1+r) decay, all clients each round"),
               ("Tuning (all methods)", "random search, 6 trials, same random stream; objective = mean validation F1 over the corpora of seeds 0--2; test never used"),
               ("Search space: SVM family", "lr $\\in$ [0.005, 0.3] (log), C $\\in$ [0.3, 100] (log), loss $\\in$ \\{hinge, sq. hinge\\}; FedProx: $\\mu_p$ $\\in$ [1e-3, 1] (log)"),
               ("Search space: MLP / RF", "MLP: lr $\\in$ [0.005, 0.3], L2 $\\in$ [1e-5, 1e-2] (log); RF: trees $\\in$ \\{50,100\\} $\\times$ depth $\\in$ \\{10,20,30\\} (6-point grid)"),
               ("Senior baseline search", "sklearn LinearSVC: C $\\in$ [0.3, 100] (log), loss $\\in$ \\{hinge, sq. hinge\\}, max\\_iter 1000")]
    t2 = pd.DataFrame(envrows, columns=["Item", "Setting"])
    write("table2_setup", t2, "Experimental setup (CPU only) and tuning search spaces. Tuned values per dataset/condition are in results/tables/tuned\\_hyperparameters.csv.",
          "tab:setup", "p{2.6cm}p{14.4cm}", star=True, size="\\tiny", raw=True)
    trows = []
    for (ds, c), t in tuning.items():
        for m, v in t.items():
            trows.append(dict(dataset=ds, cond=c, method=m, best_val_f1=v["best_val_f1"], **{k: x for k, x in v["best_hp"].items()}))
    pd.DataFrame(trows).to_csv(TAB / "tuned_hyperparameters.csv", index=False)

    # ------------------------------------------------------------------------------------------ Table III main
    cols = [(ds, c) for ds, cs in CONDS.items() for c in cs]
    hdr = ["Method"] + [f"{DS_LABEL[ds][:3] if ds=='ciciot' else 'NBa'} {COND_LABEL[c]}" for ds, c in cols]
    grid = {m: [mean_std(cell(df, ds, c, m)) for ds, c in cols] for m in MAIN}
    out = {m: [] for m in MAIN}
    for j in range(len(cols)):
        col = bold_best([grid[m][j] for m in MAIN])
        for m, s in zip(MAIN, col):
            out[m].append(s)
    t3 = pd.DataFrame([[esc(NAMES[m])] + out[m] for m in MAIN], columns=hdr)
    write("table3_main_f1", t3, f"Test F1 (attack class), mean $\\pm$ std over {n_seeds} seeds; best in bold. ``Senior'' fits preprocessing on all rows before the split; Central rows are not federated.",
          "tab:main", "l" + "c" * len(cols), star=True, size="\\tiny", raw=True)
    full = []
    for ds, c in cols:
        for m in MAIN:
            r = {"dataset": ds, "cond": c, "method": m}
            for k in ("f1", "rec", "prec", "acc", "fpr", "mcc"):
                mu, sd = mean_std(cell(df, ds, c, m, k))
                r[f"{k}_mean"], r[f"{k}_std"] = mu, sd
            full.append(r)
    pd.DataFrame(full).to_csv(TAB / "table3_main_all_metrics.csv", index=False)

    # ------------------------------------------------------------------------------------------ Table IV ablation
    grid = {m: [mean_std(cell(df, ds, c, m)) for ds, c in cols] for m in ABL}
    out = {m: [] for m in ABL}
    for j in range(len(cols)):
        col = bold_best([grid[m][j] for m in ABL])
        for m, s in zip(ABL, col):
            out[m].append(s)
    nf = {m: np.mean([df[(df.method == m) & (df.dataset == ds) & (df.cond == cols[0][1])].n_feat.mean() for ds in ("ciciot", "nbaiot")]) for m in ABL}
    t4 = pd.DataFrame([[esc(NAMES[m])] + out[m] + [f"{np.mean([df[(df.method == m) & (df.dataset == 'ciciot')].n_feat.mean()]):.0f} / {df[(df.method == m) & (df.dataset == 'nbaiot')].n_feat.mean():.0f}"] for m in ABL],
                      columns=hdr + ["\\#feat (CIC / NBa)"])
    write("table4_ablation", t4, "Ablations (test F1, mean $\\pm$ std): A1 per-client scalers; A2 no selection; A3 one-shot averaging; A4 Pearson-only / LDA-only. A1 and A3 inherit the tuned FedStat-FS hyper-parameters; A2 and A4 are tuned separately.",
          "tab:ablation", "l" + "c" * len(cols) + "c", star=True, size="\\tiny", raw=True)
    pd.DataFrame([dict(method=m, dataset=ds, cond=c, f1_mean=mean_std(cell(df, ds, c, m))[0], f1_std=mean_std(cell(df, ds, c, m))[1]) for m in ABL for ds, c in cols]
                 ).to_csv(TAB / "table4_ablation_numeric.csv", index=False)

    # ------------------------------------------------------------------------------------------ Table V significance
    sig = significance_table(df)
    sig.to_csv(TAB / "table5_significance_all.csv", index=False)
    key = [("ciciot", "iid"), ("ciciot", "dir0.1"), ("nbaiot", "device")]
    rows = []
    for A, B, why in __import__("analysis_lib").PAIRS:
        if B in ("fedstat_prox", "abl_localscale"):        # kept in the CSV; omitted from the printed table for space
            continue
        r = {"Comparison (A vs B)": esc(f"{NAMES[A]} vs {NAMES[B]}")}
        for ds, c in key:
            s = sig[(sig.dataset == ds) & (sig.cond == c) & (sig.A == A) & (sig.B == B)]
            if len(s):
                s = s.iloc[0]
                tag = f"{DS_LABEL[ds][:3]} {COND_LABEL[c]}"
                r[f"{tag}: $\\Delta$F1"] = f"{s['diff']:+.4f}"
                r[f"{tag}: $p$ / Holm"] = f"{fmt_p(s['p_w'])}/{fmt_p(s['p_w_holm'])}"
        rows.append(r)
    t5 = pd.DataFrame(rows)
    write("table5_significance", t5, f"Paired tests over {n_seeds} seeds (test F1): $\\Delta$F1 = A $-$ B; $p$ = Wilcoxon signed-rank, raw / Holm over all {len(sig)} tests (full list with intervals in the CSV; smallest attainable $p$ = {2 / 2 ** n_seeds:.4f}).",
          "tab:sig", "lcccccc", star=True, size="\\tiny", raw=True)

    # ------------------------------------------------------------------------------------------ Table VII cost
    cp = json.load(open(ROOT / "results" / "cost_profile.json")) if (ROOT / "results" / "cost_profile.json").exists() else {}
    rows = []
    for m in ("central_svm", "central_rf", "senior", "fedstat", "fedstat_prox", "fedstat_mlp"):
        if m not in cp:
            continue
        r = cp[m]
        rows.append({"Method": esc(NAMES[m]), "Train (s)": f"{r['train_sec']:.1f}", "Preproc. (s)": f"{r['prep_sec']:.2f}", "Inference (ms/1k flows)": f"{r['infer_ms_per_1k']:.2f}",
                     "Parameters": f"{r['n_params']:,}", "Peak extra RAM (MB)": f"{r['peak_incremental_mb']:.0f}",
                     "Model traffic (KB, whole run)": f"{r['comm_model_bytes'] / 1024:.1f}", "Stats upload (KB/client)": f"{(r['comm_prep_bytes'] / 10 / 1024):.1f}" if r['comm_prep_bytes'] else "0"})
    t7 = pd.DataFrame(rows)
    write("table7_cost", t7, "Computational cost on CPU (CIC-IoT2023, seed 0, IID clients; whole simulated federation on one machine; cold run in a fresh process, median of three; the RAM column is noisy, roughly $\pm$100 MB, and dominated by data copies). "
          "Traffic counts float32 model up/down-loads over all rounds; the statistics upload is float64, once.",
          "tab:cost", "lccccccc", star=True, size="\\tiny", raw=True)

    # ------------------------------------------------------------------------------------------ Table VI literature (quoted)
    lit = pd.DataFrame([
        ["Karunamurthy et al. \\cite{karunamurthy2025}", "MQTTset (6 classes)", "FL-CNN + chimp-optimisation selection", "Acc. 95.59\\%, macro-F1 94.04\\% (their Table 10)", "IID split"],
        ["Mohamed \\& Kundu (senior) \\cite{senior2026}", "CIC-IoT2023 (binary)", "Fed. linear SVM + Pearson$\\cap$LDA", "F1 98.81\\%, Acc. 98.84\\% (reported); 99.66\\% F1 in our re-run of their code", "IID, pooled pre-split preprocessing"],
        ["Bouzinis et al. (StatAvg) \\cite{bouzinis2025}", "TON-IoT, CIC-IoT-2023", "Global normalisation from shared feature statistics (deep FL)", "Accuracy gain 4--17\\% over normalisation baselines (abstract)", "non-IID features"],
        ["Beshah et al. (FedDWC) \\cite{fedDWC2025}", "IoTID20", "Dynamic weighted clustered FL", "Acc. +1.9/+1.31/+1.01\\% over FedAvg/FedProx/IFCA (abstract)", "non-IID, 200 devices"],
        ["Nguyen et al. (Fed-FeRe) \\cite{nguyen2026}", "IoT IDS benchmarks", "Chi-square selection + GRU", "+3.06\\% precision, $\\approx$2\\% less traffic vs FedAvg (abstract)", "non-IID"],
        ["This work (FedStat-FS)", "CIC-IoT2023 / N-BaIoT", "Federated-statistics preprocessing + FedAvg linear SVM", "see Table~\\ref{tab:main}", "IID, Dirichlet, device"]],
        columns=["Study", "Dataset", "Technique", "Reported result (as stated by the source)", "Data regime"])
    write("table6_literature", lit, "Comparison with the literature. Numbers are quoted from the cited sources; datasets, splits and metrics differ, so the rows are not directly comparable.",
          "tab:lit", "p{2.6cm}p{1.9cm}p{2.8cm}p{4.2cm}p{2.0cm}", star=True, size="\\tiny", raw=True)

    # ------------------------------------------------------------------------------------------ Table VIII related work features
    Y, N, P = "yes", "no", "part"
    rel = pd.DataFrame([
        ["Karunamurthy et al. \\cite{karunamurthy2025}", "MQTTset", "CNN", "wrapper (COA)", Y, N, N, N],
        ["Devine et al. \\cite{devine2025}", "IoT DDoS data", "SVM/RF/ANN", "data preparation", Y, N, N, P],
        ["Yang \\& Kondo \\cite{yang2021}", "NSL-KDD", "on-device NN", "greedy per attack", Y, N, N, N],
        ["FD-IDS \\cite{fdids2025}", "Edge-IIoT, N-BaIoT", "NN + KD + prox", "mutual information", Y, Y, N, N],
        ["StatAvg \\cite{bouzinis2025}", "TON-IoT, CIC-IoT-2023", "deep FL", "-- (normalisation)", Y, Y, P, N],
        ["Fed-FeRe \\cite{nguyen2026}", "IoT IDS data", "GRU", "chi-square", Y, Y, N, N],
        ["FedDWC \\cite{fedDWC2025}", "IoTID20", "clustered FL", "--", Y, Y, N, N],
        ["Senior \\cite{senior2026}", "CIC-IoT2023", "linear SVM", "Pearson$\\cap$LDA", Y, N, N, N],
        ["FedStat-FS (ours)", "CIC-IoT2023, N-BaIoT", "linear SVM (+MLP/prox baselines)", "Pearson$\\cap$LDA, exact", Y, Y, Y, Y]],
        columns=["Method", "Dataset(s)", "Model", "Feature selection", "Federated", "Non-IID", "Fed. preprocessing statistics", "Leakage-free protocol + audit"])
    rel.to_csv(TAB / "table8_related.csv", index=False)
    quoted = {"Karunamurthy": "acc. 95.59\\%, macro-F1 94.04\\%", "Senior": "F1 98.81\\% (99.66\\% in our re-run)", "StatAvg": "acc. gain 4--17\\% over normalisation baselines",
              "FedDWC": "acc. +1.9/+1.31/+1.01\\% over FedAvg/FedProx/IFCA", "Fed-FeRe": "+3.06\\% precision, $\\approx$2\\% less traffic than FedAvg",
              "FedStat-FS": "Table~\\ref{tab:main}"}
    comb = rel.copy()
    comb["Reported result (quoted)"] = [next((v for k, v in quoted.items() if k in name), "not stated in abstract") for name in rel["Method"]]
    comb.columns = ["Method", "Dataset(s)", "Model", "Selection", "Fed.", "Non-IID", "Fed. prep. stats", "Leakage-free audit", "Reported result (quoted)"]
    write("table8_related", comb, "Related-work comparison and reported results (as far as the cited abstracts/papers state; ``part'' = partial or not stated). Reported numbers are quoted from each source; datasets, splits and metrics differ, so rows are not directly comparable.",
          "tab:related", "p{2.3cm}p{1.9cm}p{1.6cm}p{1.6cm}p{0.6cm}p{0.8cm}p{1.0cm}p{1.1cm}p{3.6cm}", star=True, size="\\tiny", raw=True)
    tex = (TEX / "table8_related.tex").read_text(encoding="utf-8").replace("\\label{tab:related}", "\\label{tab:related}\\label{tab:lit}")
    (TEX / "table8_related.tex").write_text(tex, encoding="utf-8")
    print("tables written:", sorted(p.name for p in TAB.glob("table*.tex")))


if __name__ == "__main__":
    main()
