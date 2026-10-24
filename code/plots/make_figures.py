"""
Result figures (Phase 4.5-D): every plot is generated from results/raw/*.jsonl; nothing is typed by hand.
  fig_curves       training / validation loss and accuracy per round (FedStat-FS, seed 0)
  fig_bar_f1       bar chart of test F1 (all methods x conditions) + error-rate view (log scale)
  fig_confusion    confusion matrices (summed over seeds)
  fig_ablation     ablation (error = 1-F1, log scale)
  fig_sensitivity  hyper-parameter sensitivity (Pearson tau, LDA top-k, rounds), validation split
  fig_error        error analysis: per attack category error rate
  fig_selection    agreement between locally voted and federated selections
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis_lib import CONDS, COND_LABEL, DS_LABEL, MAIN, ABL, load_final, load_sens, cell, mean_std
from style import save, COL, NAMES, METHOD_COL

COLS = [(ds, c) for ds, cs in CONDS.items() for c in cs]
XL = [f"{'CIC' if ds == 'ciciot' else 'NBa'}\n{COND_LABEL[c]}" for ds, c in COLS]
ABL_COL = {"fedstat": COL["proposed"], "abl_localscale": COL["local"], "abl_nosel": "#999999", "abl_oneshot": COL["senior_noleak"],
           "abl_pearson": COL["central"], "abl_lda": "#8C564B"}


def fig_curves(df):
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.55))
    for (ds, cond), ls, lab in [(("ciciot", "iid"), "-", "CIC IID"), (("ciciot", "dir0.1"), "--", "CIC Dir(0.1)"), (("nbaiot", "device"), ":", "NBa device")]:
        r = df[(df.dataset == ds) & (df.cond == cond) & (df.method == "fedstat") & (df.seed == 0)]
        if not len(r) or not isinstance(r.iloc[0]["curves"], list):
            continue
        h = pd.DataFrame(r.iloc[0]["curves"])
        ax[0].plot(h["round"], h["train_loss"], ls, color=COL["proposed"], label=lab)
        ax[1].plot(h["round"], h["val_loss"], ls, color=COL["senior"], label=lab)
        ax[2].plot(h["round"], h["val_acc"], ls, color=COL["central"], label=lab)
    for a, t, y in zip(ax, ["Training loss (local, mean over clients)", "Validation hinge loss", "Validation accuracy"], ["loss", "hinge loss", "accuracy"]):
        a.set_title(t, fontsize=6.6)
        a.set_xlabel("communication round")
        a.set_ylabel(y)
    ax[0].legend(frameon=False)
    ax[2].set_ylim(0.95, 1.0)
    fig.tight_layout()
    save(fig, "fig_curves")


def fig_bar_f1(df):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3))
    w = 0.8 / len(MAIN)
    for i, m in enumerate(MAIN):
        mu = [mean_std(cell(df, ds, c, m))[0] for ds, c in COLS]
        sd = [mean_std(cell(df, ds, c, m))[1] for ds, c in COLS]
        ax[0].bar(np.arange(len(COLS)) + (i - len(MAIN) / 2 + 0.5) * w, mu, w, yerr=sd, color=METHOD_COL[m], label=NAMES[m], error_kw=dict(lw=0.5, capsize=1))
        ax[1].plot(np.arange(len(COLS)), 1 - np.array(mu), "o-", color=METHOD_COL[m], ms=3, lw=0.9, label=NAMES[m])
    ax[0].set_ylim(0.3, 1.005)
    ax[0].set_ylabel("test F1")
    ax[0].set_xticks(range(len(COLS)))
    ax[0].set_xticklabels(XL)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, frameon=False, loc="lower center", fontsize=5.8, bbox_to_anchor=(0.5, -0.03))
    ax[0].set_title("(a) Test F1, mean $\\pm$ std over seeds", fontsize=7)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("error  1 - F1  (log)")
    ax[1].set_xticks(range(len(COLS)))
    ax[1].set_xticklabels(XL)
    ax[1].set_title("(b) Error 1-F1 (log scale)", fontsize=7)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fig_bar_f1")


def fig_confusion(df):
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.55))
    items = [("fedstat", "ciciot", "dir0.1"), ("local_prep_fedavg", "ciciot", "dir0.1"), ("fedstat", "nbaiot", "device"), ("local_prep_fedavg", "nbaiot", "device")]
    for a, (m, ds, c) in zip(ax, items):
        d = df[(df.method == m) & (df.dataset == ds) & (df.cond == c)]
        tn, fp, fn, tp = d.tn.sum(), d.fp.sum(), d.fn.sum(), d.tp.sum()
        M = np.array([[tn, fp], [fn, tp]], float)
        R = M / M.sum(1, keepdims=True)
        a.imshow(R, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{R[i, j] * 100:.1f}%\n({int(M[i, j]):,})", ha="center", va="center", fontsize=5.8, color="white" if R[i, j] > 0.5 else "black")
        a.set_xticks([0, 1])
        a.set_yticks([0, 1])
        a.set_xticklabels(["benign", "attack"])
        a.set_yticklabels(["benign", "attack"])
        a.set_xlabel("predicted")
        a.set_ylabel("actual")
        a.set_title(f"{NAMES[m].split(' (')[0]}\n{'CIC' if ds == 'ciciot' else 'NBa'} {COND_LABEL[c]}", fontsize=6.3)
        a.grid(False)
    fig.tight_layout()
    save(fig, "fig_confusion")


def fig_ablation(df):
    fig, ax = plt.subplots(figsize=(4.6, 1.9))
    w = 0.8 / len(ABL)
    for i, m in enumerate(ABL):
        mu = np.array([mean_std(cell(df, ds, c, m))[0] for ds, c in COLS])
        ax.bar(np.arange(len(COLS)) + (i - len(ABL) / 2 + 0.5) * w, 1 - mu, w, color=ABL_COL[m], label=NAMES[m])
    ax.set_yscale("log")
    ax.set_ylabel("test error  1 - F1  (log)")
    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels(XL)
    ax.legend(ncol=3, frameon=False, fontsize=5.8)
    fig.tight_layout()
    save(fig, "fig_ablation")


def fig_sensitivity(sens):
    if not len(sens):
        return
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.55))
    for a, p, lab in zip(ax, ["tau", "lda_top", "rounds"], ["Pearson threshold $\\tau$", "LDA top-$k$", "rounds $T$"]):
        for m, colr, mk in [("fedstat", COL["proposed"], "o"), ("local_prep_fedavg", COL["local"], "s")]:
            d = sens[(sens.method == m) & (sens.param == p)].groupby("value").f1.agg(["mean", "std"]).reset_index()
            a.errorbar(d["value"], d["mean"], yerr=d["std"], fmt=mk + "-", color=colr, ms=3, lw=0.9, capsize=1.5, label=NAMES[m].split(" (")[0])
        a.set_xlabel(lab)
        a.set_ylabel("validation F1")
        if p == "rounds":
            a.set_xscale("log")
    ax[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig_sensitivity")


def _percat(df, ds, cond, method):
    tot = {}
    for _, r in df[(df.dataset == ds) & (df.cond == cond) & (df.method == method)].iterrows():
        for k, v in r["per_cat"].items():
            n0, e0 = tot.get(k, (0, 0.0))
            tot[k] = (n0 + v["n"], e0 + v["n"] * v["err"])
    return {k: (e / n if n else np.nan, n) for k, (n, e) in tot.items()}


def fig_error(df):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3))
    meths = ["fedstat", "local_prep_fedavg", "senior", "central_rf"]
    for a, (ds, c) in zip(ax, [("ciciot", "dir0.3"), ("nbaiot", "device")]):
        pcs = {m: _percat(df, ds, c, m) for m in meths}
        cats = sorted(pcs["fedstat"].keys(), key=lambda k: -pcs["fedstat"][k][0])
        y = np.arange(len(cats))
        w = 0.8 / len(meths)
        for i, m in enumerate(meths):
            a.barh(y + (i - len(meths) / 2 + 0.5) * w, [pcs[m].get(k, (np.nan, 0))[0] for k in cats], w, color=METHOD_COL[m], label=NAMES[m])
        a.set_yticks(y)
        a.set_yticklabels([f"{k.replace('DDoS-', '').replace('BenignTraffic', 'benign').replace('benign_traffic', 'benign')} (n={pcs['fedstat'][k][1]:,})" for k in cats], fontsize=5.2)
        a.invert_yaxis()
        a.set_xscale("symlog", linthresh=1e-3)
        a.set_xlabel("error rate per class (symlog)")
        a.set_title(f"{'CIC-IoT2023' if ds == 'ciciot' else 'N-BaIoT'}, {COND_LABEL[c]} (test, summed over seeds)", fontsize=6.6)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, frameon=False, fontsize=5.8, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig_error")


def fig_selection(df):
    fig, ax = plt.subplots(figsize=(2.4, 1.9))
    xs, mu, sd, nf, nl = [], [], [], [], []
    for j, (ds, c) in enumerate(COLS):
        a = df[(df.dataset == ds) & (df.cond == c) & (df.method == "fedstat")].sort_values("seed")
        b = df[(df.dataset == ds) & (df.cond == c) & (df.method == "local_prep_fedavg")].sort_values("seed")
        jac = [len(set(x) & set(y)) / max(len(set(x) | set(y)), 1) for x, y in zip(a["keep"], b["keep"])]
        mu.append(np.mean(jac))
        sd.append(np.std(jac, ddof=1) if len(jac) > 1 else 0)
        nf.append(a.n_feat.mean())
        nl.append(b.n_feat.mean())
    x = np.arange(len(COLS))
    ax.bar(x, mu, 0.6, yerr=sd, color=COL["local"], error_kw=dict(lw=0.6, capsize=1.5))
    for i in range(len(COLS)):
        ax.text(i, 0.03, f"{nl[i]:.0f}\nvs {nf[i]:.0f}", ha="center", fontsize=5.2, color="white" if mu[i] > 0.2 else "black")
    ax.set_xticks(x)
    ax.set_xticklabels(XL, fontsize=4.4)
    ax.set_ylabel("Jaccard(local vote, federated)")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save(fig, "fig_selection")


if __name__ == "__main__":
    df = load_final()
    sens = load_sens()
    fig_curves(df)
    fig_bar_f1(df)
    fig_confusion(df)
    fig_ablation(df)
    fig_sensitivity(sens)
    fig_error(df)
    fig_selection(df)
    print("figures written")
