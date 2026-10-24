"""
Six diagrams (Phase 4.5-A), drawn at their FINAL PRINT SIZE (fonts >= 4.6 pt at 1:1). Each mirrors the implemented code:
  D1 system architecture           -> code/fedstat/prep.py + fl.py           (single column, 3.45 in)
  D2 end-to-end pipeline           -> code/run_experiments.py                (double column, 7.0 x 1.0 in)
  D3 preprocessing flowchart       -> code/fedstat/data.py, prep.py          (half page width, 3.5 in)
  D4 model architectures           -> code/fedstat/models.py, fl.py::_mlp    (double column, 7.0 x 1.1 in)
  D5 senior vs proposed            -> code/fedstat/methods.py ('senior' vs 'fedstat')   (single column, 3.45 in)
  D6 algorithm flowchart           -> Algorithm 1 (03_proposed_method.md)    (half page width, 3.5 in)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from style import COL, save


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.grid(False)
    return fig, ax


def box(ax, x, y, w, h, text, fc="#FFFFFF", ec="#333333", fs=5.0, bold=False, ls="-", lw=0.7, tc="#111111"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.04", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            color=tc, linespacing=1.1)


def arrow(ax, p0, p1, color="#333333", ls="-", lw=0.7):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=5.5, lw=lw, color=color, ls=ls, shrinkA=0.5, shrinkB=0.5))


# ------------------------------------------------------------------------------------------------------ D1
def d1_architecture():
    fig, ax = canvas(3.45, 2.5)
    sx, sw = 0.55, 2.35
    box(ax, sx, 1.42, sw, 0.72, "Aggregation server\nsums client statistics -> Pearson, LDA,\nintersection F, global scaler (m, s)\nFedAvg: theta = sum_k (n_k/n) theta_k",
        fc=COL["box_server"], fs=4.8)
    cx = [0.05, 1.2, 2.35]
    for x, lab in zip(cx, ["Client 1", "Client k", "Client K"]):
        box(ax, x, 0.32, 1.05, 0.72, f"{lab}\nlocal flows D_k\n(never sent)\nstatistics + SGD", fc=COL["box_data"], fs=4.8)
    for i, x in enumerate(cx):
        xc = x + 0.525
        xs = sx + sw * (i + 1) / 4.0
        arrow(ax, (xc - 0.12, 1.04), (xs - 0.08, 1.42), color=COL["proposed"])
        arrow(ax, (xs + 0.08, 1.42), (xc + 0.12, 1.04), color=COL["senior"], ls="--")
    ax.text(0.03, 2.47, "Phase 1 (once): up n_kc, s_kc, M_kc;\ndown mu, then (F, m, s)", fontsize=4.6, color=COL["proposed"], va="top")
    ax.text(3.42, 2.47, "Phase 2 (T rounds):\nup theta_k, down theta^t", fontsize=4.6, color=COL["senior"], va="top", ha="right")
    ax.text(0.05, 0.15, "solid blue = upload      dashed orange = download", fontsize=4.6, color="#444")
    return fig


# ------------------------------------------------------------------------------------------------------ D2
def d2_pipeline():
    fig, ax = canvas(7.0, 1.0)
    steps = ["Data\nCIC-IoT2023\nN-BaIoT", "De-duplicate,\nseeded\nsample", "Split\n60/20/20", "Client partition\nIID / Dir(a) /\ndevice",
             "Federated\npreprocessing\n(statistics)", "Federated\ntraining\n(FedAvg)", "Evaluation\nval: tuning\ntest: once", "Outputs\ntables,\nfigures"]
    w, g = 0.76, 0.12
    cols = [COL["box_data"]] * 4 + [COL["box_proc"], COL["box_proc"], COL["box_server"], COL["box_good"]]
    for i, (t, c) in enumerate(zip(steps, cols)):
        x = 0.05 + i * (w + g)
        box(ax, x, 0.08, w, 0.66, t, fc=c, fs=5.0)
        if i:
            arrow(ax, (x - g + 0.005, 0.41), (x - 0.005, 0.41))
    ax.text(0.05, 0.88, "public real data", fontsize=5.2, color="#333", style="italic")
    ax.text(0.05 + 4 * (w + g), 0.88, "proposed FedStat-FS", fontsize=5.2, color=COL["proposed"], style="italic", fontweight="bold")
    ax.text(0.05 + 6 * (w + g), 0.88, "test split used once", fontsize=5.2, color=COL["senior"], style="italic")
    return fig


# ------------------------------------------------------------------------------------------------------ D3
def d3_preprocessing():
    fig, ax = canvas(3.5, 3.0)
    fs, bh, st = 5.0, 0.27, 0.36
    y = 2.7
    items = [("Raw flow records (real, public)", COL["box_data"]),
             ("Drop exact duplicate feature vectors (before any split)", COL["box_data"]),
             ("Stratified split 60 / 20 / 20", COL["box_data"]),
             ("Partition TRAIN rows over K clients (statistics: train only)", COL["box_data"]),
             ("Variance filter  v_j >= 0.01", COL["box_proc"])]
    for i, (t, c) in enumerate(items):
        box(ax, 0.2, y - i * st, 3.1, bh, t, fc=c, fs=fs)
        if i:
            arrow(ax, (1.75, y - i * st + st), (1.75, y - i * st + bh))
    y2 = y - 5 * st
    box(ax, 0.2, y2, 1.5, bh, "Pearson |rho_j| > 0.2", fc=COL["box_proc"], fs=fs)
    box(ax, 1.8, y2, 1.5, bh, "LDA top-20 |w_LDA|", fc=COL["box_proc"], fs=fs)
    arrow(ax, (1.2, y2 + st), (0.95, y2 + bh))
    arrow(ax, (2.3, y2 + st), (2.55, y2 + bh))
    box(ax, 0.2, y2 - st, 3.1, bh, "Intersection  F = F_P ∩ F_L", fc=COL["box_proc"], fs=fs)
    arrow(ax, (0.95, y2), (1.3, y2 - st + bh))
    arrow(ax, (2.55, y2), (2.2, y2 - st + bh))
    box(ax, 0.2, y2 - 2 * st, 3.1, bh, "Global standardisation (m_j, s_j) from aggregated statistics", fc=COL["box_proc"], fs=fs)
    arrow(ax, (1.75, y2 - st), (1.75, y2 - 2 * st + bh))
    box(ax, 0.2, y2 - 3 * st, 3.1, bh, "Model input  z = (x_F - m) / s", fc=COL["box_good"], fs=fs, bold=True)
    arrow(ax, (1.75, y2 - 2 * st), (1.75, y2 - 3 * st + bh))
    ax.set_ylim(y2 - 3 * st - 0.08, y + bh + 0.08)
    return fig


# ------------------------------------------------------------------------------------------------------ D4
def d4_models():
    fig, ax = canvas(7.0, 1.1)
    ax.text(0.05, 1.0, "(a) Linear SVM detector (federated)", fontsize=5.6, fontweight="bold")
    seq = [("input z\n(|F| = 16-17)", 0.05, 0.85, COL["box_data"]), ("f = w^T z + b\nw in R^|F|, b in R\n(|F|+1 parameters)", 1.0, 1.15, COL["box_proc"]),
           ("sign(f)\nDDoS / benign", 2.3, 0.9, COL["box_good"])]
    for t, x, w, c in seq:
        box(ax, x, 0.25, w, 0.55, t, fc=c, fs=5.0)
    arrow(ax, (0.9, 0.52), (1.0, 0.52))
    arrow(ax, (2.15, 0.52), (2.3, 0.52))
    ax.text(0.05, 0.08, "loss: hinge or squared hinge + L2 (+ prox term)", fontsize=4.8)
    ax.text(3.7, 1.0, "(b) Federated MLP baseline (Keras)", fontsize=5.6, fontweight="bold")
    layers = [("input\n(|F|)", 3.7, COL["box_data"]), ("Dense 100\nReLU", 4.55, COL["box_proc"]), ("Dense 50\nReLU", 5.4, COL["box_proc"]),
              ("Dense 1\nsigmoid", 6.25, COL["box_good"])]
    for t, x, c in layers:
        box(ax, x, 0.25, 0.68, 0.55, t, fc=c, fs=5.0)
    for x in (4.38, 5.23, 6.08):
        arrow(ax, (x, 0.52), (x + 0.17, 0.52))
    ax.text(3.7, 0.08, "shapes: (B,|F|) -> (B,100) -> (B,50) -> (B,1);  about 6.8k parameters", fontsize=4.8)
    return fig


# ------------------------------------------------------------------------------------------------------ D5
def d5_comparison():
    fig, ax = canvas(3.45, 2.75)
    ax.text(0.85, 2.65, "Senior pipeline (as released)", ha="center", fontsize=5.4, fontweight="bold", color=COL["senior"])
    ax.text(2.6, 2.65, "Proposed FedStat-FS", ha="center", fontsize=5.4, fontweight="bold", color=COL["proposed"])
    ax.plot([1.725, 1.725], [0.02, 2.6], color="#999", lw=0.5, ls=":")
    L = ["Pool ALL raw flows\n(train + test together)", "Variance / Pearson / LDA / scaler\nfitted on pooled data BEFORE split",
         "Random 80/20 split (duplicates\ncan straddle the split)", "Per client: NEW LinearSVC each\nround (global model unused)", "One size-weighted average\n= one-shot averaging"]
    R = ["Flows stay on the clients\n(rows never leave)", "De-duplicate, split, then fit ONLY on\ntraining shards (additive statistics)",
         "Exact Pearson ∩ LDA subset F,\nglobal scaler (m, s): 2 uploads", "Warm-start FedAvg / FedProx:\nglobal model -> local SGD -> avg", "T rounds; test split touched once"]
    bh, st = 0.37, 0.49
    for i, (t, t2) in enumerate(zip(L, R)):
        y = 2.15 - i * st
        box(ax, 0.03, y, 1.66, bh, t, fc=COL["box_bad"], fs=4.6)
        box(ax, 1.76, y, 1.66, bh, t2, fc=COL["box_good"], fs=4.6)
        if i:
            arrow(ax, (0.86, y + st), (0.86, y + bh))
            arrow(ax, (2.59, y + st), (2.59, y + bh))
    return fig


# ------------------------------------------------------------------------------------------------------ D6
def d6_algorithm():
    fig, ax = canvas(3.5, 3.0)
    fs, bh, st = 5.0, 0.27, 0.33
    xb, wb = 0.15, 2.85
    n = 8
    ys = [2.75 - i * st for i in range(n)]
    txt = ["Clients: send n_kc, s_kc   (Eq. 2)", "Server: mu_c, mu ; broadcast mu   (Eq. 3)", "Clients: send M_kc around mu   (Eq. 4)",
           "Server: v, rho, S_W ; F_1, F_P, F_L   (Eq. 5-8)", "Server: F = F_P ∩ F_L ; (m, s) ; broadcast   (Eq. 9-10)",
           "theta^0 = 0 ;  t = 0", "Each client: z = (x_F - m)/s ; E epochs SGD on H_k   (Eq. 11)",
           "Server: theta^{t+1} = sum_k (n_k/n) theta_k   (Eq. 12)"]
    for i, (y, t) in enumerate(zip(ys, txt)):
        box(ax, xb, y, wb, bh, t, fc=COL["box_proc"] if i < 5 else COL["box_server"], fs=fs)
    for i in range(n - 1):
        arrow(ax, (xb + wb / 2, ys[i]), (xb + wb / 2, ys[i + 1] + bh))
    xr = xb + wb + 0.2
    ax.plot([xb + wb, xr], [ys[7] + bh / 2, ys[7] + bh / 2], color="#333", lw=0.8)
    ax.plot([xr, xr], [ys[7] + bh / 2, ys[6] + bh / 2], color="#333", lw=0.8)
    arrow(ax, (xr, ys[6] + bh / 2), (xb + wb, ys[6] + bh / 2))
    ax.text(xr + 0.06, (ys[6] + ys[7]) / 2 + bh / 2, "t < T", fontsize=5, rotation=90, va="center")
    yl = ys[7] - st
    box(ax, xb + 0.3, yl, wb - 0.6, bh, "t = T:  return theta_T, (F, m, s)", fc=COL["box_good"], fs=fs, bold=True)
    arrow(ax, (xb + wb / 2, ys[7]), (xb + wb / 2, yl + bh))
    ax.set_ylim(yl - 0.08, ys[0] + bh + 0.08)
    return fig


if __name__ == "__main__":
    for name, fn in [("d1_system_architecture", d1_architecture), ("d2_pipeline_flowchart", d2_pipeline),
                     ("d3_preprocessing_flowchart", d3_preprocessing), ("d4_model_architecture", d4_models),
                     ("d5_senior_vs_proposed", d5_comparison), ("d6_algorithm_flowchart", d6_algorithm)]:
        save(fn(), name, where="diagrams")
        print("saved", name)
