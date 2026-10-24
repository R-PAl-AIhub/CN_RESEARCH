"""One consistent colour scheme and font for every figure (Okabe-Ito colour-blind-safe palette)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIG_D = ROOT / "figures" / "diagrams"
FIG_R = ROOT / "figures" / "results"
TAB = ROOT / "results" / "tables"
for p in (FIG_D, FIG_R, TAB):
    p.mkdir(parents=True, exist_ok=True)

# method -> colour (used in every result figure)
COL = {
    "proposed": "#0072B2", "senior": "#D55E00", "senior_noleak": "#E69F00", "local": "#CC79A7", "central": "#009E73",
    "mlp": "#56B4E9", "prox": "#7F7F7F", "rf": "#000000", "neutral": "#BBBBBB", "dark": "#333333",
    "box_data": "#E8F1F8", "box_proc": "#FFF4D6", "box_server": "#DDEEDD", "box_bad": "#FBE3D6", "box_good": "#D7ECF7",
}
NAMES = {
    "central_svm": "Central SVM", "central_rf": "Central RF", "senior": "Senior (as released)", "senior_noleak": "Senior (no leak)",
    "local_prep_fedavg": "Local-prep FedAvg", "fedstat": "FedStat-FS (proposed)", "fedstat_prox": "FedStat-FS + FedProx",
    "fedstat_mlp": "FedAvg-MLP (fed. prep)", "abl_localscale": "A1: local scalers", "abl_nosel": "A2: no selection",
    "abl_oneshot": "A3: one-shot averaging", "abl_pearson": "A4a: Pearson only", "abl_lda": "A4b: LDA only",
}
METHOD_COL = {"central_svm": COL["central"], "central_rf": COL["rf"], "senior": COL["senior"], "senior_noleak": COL["senior_noleak"],
              "local_prep_fedavg": COL["local"], "fedstat": COL["proposed"], "fedstat_prox": COL["prox"], "fedstat_mlp": COL["mlp"]}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "svg.fonttype": "none", "axes.grid": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.4,
})


def save(fig, name, where="results", tight=True):
    """Save vector PDF + SVG and 300-DPI PNG."""
    d = FIG_D if where == "diagrams" else FIG_R
    for ext in ("pdf", "svg", "png"):
        fig.savefig(d / f"{name}.{ext}", bbox_inches="tight" if tight else None, pad_inches=0.03)
    plt.close(fig)
