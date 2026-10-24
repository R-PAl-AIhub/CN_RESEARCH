"""Evaluation metrics (Eq. 13-19 of the paper)."""
import numpy as np


def confusion(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(int)
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    return tp, tn, fp, fn


def scores(y, p) -> dict:
    tp, tn, fp, fn = confusion(y, p)
    eps = 1e-12
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    return dict(acc=(tp + tn) / max(tp + tn + fp + fn, 1), prec=prec, rec=rec, f1=f1,
                fpr=fp / (fp + tn + eps), fnr=fn / (fn + tp + eps), mcc=mcc, tp=tp, tn=tn, fp=fp, fn=fn)


def jaccard(a, b) -> float:
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / max(len(a | b), 1)
