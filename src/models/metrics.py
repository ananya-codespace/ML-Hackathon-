"""All metrics in one place so every model is scored identically."""
import numpy as np
import pandas as pd

from . import config as C


def log_loss_binary(y, p):
    p = np.clip(p, C.EPS, 1 - C.EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _confusion(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def _f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def prf_at(y, p, thr):
    """Row-level precision/recall/F1 of the positive class and macro F1 (mean of both classes)."""
    tp, fp, fn, tn = _confusion(y, (p >= thr).astype(int))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1_pos, f1_neg = _f1(tp, fp, fn), _f1(tn, fn, fp)
    return dict(threshold=float(thr), precision=prec, recall=rec, f1=f1_pos,
                macro_f1=(f1_pos + f1_neg) / 2, tp=tp, fp=fp, fn=fn, tn=tn)


def best_threshold(y, p, grid=None):
    """Threshold maximising macro F1 on the given (out-of-fold, TRAIN-only) predictions.

    Vectorised through one sort, so it is cheap enough to call inside every
    Bayesian-optimisation trial.
    """
    grid = np.linspace(0.02, 0.98, 97) if grid is None else grid
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order].astype(np.int64)
    ctp = np.concatenate([[0], np.cumsum(ys)])
    P, N = int(ys.sum()), len(ys) - int(ys.sum())
    k = np.searchsorted(-ps, -grid, side="right")            # rows with p >= thr
    tp = ctp[k]; fp = k - tp; fn = P - tp; tn = N - fp
    f1p = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    f1n = 2 * tn / np.maximum(2 * tn + fn + fp, 1)
    macro = (f1p + f1n) / 2
    i = int(np.argmax(macro))
    return float(grid[i]), float(macro[i])


def top1_predictions(meta: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    """Highest-probability candidate for every GPS observation.

    Returns one row per (trip_id, obs_idx): predicted arc_id, whether it is the
    reference arc (`correct`), and whether the reference arc was in the candidate
    set at all (`has_true`).
    """
    d = meta[C.KEY_COLS + [C.LABEL_COL]].reset_index(drop=True).copy()
    d["prob"] = prob
    g = d.groupby(["trip_id", "obs_idx"], sort=True)
    idx = g["prob"].idxmax()
    pred = d.loc[idx.to_numpy(), C.KEY_COLS + [C.LABEL_COL, "prob"]].reset_index(drop=True)
    pred = pred.rename(columns={C.LABEL_COL: "correct", "prob": "top_prob"})
    pred["has_true"] = g[C.LABEL_COL].max().to_numpy().astype(int)
    return pred


def obs_metrics(pred: pd.DataFrame):
    n = len(pred)
    n_has = int(pred.has_true.sum())
    return dict(
        n_obs=n,
        top1_accuracy=float(pred.correct.sum() / n),                         # misses count as wrong
        top1_given_hit=float(pred.correct.sum() / n_has) if n_has else float("nan"),
        candidate_miss_rate=float(1 - n_has / n),
    )


def adjacency_continuity(pred: pd.DataFrame, arcs_by_trip: dict) -> float:
    """FALLBACK route-continuity: share of consecutive predicted arcs that are equal
    or head-to-tail connected.  `arcs_by_trip[trip_id]` is an (n_arcs, 2) int array
    (from_node, to_node).  Use C's official function when it is available."""
    ok = tot = 0
    for trip, g in pred.sort_values(["trip_id", "obs_idx"]).groupby("trip_id"):
        a = g.arc_id.to_numpy(); arcs = arcs_by_trip[trip]
        if len(a) < 2:
            continue
        cur, nxt = a[:-1], a[1:]
        ok += int(((cur == nxt) | (arcs[cur, 1] == arcs[nxt, 0])).sum()); tot += len(cur)
    return ok / tot if tot else float("nan")


def full_metrics(y, prob, meta, thr):
    out = prf_at(y, prob, thr)
    out["log_loss"] = log_loss_binary(y, prob)
    out.update(obs_metrics(top1_predictions(meta, prob)))
    return out
