"""Route-level metrics shared by the model comparison (B) and the decoder (C)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .network import RoadNetwork


def route_continuity(pred: np.ndarray, net: RoadNetwork) -> float:
    """Share of consecutive GPS points whose predicted arcs are drivable in
    sequence: the same arc, or the second starts where the first ends.
    Points predicted as -1 (no prediction) are dropped first."""
    pred = np.asarray(pred)
    pred = pred[pred >= 0]
    if len(pred) < 2:
        return float("nan")
    a, b = pred[:-1], pred[1:]
    ok = (a == b) | (net.dst[a] == net.src[b])
    return float(ok.mean())


def transition_continuity(pred: np.ndarray, net: RoadNetwork) -> float:
    """Like route_continuity but only over points where the arc changes, so the
    long runs on the same arc do not hide jumps."""
    pred = np.asarray(pred)
    pred = pred[pred >= 0]
    a, b = pred[:-1], pred[1:]
    change = a != b
    if not change.any():
        return float("nan")
    return float((net.dst[a[change]] == net.src[b[change]]).mean())


def collapse(seq: np.ndarray) -> np.ndarray:
    seq = np.asarray(seq)
    seq = seq[seq >= 0]
    if len(seq) == 0:
        return seq
    return seq[np.r_[True, seq[1:] != seq[:-1]]]


def sequence_metrics(pred: np.ndarray, route: np.ndarray, net: RoadNetwork) -> dict:
    """Arc-set precision / recall / F1 (length-weighted) between the predicted
    route and the reference route."""
    p, r = set(collapse(pred).tolist()), set(np.asarray(route).tolist())
    inter = p & r
    lp = net.length[list(p)].sum() if p else 0.0
    lr = net.length[list(r)].sum() if r else 0.0
    li = net.length[list(inter)].sum() if inter else 0.0
    prec = li / lp if lp else 0.0
    rec = li / lr if lr else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}


def point_accuracy(pred: np.ndarray, labels: np.ndarray) -> float:
    return float((np.asarray(pred) == np.asarray(labels)).mean())


def candidate_miss_rate(cands: pd.DataFrame, labels: np.ndarray) -> float:
    """Share of GPS points whose reference arc is not among their candidates."""
    hit = cands["arc_id"].to_numpy() == labels[cands["obs_idx"].to_numpy()]
    covered = np.zeros(len(labels), dtype=bool)
    covered[cands["obs_idx"].to_numpy()[hit]] = True
    return float(1.0 - covered.mean())


def top1_from_probs(cands: pd.DataFrame, probs: np.ndarray, n_obs: int) -> np.ndarray:
    """Classifier-only prediction: highest-probability candidate per point."""
    df = cands[["obs_idx", "arc_id"]].assign(p=probs)
    best = df.loc[df.groupby("obs_idx")["p"].idxmax()]
    pred = np.full(n_obs, -1, dtype=np.int64)
    pred[best["obs_idx"].to_numpy()] = best["arc_id"].to_numpy()
    return pred
