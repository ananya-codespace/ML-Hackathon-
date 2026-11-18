"""Candidate arcs per GPS point, and a distance-only stand-in classifier.

This is a minimal candidate generator so the decoder can be developed and
evaluated before the shared feature table (Person A) and the trained model
(Person B) are ready. `decode` only needs the columns obs_idx and arc_id, so
the real candidate table and `predict_proba` output drop straight in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from .network import RoadNetwork, Trip


def arc_lines(net: RoadNetwork) -> np.ndarray:
    coords = np.stack([net.nodes_xy[net.src], net.nodes_xy[net.dst]], axis=1)
    return shapely.linestrings(coords)


def build_candidates(trip: Trip, net: RoadNetwork, k: int = 8, radius: float = 50.0,
                     fallback_nearest: bool = True) -> pd.DataFrame:
    """One row per (GPS point, candidate arc): the k nearest arcs within
    `radius` metres. Points with nothing in range get their single nearest
    arc if `fallback_nearest`, otherwise no rows (the decoder handles that)."""
    lines = arc_lines(net)
    tree = STRtree(lines)
    pts = shapely.points(trip.track_xy)

    pi, ai = tree.query(pts, predicate="dwithin", distance=radius)
    if fallback_nearest:
        missing = np.setdiff1d(np.arange(len(pts)), pi)
        if len(missing):
            mi, ma = tree.query_nearest(pts[missing])
            pi = np.concatenate([pi, missing[mi]])
            ai = np.concatenate([ai, ma])

    dist = shapely.distance(pts[pi], lines[ai])
    order = np.lexsort((dist, pi))
    pi, ai, dist = pi[order], ai[order], dist[order]
    # rank within each point
    starts = np.r_[0, np.flatnonzero(np.diff(pi)) + 1]
    rank = np.arange(len(pi)) - np.repeat(starts, np.diff(np.r_[starts, len(pi)]))
    keep = rank < k
    pi, ai, dist, rank = pi[keep], ai[keep], dist[keep], rank[keep]
    offset = shapely.line_locate_point(lines[ai], pts[pi])

    return pd.DataFrame({
        "trip_id": trip.trip_id,
        "obs_idx": pi.astype(np.int64),
        "arc_id": ai.astype(np.int64),
        "dist": dist,
        "offset": offset,
        "rank": rank.astype(np.int64),
    })


def distance_probs(cands: pd.DataFrame, sigma: float = 10.0) -> np.ndarray:
    """Stand-in classifier: Gaussian GPS-error likelihood normalised over each
    point's candidates. Its argmax is simply the nearest arc."""
    logit = -0.5 * (cands["dist"].to_numpy() / sigma) ** 2
    logit = logit - cands.assign(_l=logit).groupby("obs_idx")["_l"].transform("max").to_numpy()
    e = np.exp(logit)
    z = cands.assign(_e=e).groupby("obs_idx")["_e"].transform("sum").to_numpy()
    return e / z
