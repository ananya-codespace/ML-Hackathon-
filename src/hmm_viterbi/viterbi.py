"""HMM map-matching decoded with Viterbi.

Hidden state at step t : one of the candidate arcs of GPS point t, together
                         with the point's projected offset along that arc.
Emission  log E_t(j)   : w * log p_t(j), p from the candidate classifier.
Transition log T(i->j) : -|d_route(i, j) - d_gps| / beta, where d_route is the
                         driving distance from point t-1's position on arc i to
                         point t's position on arc j (bounded Dijkstra) and
                         d_gps is the straight-line distance between the two
                         GPS points. Unreachable (or > limit) -> -inf.
Recurrence             : S_t(j) = max_i [S_{t-1}(i) + log T(i->j)] + log E_t(j)
                         back_t(j) = argmax_i [...]; backtrack from argmax S_T.

Edge cases
  * GPS jitter while stopped makes the projected offset move backwards along
    the same arc; up to same_arc_tol metres of that is treated as standing
    still instead of as an (unreachable) U-turn.
  * point with no candidates -> skipped by the HMM, then given the arc decoded
    for the nearest earlier point (forward fill; backward fill at the start).
  * time gap > max_gap_s     -> chain break: the driving distance is unknown,
    so every state restarts from the best previous score plus its emission.
  * no reachable transition  -> same chain break (GPS jump, network hole, or a
    reference route that leaves the supplied map).
All arithmetic is in log space; scores are re-centred every step.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .network import RoadNetwork, Trip


@dataclass
class HMMParams:
    beta: float = 5.0  # metres; transition scale
    emission_weight: float = 1.0
    max_gap_s: float = 60.0
    detour_factor: float = 2.0  # search budget = factor * d_gps + slack
    detour_slack: float = 100.0
    same_arc_tol: float = 30.0  # metres of backward GPS jitter allowed on one arc
    min_prob: float = 1e-12

    def to_dict(self) -> dict:
        return asdict(self)


def _ensure_offsets(cands: pd.DataFrame, trip: Trip, net: RoadNetwork) -> np.ndarray:
    if "offset" in cands:
        return cands["offset"].to_numpy(dtype=float)
    obs = cands["obs_idx"].to_numpy()
    arc = cands["arc_id"].to_numpy()
    a = net.nodes_xy[net.src[arc]]
    ab = net.nodes_xy[net.dst[arc]] - a
    l2 = np.einsum("ij,ij->i", ab, ab)
    t = np.einsum("ij,ij->i", trip.track_xy[obs] - a, ab) / np.where(l2 > 0, l2, 1)
    return np.clip(t, 0, 1) * np.sqrt(l2)


def decode(cands: pd.DataFrame, probs: np.ndarray, trip: Trip, net: RoadNetwork,
           params: HMMParams | None = None) -> tuple[np.ndarray, dict]:
    """Return (one predicted arc_id per GPS point, info dict).

    `cands` needs columns obs_idx, arc_id (offset optional); `probs` is the
    classifier probability for each row, in the same order. Row order is free.
    """
    p = params or HMMParams()
    n_obs = len(trip.track_xy)
    offsets = _ensure_offsets(cands, trip, net)
    order = np.argsort(cands["obs_idx"].to_numpy(), kind="stable")
    obs = cands["obs_idx"].to_numpy()[order]
    arcs = cands["arc_id"].to_numpy()[order]
    offs = offsets[order]
    loge = p.emission_weight * np.log(np.maximum(np.asarray(probs, dtype=float)[order], p.min_prob))

    steps = np.unique(obs)
    bounds = np.searchsorted(obs, np.r_[steps, n_obs + 1])
    xy, times = trip.track_xy, trip.times

    step_arcs: list[np.ndarray] = []
    backs: list[np.ndarray | None] = []
    n_breaks_gap = n_breaks_unreach = 0
    score = None
    prev_t = prev_arcs = prev_offs = None

    for s, t in enumerate(steps):
        lo, hi = bounds[s], bounds[s + 1]
        c_arcs, c_offs, c_loge = arcs[lo:hi], offs[lo:hi], loge[lo:hi]
        if score is None:
            score, back = c_loge.copy(), None
        else:
            gap = times[t] - times[prev_t]
            if gap > p.max_gap_s:
                trans = None
                n_breaks_gap += 1
            else:
                d_gps = float(math.hypot(*(xy[t] - xy[prev_t])))
                limit = p.detour_factor * d_gps + p.detour_slack
                trans = np.full((len(prev_arcs), len(c_arcs)), -np.inf)
                for i, (ai, oi) in enumerate(zip(prev_arcs.tolist(), prev_offs.tolist())):
                    for j, (aj, oj) in enumerate(zip(c_arcs.tolist(), c_offs.tolist())):
                        d = net.route_distance(ai, oi, aj, oj, limit, p.same_arc_tol)
                        if d < math.inf:
                            trans[i, j] = -abs(d - d_gps) / p.beta
            if trans is not None:
                m = score[:, None] + trans
                back = np.argmax(m, axis=0)
                best = m[back, np.arange(len(c_arcs))]
                if np.isneginf(best).all():
                    trans = None
                    n_breaks_unreach += 1
                else:
                    score = best + c_loge
            if trans is None:  # chain break: restart from best previous state
                back = np.full(len(c_arcs), int(np.argmax(score)))
                score = score.max() + c_loge
        score = score - score[np.isfinite(score)].max()
        step_arcs.append(c_arcs)
        backs.append(back)
        prev_t, prev_arcs, prev_offs = t, c_arcs, c_offs

    pred = np.full(n_obs, -1, dtype=np.int64)
    if len(steps):
        k = int(np.argmax(score))
        for s in range(len(steps) - 1, -1, -1):
            pred[steps[s]] = step_arcs[s][k]
            if backs[s] is not None:
                k = int(backs[s][k])
        # fill points with no candidates
        filled = pd.Series(np.where(pred >= 0, pred, np.nan)).ffill().bfill()
        pred = filled.to_numpy().astype(np.int64)

    info = {
        "n_steps": int(len(steps)),
        "n_no_candidate": int(n_obs - len(steps)),
        "breaks_time_gap": n_breaks_gap,
        "breaks_unreachable": n_breaks_unreach,
    }
    return pred, info
