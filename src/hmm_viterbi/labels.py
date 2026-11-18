"""Per-GPS-point reference labels from the ordered .route.

.route lists the arcs driven, in order, but says nothing about which GPS point
belongs to which arc. We align points to route arcs with a monotone dynamic
programme: point i gets route position j(i), with j non-decreasing in i, and
the total point-to-arc distance is minimised. Monotonicity is what makes loops
and roads driven twice come out right, unlike "nearest route arc".

Used for training labels and evaluation only -- never on the prediction path.
Limitation: it trusts .route; if .route itself is wrong (about 2% of
consecutive route arcs are disconnected in some trips) the labels inherit it.
"""
from __future__ import annotations

import numpy as np

from .network import RoadNetwork, Trip


def _point_to_segments(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    t = np.where(denom > 0, np.einsum("ij,ij->i", p - a, ab) / np.where(denom > 0, denom, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.hypot(*(p - proj).T)


def reference_labels(trip: Trip, net: RoadNetwork) -> tuple[np.ndarray, np.ndarray]:
    """Returns (label_arc_id per point, distance in metres to that arc)."""
    route = trip.route
    if route is None or len(route) == 0:
        raise ValueError(f"trip {trip.trip_id} has no route loaded")
    a = net.nodes_xy[net.src[route]]
    b = net.nodes_xy[net.dst[route]]
    n, r = len(trip.track_xy), len(route)
    idx = np.arange(r)
    back = np.empty((n, r), dtype=np.int16 if r < 32767 else np.int32)

    cost = _point_to_segments(trip.track_xy[0], a, b)
    back[0] = idx
    for i in range(1, n):
        pm = np.minimum.accumulate(cost)
        arg = np.maximum.accumulate(np.where(cost == pm, idx, 0))
        cost = _point_to_segments(trip.track_xy[i], a, b) + pm
        back[i] = arg

    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(cost))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    labels = route[pos]
    d = _point_to_segments(trip.track_xy, a[pos], b[pos])
    return labels, d
