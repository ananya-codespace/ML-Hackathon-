"""Trip loading, per-trip projection to metres, and the directed road network.

Each trip folder is self-contained:
    <id>.track  lon lat t      (GPS observations)
    <id>.nodes  lon lat        (row index = node_id)
    <id>.arcs   from to        (row index = arc_id, directed)
    <id>.route  arc_id         (reference route, ordered) -- server-side / evaluation only

Trips are spread across the world, so a single projected CRS does not work.
Each trip gets its own azimuthal equidistant projection centred on the median
GPS position: distances from the centre are exact and distortion stays well
below GPS noise at the scale of one trip (tens of km).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import Transformer


@dataclass
class Trip:
    trip_id: str
    track_lonlat: np.ndarray  # (N, 2)
    times: np.ndarray  # (N,)
    nodes_lonlat: np.ndarray  # (M, 2)
    arcs: np.ndarray  # (A, 2) int
    route: np.ndarray | None  # (R,) int, None when not loaded
    track_xy: np.ndarray  # (N, 2) metres
    nodes_xy: np.ndarray  # (M, 2) metres
    transformer: Transformer


def _projector(lon0: float, lat0: float) -> Transformer:
    crs = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def load_trip(trip_dir: str | Path, with_route: bool = True) -> Trip:
    """Load one trip folder. Set with_route=False on the prediction path so the
    reference route can never leak into candidates, features or decoding."""
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = np.loadtxt(trip_dir / f"{tid}.track", ndmin=2)
    nodes = np.loadtxt(trip_dir / f"{tid}.nodes", ndmin=2)
    arcs = np.loadtxt(trip_dir / f"{tid}.arcs", dtype=np.int64, ndmin=2)
    route = None
    if with_route and (trip_dir / f"{tid}.route").exists():
        route = np.loadtxt(trip_dir / f"{tid}.route", dtype=np.int64, ndmin=1)

    lon0, lat0 = np.median(track[:, 0]), np.median(track[:, 1])
    tf = _projector(lon0, lat0)
    tx, ty = tf.transform(track[:, 0], track[:, 1])
    nx, ny = tf.transform(nodes[:, 0], nodes[:, 1])
    return Trip(
        trip_id=tid,
        track_lonlat=track[:, :2],
        times=track[:, 2],
        nodes_lonlat=nodes,
        arcs=arcs,
        route=route,
        track_xy=np.column_stack([tx, ty]),
        nodes_xy=np.column_stack([nx, ny]),
        transformer=tf,
    )


class RoadNetwork:
    """Directed arc graph. Arc a leads into arc b when dst[a] == src[b]."""

    def __init__(self, nodes_xy: np.ndarray, arcs: np.ndarray):
        self.nodes_xy = nodes_xy
        self.arcs = arcs
        self.src = arcs[:, 0]
        self.dst = arcs[:, 1]
        seg = nodes_xy[self.dst] - nodes_xy[self.src]
        self.length = np.hypot(seg[:, 0], seg[:, 1])
        n_nodes = len(nodes_xy)

        # CSR adjacency: outgoing arcs of each node
        order = np.argsort(self.src, kind="stable")
        ptr = np.searchsorted(self.src[order], np.arange(n_nodes + 1))
        # python lists are much faster than numpy scalars inside Dijkstra loops
        self._out = order.tolist()
        self._ptr = ptr.tolist()
        self._dst_l = self.dst.tolist()
        self._src_l = self.src.tolist()
        self._len_l = self.length.tolist()
        self._cache: dict[int, tuple[float, dict[int, float]]] = {}

    @classmethod
    def from_trip(cls, trip: Trip) -> "RoadNetwork":
        return cls(trip.nodes_xy, trip.arcs)

    # --- connectivity -------------------------------------------------
    def successors(self, arc: int) -> list[int]:
        n = self._dst_l[arc]
        return self._out[self._ptr[n] : self._ptr[n + 1]]

    def connected(self, a: int, b: int) -> bool:
        """True if b == a or the vehicle can drive from a straight into b."""
        return a == b or self._dst_l[a] == self._src_l[b]

    # --- shortest paths -----------------------------------------------
    def node_distances(self, source: int, limit: float) -> dict[int, float]:
        """Bounded Dijkstra from `source`; returns {node: metres} for nodes
        reachable within `limit`. Cached per source node."""
        hit = self._cache.get(source)
        if hit is not None and hit[0] >= limit:
            return hit[1]
        limit = math.ceil(limit / 100.0) * 100.0  # round up so the cache is reused
        out, ptr, dst, ln = self._out, self._ptr, self._dst_l, self._len_l
        dist = {source: 0.0}
        heap = [(0.0, source)]
        done = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for k in range(ptr[u], ptr[u + 1]):
                a = out[k]
                v = dst[a]
                nd = d + ln[a]
                if nd <= limit and nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        self._cache[source] = (limit, dist)
        return dist

    def route_distance(self, a: int, off_a: float, b: int, off_b: float,
                       limit: float, same_arc_tol: float = 5.0) -> float:
        """Driving distance from position off_a on arc a to position off_b on
        arc b (offsets in metres from each arc's start). inf if > limit."""
        if a == b and off_b >= off_a - same_arc_tol:
            return max(off_b - off_a, 0.0)
        rest = self._len_l[a] - off_a
        budget = limit - rest - off_b
        if budget < 0:
            return math.inf
        d = self.node_distances(self._dst_l[a], budget).get(self._src_l[b])
        if d is None or d > budget:
            return math.inf
        return rest + d + off_b
