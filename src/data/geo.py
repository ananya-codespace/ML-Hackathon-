import numpy as np, pandas as pd, shapely
from shapely import STRtree
from pyproj import Transformer

def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"

def crs_for_track(track):
    return utm_epsg(track["lon"].median(), track["lat"].median())

_tf_cache = {}
def to_xy(lon, lat, crs):
    if crs not in _tf_cache:
        _tf_cache[crs] = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = _tf_cache[crs].transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.column_stack([x, y])

class RoadNetwork:
    def __init__(self, nodes, arcs, crs):
        self.crs = crs
        self.node_xy = to_xy(nodes["lon"].values, nodes["lat"].values, crs)
        f, t = arcs["from_node"].values, arcs["to_node"].values
        self.from_node, self.to_node = f, t
        self.start, self.end = self.node_xy[f], self.node_xy[t]
        d = self.end - self.start
        self.n_arcs = len(arcs)
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.bearing = np.arctan2(d[:, 0], d[:, 1])  # radians, clockwise from north
        self.geoms = shapely.linestrings(np.stack([self.start, self.end], axis=1))
        self.tree = STRtree(self.geoms)

    def candidates(self, pts_xy, radius=50.0, k=None):
        """One row per (obs_idx, arc_id) within `radius` m; keeps the k nearest if k is set."""
        pts = shapely.points(pts_xy)
        obs, arc = self.tree.query(pts, predicate="dwithin", distance=radius)
        df = pd.DataFrame({"obs_idx": obs, "arc_id": arc})
        df["dist"] = shapely.distance(pts[obs], self.geoms[arc])
        # Two-way streets give two arcs at exactly the same distance, so break ties by arc_id
        # with a stable sort: otherwise `rank` and the top-k cut change between library versions.
        df = df.sort_values(["obs_idx", "dist", "arc_id"], kind="stable").reset_index(drop=True)
        df["rank"] = df.groupby("obs_idx").cumcount()
        return df if k is None else df[df["rank"] < k].reset_index(drop=True)