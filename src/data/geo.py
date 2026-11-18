import numpy as np, pandas as pd, shapely
from shapely import STRtree
from pyproj import Transformer

CRS = "EPSG:32632"  # UTM 32N: Darmstadt is at ~8.6E, inside zone 32 (6-12E)
_tf = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)

def to_xy(lon, lat):
    x, y = _tf.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.column_stack([x, y])

class RoadNetwork:
    def __init__(self, nodes, arcs):
        self.node_xy = to_xy(nodes["lon"].values, nodes["lat"].values)
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
        df = df.sort_values(["obs_idx", "dist"]).reset_index(drop=True)
        df["rank"] = df.groupby("obs_idx").cumcount()
        return df if k is None else df[df["rank"] < k].reset_index(drop=True)