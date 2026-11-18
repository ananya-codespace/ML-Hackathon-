import sys
from pathlib import Path
import numpy as np, pandas as pd, shapely
sys.path.append(str(Path(__file__).resolve().parents[1] / "data"))
from loader import DATA, load_trip
from geo import RoadNetwork, to_xy, crs_for_track
from labels import label_points

R, K = 75.0, 10
FEATURES = ["dist", "rank", "dist_min", "dist_gap", "n_cand", "t_along", "arc_len",
            "cos_head", "head_ok", "speed", "dt_prev", "dt_next", "dist_prev", "dist_next"]

def find_partition(trip_id):
    for p in DATA.iterdir():
        if (p / trip_id).is_dir():
            return p.name
    raise FileNotFoundError(trip_id)

def _kinematics(xy, t):
    n = len(xy); i = np.arange(n)
    prev, nxt = np.maximum(i - 1, 0), np.minimum(i + 1, n - 1)
    d = xy[nxt] - xy[prev]; disp = np.hypot(d[:, 0], d[:, 1]); span = t[nxt] - t[prev]
    speed = np.nan_to_num(disp / np.where(span > 0, span, np.nan))
    heading = np.arctan2(d[:, 0], d[:, 1])               # clockwise from north, same as arc bearing
    head_ok = (disp >= 3.0) & ((t - t[prev]) <= 10) & ((t[nxt] - t) <= 10)
    return prev, nxt, speed, heading, head_ok, t - t[prev], t[nxt] - t

def build_trip(trip_id, with_label=False, radius=R, k=K):
    """Returns (candidate_table, points_table or None). Labels are read ONLY if with_label=True."""
    part = find_partition(trip_id)
    d = load_trip(part, trip_id, with_route=with_label)
    crs = crs_for_track(d["track"])
    net = RoadNetwork(d["nodes"], d["arcs"], crs)
    xy = to_xy(d["track"].lon, d["track"].lat, crs); t = d["track"]["t"].to_numpy(float)
    n = len(xy)
    prev, nxt, speed, heading, head_ok, dt_prev, dt_next = _kinematics(xy, t)

    c = net.candidates(xy, radius=radius, k=k)
    o, a = c.obs_idx.values, c.arc_id.values
    s, v = net.start[a], net.end[a] - net.start[a]
    t_along = np.clip(((xy[o] - s) * v).sum(1) / np.maximum((v * v).sum(1), 1e-9), 0, 1)
    ang = np.abs((heading[o] - net.bearing[a] + np.pi) % (2 * np.pi) - np.pi)
    pts = shapely.points(xy)
    dist_min = c.groupby("obs_idx")["dist"].transform("min").values

    df = pd.DataFrame({
        "trip_id": trip_id, "obs_idx": o, "arc_id": a,
        "dist": c.dist.values, "rank": c["rank"].values, "dist_min": dist_min,
        "dist_gap": c.dist.values - dist_min, "n_cand": c.groupby("obs_idx")["dist"].transform("size").values,
        "t_along": t_along, "arc_len": net.length[a],
        "cos_head": np.where(head_ok[o], np.cos(ang), 0.0), "head_ok": head_ok[o].astype(float),
        "speed": speed[o], "dt_prev": dt_prev[o], "dt_next": dt_next[o],
        "dist_prev": shapely.distance(pts[prev[o]], net.geoms[a]),   # neighbouring-point context
        "dist_next": shapely.distance(pts[nxt[o]], net.geoms[a]),
    })
    df[FEATURES] = df[FEATURES].astype("float32")
    if not with_label:
        return df, None
    lab, ldist = label_points(net, xy, d["route"].to_numpy())
    df["y"] = (a == lab[o]).astype("int8")
    ptab = pd.DataFrame({"trip_id": trip_id, "obs_idx": np.arange(n), "label_arc_id": lab,
                         "label_dist_m": ldist, "n_cands": np.bincount(o, minlength=n),
                         "label_in_cands": np.bincount(o[df.y.values == 1], minlength=n) > 0})
    return df, ptab

def build_candidates(trip_id):          # the agreed interface; never touches labels
    return build_trip(trip_id, with_label=False)[0]