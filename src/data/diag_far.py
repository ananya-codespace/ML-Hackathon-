import numpy as np, pandas as pd, shapely
from shapely import STRtree
from loader import list_trips, load_trip, ROOT
from geo import RoadNetwork, to_xy, crs_for_track

rows = []
for tid in list_trips("train"):
    d = load_trip("train", tid)
    crs = crs_for_track(d["track"])
    net = RoadNetwork(d["nodes"], d["arcs"], crs)
    p = to_xy(d["track"].lon, d["track"].lat, crs)
    r = d["route"].to_numpy()
    tree = STRtree(net.geoms[r])
    idx, dist = tree.query_nearest(shapely.points(p), return_distance=True, all_matches=False)
    j = r[idx[1]]
    s, e = net.start[j], net.end[j]
    v = e - s
    tt = np.clip(((p - s) * v).sum(1) / np.maximum((v * v).sum(1), 1e-9), 0, 1)
    off = p - (s + tt[:, None] * v)            # offset from nearest route point (metres)
    far = dist > 50
    n = len(p); pos = np.arange(n) / n
    dt = np.r_[0, np.diff(d["track"]["t"].to_numpy())]
    if far.sum() > 0:
        mean_off = off[far].mean(0)
        consistency = np.hypot(*mean_off) / dist[far].mean()   # ~1 = systematic shift, ~0 = scattered
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, far.astype(int), 0]) != 0))[::2]
        longest = int(runs.max())
    else:
        consistency, longest, mean_off = np.nan, 0, np.array([0, 0])
    rows.append(dict(trip_id=tid, n_pts=n, far50_pct=round(far.mean() * 100, 1),
                     far_in_first_last_10pct=round(((pos < .1) | (pos > .9))[far].mean() * 100, 1) if far.any() else 0,
                     far_after_gap_pct=round((dt[far] > 10).mean() * 100, 1) if far.any() else 0,
                     longest_far_run=longest, shift_consistency=round(consistency, 2),
                     mean_dx=round(mean_off[0], 1), mean_dy=round(mean_off[1], 1)))
df = pd.DataFrame(rows).sort_values("far50_pct", ascending=False)
df.to_csv(ROOT / "results" / "far_points_diagnosis.csv", index=False)
print((df.far50_pct > 5).sum(), "of 80 trips have >5% points further than 50 m")
print(df.head(15).to_string())