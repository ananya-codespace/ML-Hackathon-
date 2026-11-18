import time, numpy as np, pandas as pd, shapely
from shapely import STRtree
from loader import list_trips, load_trip, ROOT
from geo import RoadNetwork, to_xy, crs_for_track

rows, dists = [], []
for part in ["train", "student_test"]:
    for tid in list_trips(part):
        d = load_trip(part, tid)
        crs = crs_for_track(d["track"])
        t0 = time.time(); net = RoadNetwork(d["nodes"], d["arcs"], crs); build = time.time() - t0
        pts_xy = to_xy(d["track"].lon, d["track"].lat, crs)
        rec = dict(partition=part, trip_id=tid, crs=crs, n_arcs=net.n_arcs, build_s=round(build, 2),
                   finite=bool(np.isfinite(net.node_xy).all() and np.isfinite(pts_xy).all()))
        if part == "train":
            pts = shapely.points(pts_xy)
            rtree = STRtree(net.geoms[d["route"].to_numpy()])
            _, dist = rtree.query_nearest(pts, return_distance=True, all_matches=False)
            dists.append(dist)
            rec.update(p50=np.percentile(dist, 50), p95=np.percentile(dist, 95),
                       p99=np.percentile(dist, 99), max=dist.max())
        rows.append(rec)

res = pd.DataFrame(rows).round(1)
res.to_csv(ROOT / "results" / "projection_and_route_distance.csv", index=False)
print("all finite:", bool(res.finite.all()), "| zones used:", res.crs.nunique(), "| slowest build (s):", res.build_s.max())
a = np.concatenate(dists)
print("Metres from each GPS point to nearest reference-route arc (train, all points):")
print({q: round(float(np.percentile(a, q)), 1) for q in [50, 90, 95, 99, 99.9, 100]})
print("share within 20/30/50/100 m:", {r: round(float((a <= r).mean() * 100), 2) for r in [20, 30, 50, 100]})
print(res[res.partition == "train"].sort_values("max", ascending=False).head(6).to_string())