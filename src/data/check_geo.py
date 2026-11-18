import time, numpy as np, pandas as pd, shapely
from loader import list_trips, load_trip, ROOT
from geo import RoadNetwork, to_xy

lon, lat, rows, dists = [], [], [], []
for part in ["train", "student_test"]:
    for tid in list_trips(part):
        d = load_trip(part, tid)
        lon += [d["track"].lon.min(), d["track"].lon.max(), d["nodes"].lon.min(), d["nodes"].lon.max()]
        lat += [d["track"].lat.min(), d["track"].lat.max(), d["nodes"].lat.min(), d["nodes"].lat.max()]
        if part != "train":
            continue
        t0 = time.time(); net = RoadNetwork(d["nodes"], d["arcs"]); build = time.time() - t0
        pts = shapely.points(to_xy(d["track"].lon, d["track"].lat))
        route_tree = shapely.STRtree(net.geoms[d["route"].to_numpy()])
        _, dist = route_tree.query_nearest(pts, return_distance=True, all_matches=False)
        dists.append(dist)
        rows.append(dict(trip_id=tid, n_arcs=net.n_arcs, build_s=round(build, 2),
                         p50=np.percentile(dist, 50), p95=np.percentile(dist, 95),
                         p99=np.percentile(dist, 99), max=dist.max()))
print("lon range %.3f..%.3f | lat range %.3f..%.3f (UTM32 needs lon 6..12)" % (min(lon), max(lon), min(lat), max(lat)))
res = pd.DataFrame(rows).round(1); res.to_csv(ROOT / "results" / "point_to_route_distance.csv", index=False)
a = np.concatenate(dists)
print("ALL points, metres to nearest reference-route arc:")
print({q: round(float(np.percentile(a, q)), 1) for q in [50, 90, 95, 99, 99.9, 100]})
print("share within 20/30/50/100 m:", {r: round(float((a <= r).mean() * 100, ), 2) for r in [20, 30, 50, 100]})
print("slowest build (s):", res.build_s.max())
print(res.sort_values("max", ascending=False).head(6).to_string())