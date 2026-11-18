import numpy as np, pandas as pd
from loader import list_trips, load_trip, ROOT
from geo import to_xy

LON, LAT = (5.0, 12.0), (47.0, 53.0)   # generous box around Darmstadt / Germany
rows = []
for part in ["train", "student_test"]:
    for tid in list_trips(part):
        d = load_trip(part, tid)
        nd, tr, ar = d["nodes"], d["track"], d["arcs"]
        bad_n = ~(nd.lon.between(*LON) & nd.lat.between(*LAT))
        bad_t = ~(tr.lon.between(*LON) & tr.lat.between(*LAT))
        nxy = to_xy(nd.lon, nd.lat)
        nonfinite = ~np.isfinite(nxy).all(axis=1)
        bad_arc = bad_n.values[ar.from_node.values] | bad_n.values[ar.to_node.values]
        route_bad = int(bad_arc[d["route"].values].sum()) if "route" in d else -1
        rows.append(dict(partition=part, trip_id=tid, n_nodes=len(nd),
                         bad_nodes=int(bad_n.sum()), nonfinite_nodes=int(nonfinite.sum()),
                         bad_track_pts=int(bad_t.sum()), arcs_touching_bad=int(bad_arc.sum()),
                         route_arcs_bad=route_bad,
                         node_lon_min=nd.lon.min(), node_lon_max=nd.lon.max(),
                         node_lat_min=nd.lat.min(), node_lat_max=nd.lat.max()))
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "coord_check.csv", index=False)
flag = df[(df.bad_nodes > 0) | (df.bad_track_pts > 0) | (df.nonfinite_nodes > 0)]
print("trips checked:", len(df), "| flagged:", len(flag))
print(flag.to_string())