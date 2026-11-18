import pandas as pd
from loader import list_trips, load_trip
from pathlib import Path

rows = []
for part in ["train", "student_test"]:
    for tid in list_trips(part):
        d = load_trip(part, tid)
        tr, nd, ar, rt = d["track"], d["nodes"], d["arcs"], d["route"]
        dt = tr["t"].diff().dropna()
        rows.append(dict(
            partition=part, trip_id=tid,
            n_points=len(tr), n_nodes=len(nd), n_arcs=len(ar), n_route_arcs=len(rt),
            missing=int(tr.isna().sum().sum() + nd.isna().sum().sum() + ar.isna().sum().sum() + rt.isna().sum()),
            arc_node_ids_valid=bool(ar.min().min() >= 0 and ar.max().max() < len(nd)),
            route_ids_valid=bool(rt.min() >= 0 and rt.max() < len(ar)),
            time_monotonic=bool((dt > 0).all()),
            pct_dt_1s=round((dt == 1).mean() * 100, 1),
            max_gap_s=dt.max(),
            lon_min=tr["lon"].min(), lon_max=tr["lon"].max(),
            lat_min=tr["lat"].min(), lat_max=tr["lat"].max(),
        ))
inv = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
inv.to_csv("results/dataset_inventory.csv", index=False)
print(inv.groupby("partition")[["n_points", "n_nodes", "n_arcs", "n_route_arcs", "max_gap_s"]].describe().T.to_string())
print(inv[["missing", "arc_node_ids_valid", "route_ids_valid", "time_monotonic"]].agg(["sum"]))
print(inv.sort_values("max_gap_s", ascending=False).head(5)[["partition", "trip_id", "n_points", "max_gap_s"]])