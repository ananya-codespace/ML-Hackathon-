import numpy as np, pandas as pd
from loader import list_trips, load_trip, ROOT
from geo import RoadNetwork, to_xy, crs_for_track
from labels import label_points

R_MAX = 100
RS, KS = [25, 50, 75, 100], [5, 10, 15, 20, 30]
far = pd.read_csv(ROOT / "results" / "far_points_diagnosis.csv", dtype={"trip_id": str})
flagged = set(far.loc[far.far50_pct > 50, "trip_id"])
rows = []
for tid in list_trips("train"):
    d = load_trip("train", tid)
    crs = crs_for_track(d["track"])
    net = RoadNetwork(d["nodes"], d["arcs"], crs)
    p = to_xy(d["track"].lon, d["track"].lat, crs); n = len(p)
    lab, ldist = label_points(net, p, d["route"].to_numpy())
    cand = net.candidates(p, radius=R_MAX)
    m = cand.merge(pd.DataFrame({"obs_idx": np.arange(n), "arc_id": lab}), on=["obs_idx", "arc_id"])
    lrank = np.full(n, np.inf); lrank[m.obs_idx.values] = m["rank"].values
    for r in RS:
        cnt = np.bincount(cand.obs_idx.values[cand.dist.values <= r], minlength=n)
        for k in KS:
            miss = ~((ldist <= r) & (lrank < k))
            rows.append(dict(trip_id=tid, flagged=tid in flagged, r=r, k=k, n_pts=n,
                             misses=int(miss.sum()), n_rows=int(np.minimum(cnt, k).sum())))
res = pd.DataFrame(rows); res.to_csv(ROOT / "results" / "candidate_miss_grid_by_trip.csv", index=False)

def summarize(df):
    g = df.groupby(["r", "k"])[["n_pts", "misses", "n_rows"]].sum()
    g["miss_pct"] = (g.misses / g.n_pts * 100).round(2)
    g["rows_per_pt"] = (g.n_rows / g.n_pts).round(1)
    return g
clean = summarize(res[~res.flagged]); allt = summarize(res)
clean.to_csv(ROOT / "results" / "candidate_miss_grid_clean.csv")
print("flagged trips:", sorted(flagged))
print("\nMISS % (rows = r metres, cols = k), CLEAN trips:")
print(clean.miss_pct.unstack().to_string())
print("\nMISS % ALL trips:"); print(allt.miss_pct.unstack().to_string())
print("\nCandidate rows per GPS point, CLEAN:"); print(clean.rows_per_pt.unstack().to_string())