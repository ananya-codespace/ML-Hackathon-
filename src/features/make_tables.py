import sys, time
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import build_trip, FEATURES, R, K
from loader import DATA, ROOT, list_trips

OUT = DATA / "processed"; OUT.mkdir(exist_ok=True)
(ROOT / "artifacts").mkdir(exist_ok=True)

# ---- fixed trip-level folds: largest trips first, each to the currently lightest fold
inv = pd.read_csv(ROOT / "results" / "dataset_inventory.csv", dtype={"trip_id": str})
tr = inv[inv.partition == "train"].sort_values(["n_points", "trip_id"], ascending=[False, True])
load, fold = np.zeros(5), {}
for tid, n in zip(tr.trip_id, tr.n_points):
    f = int(np.argmin(load)); fold[tid] = f; load[f] += n
pd.DataFrame({"trip_id": list(fold), "fold": list(fold.values())}).sort_values("trip_id") \
  .to_csv(ROOT / "results" / "cv_folds.csv", index=False)
print("points per fold:", load.astype(int), "| trips per fold:", np.bincount(list(fold.values())))

def build_partition(part):
    X, P = [], []
    for i, tid in enumerate(list_trips(part)):
        t0 = time.time(); df, pt = build_trip(tid, with_label=True); X.append(df); P.append(pt)
        print(f"  {part} {tid}: {len(df):>7} rows ({time.time()-t0:.1f}s)")
    X, P = pd.concat(X, ignore_index=True), pd.concat(P, ignore_index=True)
    flag = (P.label_dist_m > 50).groupby(P.trip_id).mean() > 0.5       # misaligned-trip flag
    X["trip_flagged"], P["trip_flagged"] = X.trip_id.map(flag), P.trip_id.map(flag)
    return X, P

summary = []
for part in ["train", "student_test"]:
    X, P = build_partition(part)
    if part == "train":
        X["fold"] = X.trip_id.map(fold); P["fold"] = P.trip_id.map(fold)
        sc = StandardScaler().fit(X[FEATURES])                           # TRAIN ONLY
        joblib.dump({"scaler": sc, "features": FEATURES, "radius": R, "k": K}, ROOT / "artifacts" / "scaler.joblib")
    X.to_parquet(OUT / f"{part}_features.parquet"); P.to_parquet(OUT / f"{part}_points.parquet")
    for name, m in [("all", slice(None)), ("clean", ~P.trip_flagged)]:
        Pm = P[m]; Xm = X[~X.trip_flagged] if name == "clean" else X
        summary.append(dict(partition=part, subset=name, trips=Pm.trip_id.nunique(), points=len(Pm),
                            rows=len(Xm), pos_rate_pct=round(Xm.y.mean() * 100, 2),
                            miss_pct=round((~Pm.label_in_cands).mean() * 100, 2),
                            rows_per_point=round(len(Xm) / len(Pm), 2)))
s = pd.DataFrame(summary); s.to_csv(ROOT / "results" / "feature_table_summary.csv", index=False)
print(s.to_string()); print("flagged trips:", sorted(P[P.trip_flagged].trip_id.unique()) if part else "")