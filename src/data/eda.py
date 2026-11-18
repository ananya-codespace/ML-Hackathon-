import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from loader import list_trips, load_trip, ROOT

RES = ROOT / "results"; FIG = RES / "figures"; FIG.mkdir(parents=True, exist_ok=True)

def m_per_deg(lat):  # rough metres per degree (fine for EDA only)
    return 111320 * np.cos(np.radians(lat)), 110540

rows, all_dt = [], []
for tid in list_trips("train"):
    d = load_trip("train", tid)
    tr, nd, ar, rt = d["track"], d["nodes"], d["arcs"], d["route"].to_numpy()
    # --- track
    dt = tr["t"].diff().dropna().to_numpy(); all_dt.append(dt)
    kx, ky = m_per_deg(tr["lat"].mean())
    step = np.hypot(tr["lon"].diff().dropna() * kx, tr["lat"].diff().dropna() * ky).to_numpy()
    speed = step / np.where(dt > 0, dt, np.nan)
    big = int(np.argmax(dt))
    # --- network
    n, f, t = len(nd), ar["from_node"].to_numpy(), ar["to_node"].to_numpy()
    g = coo_matrix((np.ones(len(ar)), (f, t)), shape=(n, n))
    ncomp, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab)
    isolated = int((np.bincount(np.concatenate([f, t]), minlength=n) == 0).sum())
    key = f.astype(np.int64) * n + t
    rev_share = float(np.isin(t.astype(np.int64) * n + f, key).mean())
    # --- route connectivity: end of arc i == start of arc i+1 ?
    conn = (t[rt[:-1]] == f[rt[1:]])
    same = (rt[:-1] == rt[1:])
    rows.append(dict(
        trip_id=tid, n_points=len(tr), duration_s=tr["t"].iloc[-1] - tr["t"].iloc[0],
        pct_dt_1s=round((dt == 1).mean() * 100, 1), n_gaps_gt10s=int((dt > 10).sum()),
        max_gap_s=dt.max(), max_gap_at_idx=big, jump_at_max_gap_m=round(step[big], 1),
        median_speed_ms=round(float(np.nanmedian(speed)), 1), p99_speed_ms=round(float(np.nanpercentile(speed, 99)), 1),
        extent_x_km=round((tr["lon"].max() - tr["lon"].min()) * kx / 1000, 2),
        extent_y_km=round((tr["lat"].max() - tr["lat"].min()) * ky / 1000, 2),
        n_nodes=n, n_arcs=len(ar), isolated_nodes=isolated, n_components=ncomp,
        largest_comp_pct=round(sizes.max() / n * 100, 1), pct_arcs_with_reverse=round(rev_share * 100, 1),
        self_loops=int((f == t).sum()), dup_arcs=int(len(key) - len(np.unique(key))),
        route_len=len(rt), route_pct_connected=round(conn.mean() * 100, 2),
        route_repeats=int(same.sum()), route_unique_arcs=len(np.unique(rt)),
        route_in_one_component=bool(len(np.unique(lab[f[rt]])) == 1),
    ))

eda = pd.DataFrame(rows); eda.to_csv(RES / "eda_train_trips.csv", index=False)
pd.set_option("display.width", 200)
print(eda.describe().T[["min", "50%", "max"]].to_string())
print("\nTrips with gap > 60 s:\n", eda[eda.max_gap_s > 60][["trip_id", "n_points", "n_gaps_gt10s", "max_gap_s", "max_gap_at_idx", "jump_at_max_gap_m"]].to_string())
print("\nRoute not fully connected:\n", eda[eda.route_pct_connected < 100][["trip_id", "route_len", "route_pct_connected"]].to_string())
print("\nRoute spans >1 component:", int((~eda.route_in_one_component).sum()), "trips")
dt_all = np.concatenate(all_dt)
print("\nOverall dt==1s: %.2f%% | dt>10s: %d steps" % ((dt_all == 1).mean() * 100, (dt_all > 10).sum()))
print(pd.Series(dt_all).value_counts().head(6))

# plots
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(eda.n_points, bins=30); ax[0].set_title("GPS points per train trip")
ax[1].hist(np.log10(dt_all[dt_all > 0]), bins=40); ax[1].set_title("log10(time step, s) - train"); 
ax[2].scatter(eda.n_points, eda.n_arcs, s=12); ax[2].set_xlabel("points"); ax[2].set_ylabel("arcs"); ax[2].set_title("Trip length vs network size")
plt.tight_layout(); plt.savefig(FIG / "eda_train_overview.png", dpi=150)