import numpy as np, folium
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loader import load_trip, ROOT
from geo import RoadNetwork, to_xy, crs_for_track

TRIP, PART = "00000001", "train"          # clean trip; change to look at others
FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)

d = load_trip(PART, TRIP)
crs = crs_for_track(d["track"])
net = RoadNetwork(d["nodes"], d["arcs"], crs)
p = to_xy(d["track"].lon, d["track"].lat, crs)
near = np.unique(net.candidates(p, radius=300)["arc_id"].values)   # arcs near the track only
route = d["route"].to_numpy()

# ---- static figure (metres)
fig, ax = plt.subplots(figsize=(8, 8))
for a in near:
    ax.plot([net.start[a, 0], net.end[a, 0]], [net.start[a, 1], net.end[a, 1]], c="lightgray", lw=0.6)
for a in route:
    ax.plot([net.start[a, 0], net.end[a, 0]], [net.start[a, 1], net.end[a, 1]], c="green", lw=2)
ax.scatter(p[:, 0], p[:, 1], s=3, c="blue")
ax.set_aspect("equal"); ax.set_xlabel("x (m, UTM)"); ax.set_ylabel("y (m, UTM)")
ax.set_title(f"{PART} trip {TRIP}: GPS (blue), road network (grey), reference route (green)")
plt.tight_layout(); plt.savefig(FIG / f"trip_{TRIP}_{PART}.png", dpi=150)

# ---- Folium map
nodes = d["nodes"][["lat", "lon"]].to_numpy(); arcs = d["arcs"].to_numpy()
tr = d["track"]
m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=14, tiles="OpenStreetMap")
g_net = folium.FeatureGroup(name="Road network (within 300 m)"); g_rt = folium.FeatureGroup(name="Reference route")
g_gps = folium.FeatureGroup(name="GPS points")
for a in near:
    folium.PolyLine(nodes[arcs[a]], color="gray", weight=2, opacity=0.6).add_to(g_net)
for a in route:
    folium.PolyLine(nodes[arcs[a]], color="green", weight=5).add_to(g_rt)
for lat, lon in tr[["lat", "lon"]].to_numpy()[::3]:
    folium.CircleMarker([lat, lon], radius=2, color="blue", fill=True).add_to(g_gps)
for g in (g_net, g_rt, g_gps): g.add_to(m)
folium.LayerControl().add_to(m)
m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])
legend = f"""<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px;border:1px solid gray;font-size:13px">
<b>{PART} trip {TRIP}</b><br><span style="color:blue">&#9679;</span> GPS points<br>
<span style="color:green">&#9632;</span> Reference route<br><span style="color:gray">&#9632;</span> Road network</div>"""
m.get_root().html.add_child(folium.Element(legend))
m.save(str(ROOT / "results" / f"map_{TRIP}_{PART}.html"))
print("saved figure and map for", TRIP, "| arcs drawn:", len(near), "| route arcs:", len(route))