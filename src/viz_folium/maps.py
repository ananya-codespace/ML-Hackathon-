"""Folium maps: GPS points, road network, reference route, predicted routes.

    python -m src.viz_folium.maps --trip data/student_test/00000056 --out results/maps

Layers (each toggleable via the layer control):
    Road network (nearby)   grey thin lines
    GPS observations        blue dots (subsampled to <= max_points)
    Reference route         green, thick       (omit for the UI: server-side only)
    Classifier-only         orange, dashed     (hidden by default)
    Predicted route (HMM)   red
    HMM errors              black dots where the HMM arc != reference label
"""
from __future__ import annotations

import argparse
from pathlib import Path

import folium
import numpy as np
from branca.element import Element

COLORS = {"network": "#9e9e9e", "gps": "#1565c0", "reference": "#2e7d32",
          "classifier": "#ef6c00", "hmm": "#c62828", "error": "#000000"}


def _arcs_geojson(arc_ids, nodes_lonlat, arcs, name):
    arc_ids = np.unique(np.asarray(arc_ids)[np.asarray(arc_ids) >= 0])
    lines = [[nodes_lonlat[arcs[a, 0]].tolist(), nodes_lonlat[arcs[a, 1]].tolist()] for a in arc_ids]
    return {"type": "Feature", "properties": {"name": name},
            "geometry": {"type": "MultiLineString", "coordinates": lines}}


def _legend(items):
    rows = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;background:{c};'
        f'margin-right:6px;vertical-align:middle"></span>{label}</div>' for label, c in items)
    return Element(
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
        'padding:8px 10px;border:1px solid #888;border-radius:4px;font:12px sans-serif">'
        f'<b>Legend</b>{rows}</div>')


def trip_map(trip, net, pred_hmm=None, pred_classifier=None, labels=None,
             show_reference=True, max_points=3000, network_margin_m=300.0) -> folium.Map:
    lonlat = trip.track_lonlat
    m = folium.Map(location=[float(np.median(lonlat[:, 1])), float(np.median(lonlat[:, 0]))],
                   zoom_start=14, tiles="OpenStreetMap", control_scale=True)
    legend = []

    # nearby road network (projected bbox + margin), so big maps stay light
    lo = trip.track_xy.min(0) - network_margin_m
    hi = trip.track_xy.max(0) + network_margin_m
    mid = (net.nodes_xy[net.src] + net.nodes_xy[net.dst]) / 2
    near = np.flatnonzero(((mid >= lo) & (mid <= hi)).all(1))
    folium.GeoJson(_arcs_geojson(near, trip.nodes_lonlat, trip.arcs, "network"),
                   name="Road network (nearby)",
                   style_function=lambda _: {"color": COLORS["network"], "weight": 1, "opacity": 0.6}
                   ).add_to(m)
    legend.append(("Road network", COLORS["network"]))

    if show_reference and trip.route is not None:
        folium.GeoJson(_arcs_geojson(trip.route, trip.nodes_lonlat, trip.arcs, "reference"),
                       name="Reference route",
                       style_function=lambda _: {"color": COLORS["reference"], "weight": 7, "opacity": 0.6}
                       ).add_to(m)
        legend.append(("Reference route", COLORS["reference"]))

    if pred_classifier is not None:
        folium.GeoJson(_arcs_geojson(pred_classifier, trip.nodes_lonlat, trip.arcs, "classifier"),
                       name="Classifier-only", show=False,
                       style_function=lambda _: {"color": COLORS["classifier"], "weight": 3,
                                                 "dashArray": "6 4"}).add_to(m)
        legend.append(("Classifier-only", COLORS["classifier"]))

    if pred_hmm is not None:
        folium.GeoJson(_arcs_geojson(pred_hmm, trip.nodes_lonlat, trip.arcs, "hmm"),
                       name="Predicted route (HMM)",
                       style_function=lambda _: {"color": COLORS["hmm"], "weight": 3}).add_to(m)
        legend.append(("Predicted route (HMM)", COLORS["hmm"]))

    step = max(1, len(lonlat) // max_points)
    gps = folium.FeatureGroup(name=f"GPS observations (every {step})")
    for i in range(0, len(lonlat), step):
        folium.CircleMarker([lonlat[i, 1], lonlat[i, 0]], radius=2, color=COLORS["gps"],
                            fill=True, weight=0, fill_opacity=0.8,
                            tooltip=f"obs {i}, t={trip.times[i]:.0f}s").add_to(gps)
    gps.add_to(m)
    legend.append(("GPS observations", COLORS["gps"]))

    if pred_hmm is not None and labels is not None:
        wrong = np.flatnonzero(np.asarray(pred_hmm) != np.asarray(labels))
        err = folium.FeatureGroup(name=f"HMM errors ({len(wrong)} pts)", show=False)
        for i in wrong[:: max(1, len(wrong) // max_points)]:
            folium.CircleMarker([lonlat[i, 1], lonlat[i, 0]], radius=3, color=COLORS["error"],
                                fill=True, weight=0, fill_opacity=0.9).add_to(err)
        err.add_to(m)
        legend.append(("HMM errors", COLORS["error"]))

    folium.Marker([lonlat[0, 1], lonlat[0, 0]], tooltip="start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lonlat[-1, 1], lonlat[-1, 0]], tooltip="end",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[float(lonlat[:, 1].min()), float(lonlat[:, 0].min())],
                  [float(lonlat[:, 1].max()), float(lonlat[:, 0].max())]])
    m.get_root().html.add_child(_legend(legend))
    return m


def main():
    from src.hmm_viterbi import RoadNetwork, build_candidates, decode, distance_probs, load_trip
    from src.hmm_viterbi.labels import reference_labels
    from src.hmm_viterbi.metrics import top1_from_probs
    from src.hmm_viterbi.viterbi import HMMParams
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", nargs="+", required=True)
    ap.add_argument("--out", default="results/maps")
    ap.add_argument("--config", default="artifacts/hmm_config.json")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text()) if Path(args.config).exists() else {}
    params = HMMParams(**cfg.get("hmm", {}))
    sigma = cfg.get("stub_classifier_sigma", 10.0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for d in args.trip:
        trip = load_trip(d)
        net = RoadNetwork.from_trip(trip)
        cands = build_candidates(trip, net)
        probs = distance_probs(cands, sigma)
        hmm, _ = decode(cands, probs, trip, net, params)
        base = top1_from_probs(cands, probs, len(trip.track_xy))
        labels, _ = reference_labels(trip, net)
        path = out / f"trip_{trip.trip_id}.html"
        trip_map(trip, net, hmm, base, labels).save(path)
        print("wrote", path)


if __name__ == "__main__":
    main()
