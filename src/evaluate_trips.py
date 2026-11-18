"""End-to-end evaluation of the frozen pipeline on a folder of trips.

    python -m src.evaluate_trips --partition evaluator_test

Prediction and scoring are kept apart, the same way the API must keep them:
  predict_trip  reads only .track/.nodes/.arcs -> candidates (A), probabilities (B), HMM route (C)
  score_trip    reads .route afterwards, only to compute metrics
Nothing is fitted or tuned here: the model and decoder settings are loaded from artifacts/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src" / "features"), str(ROOT / "src" / "data")]

from build_features import build_trip  # noqa: E402  (Person A)
from labels import label_points  # noqa: E402  (Person A)
from geo import RoadNetwork as ANetwork, crs_for_track, to_xy  # noqa: E402
from loader import load_trip as a_load  # noqa: E402

from src.hmm_viterbi import RoadNetwork, decode, load_trip  # noqa: E402  (Person C)
from src.hmm_viterbi.labels import reference_labels  # noqa: E402
from src.hmm_viterbi.metrics import (point_accuracy, route_continuity, sequence_metrics,  # noqa: E402
                                     top1_from_probs)
from src.hmm_viterbi.viterbi import HMMParams  # noqa: E402
from src.models.inference import FinalClassifier  # noqa: E402  (Person B)


def predict_trip(trip_id: str, partition: str, clf: FinalClassifier, params: HMMParams) -> dict:
    """Ground truth is never read here (with_label=False / with_route=False)."""
    t0 = time.perf_counter()
    table, _ = build_trip(trip_id, with_label=False)
    probs = clf.predict_proba(table)
    trip = load_trip(ROOT / "data" / partition / trip_id, with_route=False)
    net = RoadNetwork.from_trip(trip)
    n = len(trip.track_xy)
    classifier_only = top1_from_probs(table, probs, n)
    hmm, info = decode(table[["obs_idx", "arc_id"]], probs, trip, net, params)
    return dict(trip=trip, net=net, classifier_only=classifier_only, hmm=hmm, info=info,
                seconds=time.perf_counter() - t0)


def score_trip(trip_id: str, partition: str, pred: dict) -> list[dict]:
    """Reads .route only now, after prediction is finished."""
    d = a_load(partition, trip_id, with_route=True)
    route = d["route"].to_numpy()
    crs = crs_for_track(d["track"])
    a_net = ANetwork(d["nodes"], d["arcs"], crs)
    xy = to_xy(d["track"].lon, d["track"].lat, crs)
    labels_a, _ = label_points(a_net, xy, route)  # official labels (nearest reference-route arc)
    trip = load_trip(ROOT / "data" / partition / trip_id, with_route=True)
    labels_order, _ = reference_labels(trip, pred["net"])  # sensitivity check only
    rows = []
    for method in ["classifier_only", "classifier_hmm"]:
        p = pred["classifier_only"] if method == "classifier_only" else pred["hmm"]
        rows.append(dict(
            trip_id=trip_id, method=method, n_points=len(p),
            point_accuracy=point_accuracy(p, labels_a),
            point_accuracy_order_labels=point_accuracy(p, labels_order),
            route_continuity=route_continuity(p, pred["net"]),
            **sequence_metrics(p, route, pred["net"]),
            pipeline_seconds=pred["seconds"],
            **({k: v for k, v in pred["info"].items()} if method == "classifier_hmm" else {}),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="evaluator_test")
    ap.add_argument("--out", default="results/evaluator_test")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    clf = FinalClassifier.load()
    cfg = json.loads((ROOT / "artifacts" / "hmm_config.json").read_text())
    params = HMMParams(**cfg["hmm"])
    print("model:", clf.meta["model"], "tuned" if clf.meta.get("tuned") else "untuned", "| HMM:", cfg["hmm"])

    trips = sorted(p.name for p in (ROOT / "data" / args.partition).iterdir() if p.is_dir())
    rows = []
    for tid in trips:
        pred = predict_trip(tid, args.partition, clf, params)
        r = score_trip(tid, args.partition, pred)
        rows += r
        print(f"{tid}: {r[0]['n_points']:>6} pts | classifier {r[0]['point_accuracy']:.3f} -> HMM "
              f"{r[1]['point_accuracy']:.3f} | continuity {r[1]['route_continuity']:.3f} | {pred['seconds']:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_trip.csv", index=False)
    metrics = ["point_accuracy", "point_accuracy_order_labels", "route_continuity",
               "route_precision", "route_recall", "route_f1"]
    g = df.groupby("method")
    summary = g[metrics].mean().add_prefix("mean_")
    summary["pooled_point_accuracy"] = (df.point_accuracy * df.n_points).groupby(df.method).sum() / g.n_points.sum()
    summary["n_trips"] = g.size()
    summary["n_points"] = g.n_points.sum()
    summary = summary.reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
