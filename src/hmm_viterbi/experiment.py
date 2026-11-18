"""Classifier-only vs classifier + HMM/Viterbi, and decoder parameter tuning.

Usage (from the repo root):
    python -m src.hmm_viterbi.experiment --data data --out results

1. Grid-search decoder parameters (beta, sigma) on the 80 TRAIN trips only.
2. Freeze the best config into artifacts/hmm_config.json.
3. Report per-trip and overall metrics on train and student_test.

--probs-source stub (default): the "classifier" is the distance-only stand-in
    from candidates.distance_probs (argmax = nearest arc).
--probs-source b: read Person B's results/final_model_probs_train_oof.csv and
    final_model_probs_student_test.csv (trip_id, obs_idx, arc_id, label, prob);
    outputs go to results/hmm_bmodel/ and artifacts/hmm_config.json is replaced
    by the config tuned on B's out-of-fold probabilities.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import build_candidates, distance_probs
from .labels import reference_labels
from .metrics import (candidate_miss_rate, point_accuracy, route_continuity,
                      sequence_metrics, top1_from_probs, transition_continuity)
from .network import RoadNetwork, load_trip
from .viterbi import HMMParams, decode

K, RADIUS = 8, 50.0


def prepare(trip_dir: Path):
    trip = load_trip(trip_dir)
    net = RoadNetwork.from_trip(trip)
    labels, _ = reference_labels(trip, net)
    cands = build_candidates(trip, net, k=K, radius=RADIUS)
    return trip, net, labels, cands


B_FILES = {"train": "final_model_probs_train_oof.csv",
           "student_test": "final_model_probs_student_test.csv"}


def prepare_b(data: Path, part: str, probs_dir: str):
    """Use Person B's probabilities (train = out-of-fold, so tuning stays leak-free).
    Labels come from the same table (label column), so Top-1 matches B's numbers."""
    df = pd.read_csv(Path(probs_dir) / B_FILES[part], dtype={"trip_id": str})
    df["trip_id"] = df["trip_id"].str.zfill(8)
    prepared = []
    for tid, g in df.groupby("trip_id", sort=True):
        trip = load_trip(data / part / tid)
        net = RoadNetwork.from_trip(trip)
        labels = np.full(len(trip.track_xy), -1, dtype=np.int64)
        pos = g[g["label"] == 1]
        labels[pos["obs_idx"].to_numpy()] = pos["arc_id"].to_numpy()
        cands = g[["trip_id", "obs_idx", "arc_id"]].reset_index(drop=True)
        prepared.append((trip, net, labels, cands, g["prob"].to_numpy(dtype=float)))
    return prepared


def evaluate(trip, net, labels, cands, probs, params):
    n = len(trip.track_xy)
    rows = []
    t0 = time.perf_counter()
    base = top1_from_probs(cands, probs, n)
    t1 = time.perf_counter()
    hmm, info = decode(cands, probs, trip, net, params)
    t2 = time.perf_counter()
    for name, pred, secs in [("classifier_only", base, t1 - t0), ("classifier_hmm", hmm, t2 - t1)]:
        r = {
            "trip_id": trip.trip_id, "method": name, "n_points": n,
            "point_accuracy": point_accuracy(pred, labels),
            "route_continuity": route_continuity(pred, net),
            "transition_continuity": transition_continuity(pred, net),
            **sequence_metrics(pred, trip.route, net),
            "decode_seconds": secs,
        }
        if name == "classifier_hmm":
            r.update(info)
        rows.append(r)
    return rows, base, hmm


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["point_accuracy", "route_continuity", "transition_continuity",
               "route_precision", "route_recall", "route_f1"]
    g = df.groupby("method")
    out = g[metrics].mean().add_prefix("mean_")
    w = df.assign(**{m: df[m] * df["n_points"] for m in ["point_accuracy", "route_continuity"]})
    wg = w.groupby("method")
    out["pointweighted_accuracy"] = wg["point_accuracy"].sum() / wg["n_points"].sum()
    out["pointweighted_continuity"] = wg["route_continuity"].sum() / wg["n_points"].sum()
    out["total_decode_seconds"] = g["decode_seconds"].sum()
    out["n_trips"] = g.size()
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--betas", default="2,5,10,20,50")
    ap.add_argument("--sigmas", default="5,10")
    ap.add_argument("--tols", default="5,15,30,60")
    ap.add_argument("--probs-source", choices=["stub", "b"], default="stub",
                    help="stub: distance stand-in; b: Person B's probability CSVs")
    ap.add_argument("--b-probs-dir", default="results")
    args = ap.parse_args()
    data, out, art = Path(args.data), Path(args.out), Path(args.artifacts)
    out.mkdir(parents=True, exist_ok=True)
    art.mkdir(parents=True, exist_ok=True)

    b_mode = args.probs_source == "b"
    if b_mode:
        out = out / "hmm_bmodel"
        out.mkdir(parents=True, exist_ok=True)
    parts = {}
    for part in ["train", "student_test"]:
        prepared = prepare_b(data, part, args.b_probs_dir) if b_mode else [
            prepare(d) + (None,) for d in sorted(p for p in (data / part).iterdir() if p.is_dir())]
        miss = [{"trip_id": t.trip_id, "partition": part, "n_points": len(lab),
                 "candidate_miss_rate": candidate_miss_rate(c, lab)} for t, _, lab, c, _ in prepared]
        parts[part] = prepared
        pd.DataFrame(miss).to_csv(out / f"candidate_miss_{part}.csv", index=False)
        print(f"{part}: {len(prepared)} trips prepared, mean miss rate "
              f"{np.mean([m['candidate_miss_rate'] for m in miss]):.4f}")

    def probs_for(cands, fixed, sigma):
        return fixed if fixed is not None else distance_probs(cands, sigma)

    # --- 1. tune decoder on TRAIN only ---------------------------------
    sigmas = [float("nan")] if b_mode else list(map(float, args.sigmas.split(",")))
    grid = []
    for beta, sigma, tol in itertools.product(map(float, args.betas.split(",")), sigmas,
                                              map(float, args.tols.split(","))):
        params = HMMParams(beta=beta, same_arc_tol=tol)
        rows = []
        for trip, net, labels, cands, fixed in parts["train"]:
            r, _, _ = evaluate(trip, net, labels, cands, probs_for(cands, fixed, sigma), params)
            rows += [x for x in r if x["method"] == "classifier_hmm"]
        df = pd.DataFrame(rows)
        rec = {"beta": beta, "sigma": sigma, "same_arc_tol": tol,
               "mean_point_accuracy": df["point_accuracy"].mean(),
               "mean_route_continuity": df["route_continuity"].mean(),
               "mean_route_f1": df["route_f1"].mean(),
               "decode_seconds": df["decode_seconds"].sum()}
        grid.append(rec)
        print(rec)
    grid = pd.DataFrame(grid).sort_values("mean_point_accuracy", ascending=False)
    grid.to_csv(out / "decoder_tuning_train.csv", index=False)
    best = grid.iloc[0]
    params = HMMParams(beta=float(best["beta"]), same_arc_tol=float(best["same_arc_tol"]))
    sigma = float(best["sigma"])
    config = {"hmm": params.to_dict(),
              "emission_source": "Person B final model" if b_mode else "distance stand-in",
              "candidates": {"k": K, "radius_m": RADIUS},
              "selected_by": "mean point accuracy over 80 train trips"
                             + (" (B's out-of-fold probabilities)" if b_mode else "")}
    if not b_mode:
        config["stub_classifier_sigma"] = sigma
    (art / "hmm_config.json").write_text(json.dumps(config, indent=2))
    print("best", config)

    # --- 2. report with the frozen config -------------------------------
    summaries = []
    for part, prepared in parts.items():
        rows = []
        for trip, net, labels, cands, fixed in prepared:
            r, _, _ = evaluate(trip, net, labels, cands, probs_for(cands, fixed, sigma), params)
            rows += r
        df = pd.DataFrame(rows)
        df.to_csv(out / f"hmm_vs_classifier_{part}.csv", index=False)
        s = summarise(df)
        s.insert(0, "partition", part)
        summaries.append(s)
    summary = pd.concat(summaries)
    summary.to_csv(out / "hmm_vs_classifier_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
