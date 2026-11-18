
"""
PathSnap prediction pipeline.
Connects candidate generation, trained classifier,
HMM/Viterbi decoding and Folium visualisation.
"""

import json
import time

import numpy as np
import pandas as pd

from src.api.config import HMM_CONFIG_PATH, TRIPS_DIR, TRIPS_PARTITION
from src.evaluate_trips import score_trip
from src.features.build_features import build_trip
from src.models.inference import FinalClassifier

from src.hmm_viterbi.network import load_trip, RoadNetwork
from src.hmm_viterbi.viterbi import decode, HMMParams

from src.viz_folium.maps import trip_map


_classifier = None
_hmm_params = None


def get_classifier():
    """Load the trained classifier once and reuse it."""
    global _classifier

    if _classifier is None:
        _classifier = FinalClassifier.load()

    return _classifier


def get_hmm_params():
    """Load the tuned HMM settings (artifacts/hmm_config.json) once and reuse them."""
    global _hmm_params

    if _hmm_params is None:
        config = json.loads(HMM_CONFIG_PATH.read_text())
        _hmm_params = HMMParams(**config["hmm"])

    return _hmm_params


def run_pipeline(trip_id):
    """
    Runs the actual PathSnap map-matching pipeline.

    Ground-truth routes are not loaded during prediction; the reference
    route is read only afterwards, on this server, to compute the metrics.
    """

    start = time.perf_counter()

    trip_dir = TRIPS_DIR / trip_id

    if not trip_dir.is_dir():
        raise FileNotFoundError(
            f"Trip not found in {TRIPS_PARTITION}: {trip_id}"
        )

    # 1. Generate candidates and features.
    candidates, _ = build_trip(
        trip_id,
        with_label=False
    )

    if candidates.empty:
        raise ValueError(
            f"No candidate road segments found for {trip_id}"
        )

    # 2. Load the trip and road network without reference labels.
    trip = load_trip(
        trip_dir,
        with_route=False
    )

    network = RoadNetwork.from_trip(trip)

    # 3. Get probabilities from the trained model.
    classifier = get_classifier()

    probabilities = classifier.predict_proba(candidates)

    # 4. Classifier-only predictions.
    candidate_results = candidates.copy()
    candidate_results["probability"] = probabilities

    best_indices = (
        candidate_results.groupby("obs_idx")["probability"]
        .idxmax()
    )

    classifier_route = np.full(
        len(trip.track_xy), -1, dtype=int
    )

    best_rows = candidate_results.loc[best_indices]

    classifier_route[
        best_rows["obs_idx"].to_numpy(dtype=int)
    ] = best_rows["arc_id"].to_numpy(dtype=int)

    # 5. HMM/Viterbi route reconstruction.
    hmm_route, hmm_info = decode(
        candidates,
        probabilities,
        trip,
        network,
        get_hmm_params()
    )

    processing_seconds = time.perf_counter() - start

    # 6. Score against the server-side reference route (after prediction).
    scores = score_trip(
        trip_id,
        TRIPS_PARTITION,
        {
            "net": network,
            "classifier_only": classifier_route,
            "hmm": hmm_route,
            "info": hmm_info,
            "seconds": processing_seconds,
        },
    )
    classifier_scores, hmm_scores = scores[0], scores[1]

    metrics = {
        "point_accuracy": round(hmm_scores["point_accuracy"], 4),
        "route_continuity": round(hmm_scores["route_continuity"], 4),
        "route_precision": round(hmm_scores["route_precision"], 4),
        "route_recall": round(hmm_scores["route_recall"], 4),
        "route_f1": round(hmm_scores["route_f1"], 4),
        "classifier_only_point_accuracy": round(classifier_scores["point_accuracy"], 4),
        "classifier_only_route_continuity": round(classifier_scores["route_continuity"], 4),
    }

    # 7. Generate Folium map (reference route not shown).
    map_object = trip_map(
        trip,
        network,
        pred_hmm=hmm_route,
        pred_classifier=classifier_route,
        labels=None,
        show_reference=False
    )

    map_html = map_object.get_root().render()

    return {
        "trip_id": trip_id,
        "predicted_route": hmm_route.tolist(),
        "classifier_route": classifier_route.tolist(),
        "hmm_info": hmm_info,
        "candidate_count": len(candidates),
        "metrics": metrics,
        "n_gps_points": int(len(trip.track_xy)),
        "processing_seconds": round(processing_seconds, 2),
        "is_mock": False,
        "map_html": map_html
    }
