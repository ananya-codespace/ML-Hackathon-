
"""
PathSnap prediction pipeline.
Connects candidate generation, trained classifier,
HMM/Viterbi decoding and Folium visualisation.
"""

import numpy as np
import pandas as pd

from src.data.loader import DATA
from src.features.build_features import build_trip
from src.models.inference import FinalClassifier

from src.hmm_viterbi.network import load_trip, RoadNetwork
from src.hmm_viterbi.viterbi import decode, HMMParams

from src.viz_folium.maps import trip_map


_classifier = None


def get_classifier():
    """Load the trained classifier once and reuse it."""
    global _classifier

    if _classifier is None:
        _classifier = FinalClassifier.load()

    return _classifier


def run_pipeline(trip_id):
    """
    Runs the actual PathSnap map-matching pipeline.

    Ground-truth routes are not loaded during prediction.
    """

    trip_dir = DATA / "student_test" / trip_id

    if not trip_dir.is_dir():
        raise FileNotFoundError(
            f"Student-test trip not found: {trip_id}"
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
        HMMParams()
    )

    # 6. Generate Folium map.
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
        "is_mock": False,
        "map_html": map_html
    }
