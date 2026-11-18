"""
PathSnap prediction pipeline used by the API.

    candidates + features (Person A) -> tuned classifier (Person B) -> HMM/Viterbi (Person C)

The model and decoder settings are loaded once at start-up from artifacts/.
Prediction reads only .track/.nodes/.arcs. The reference route (.route) is read
afterwards, only on the server, to compute accuracy; it is never returned to the UI.
"""

import json
import logging
import time

import numpy as np

from src.api.config import ARTIFACTS_DIR, MAX_GPS_POINTS_RETURNED, TRIPS_DIR, TRIPS_PARTITION
from src.evaluate_trips import predict_trip, score_trip
from src.hmm_viterbi.metrics import collapse
from src.hmm_viterbi.viterbi import HMMParams
from src.models.inference import FinalClassifier

logger = logging.getLogger(__name__)

_STATE = {}


def load_artifacts():
    """Load the classifier and HMM settings once (called at API start-up)."""
    if "clf" in _STATE:
        return
    t0 = time.perf_counter()
    _STATE["clf"] = FinalClassifier.load()
    cfg = json.loads((ARTIFACTS_DIR / "hmm_config.json").read_text())
    _STATE["hmm"] = HMMParams(**cfg["hmm"])
    logger.info("Loaded model %s (tuned=%s) and HMM settings %s in %.1fs; trips from %s",
                _STATE["clf"].meta.get("model"), _STATE["clf"].meta.get("tuned"),
                cfg["hmm"], time.perf_counter() - t0, TRIPS_DIR)


def available_trips():
    """Trip IDs = the trip folders present on this server."""
    if not TRIPS_DIR.is_dir():
        return []
    return sorted(p.name for p in TRIPS_DIR.iterdir() if p.is_dir())


def _latlon(lonlat):
    return [[round(float(lat), 6), round(float(lon), 6)] for lon, lat in lonlat]


def run_pipeline(trip_id):
    """Predict the route for one trip, then score it against the server-side reference."""
    load_artifacts()
    pred = predict_trip(trip_id, TRIPS_PARTITION, _STATE["clf"], _STATE["hmm"])
    rows = score_trip(trip_id, TRIPS_PARTITION, pred)  # reads .route only here
    clf_row, hmm_row = rows[0], rows[1]

    trip = pred["trip"]
    arcs = trip.arcs
    nodes = trip.nodes_lonlat
    route_arcs = collapse(pred["hmm"])  # driving order, consecutive repeats removed
    segments = [_latlon([nodes[arcs[a, 0]], nodes[arcs[a, 1]]]) for a in route_arcs]
    step = max(1, len(trip.track_lonlat) // MAX_GPS_POINTS_RETURNED)

    return {
        "trip_id": trip_id,
        "n_gps_points": int(len(trip.track_lonlat)),
        "predicted_route": [int(a) for a in route_arcs],
        "predicted_segments": segments,
        "gps_points": _latlon(trip.track_lonlat[::step]),
        "gps_points_step": int(step),
        "metrics": {
            "point_accuracy": round(hmm_row["point_accuracy"], 4),
            "route_continuity": round(hmm_row["route_continuity"], 4),
            "route_precision": round(hmm_row["route_precision"], 4),
            "route_recall": round(hmm_row["route_recall"], 4),
            "route_f1": round(hmm_row["route_f1"], 4),
            "classifier_only_point_accuracy": round(clf_row["point_accuracy"], 4),
            "classifier_only_route_continuity": round(clf_row["route_continuity"], 4),
        },
        "decoder": {
            "time_gap_breaks": int(pred["info"]["breaks_time_gap"]),
            "unreachable_breaks": int(pred["info"]["breaks_unreachable"]),
            "points_without_candidates": int(pred["info"]["n_no_candidate"]),
        },
        "processing_seconds": round(pred["seconds"], 2),
        "is_mock": False,
    }
