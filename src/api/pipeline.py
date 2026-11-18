
"""
Temporary PathSnap prediction pipeline.

These are mock implementations for backend development.
They will be replaced with the actual implementations
from Persons A, B and C.
"""


def build_candidates(trip_id):
    """
    Mock candidate generation (Person A).

    Generates sample candidate road segments for each GPS point.
    """

    candidates = [
        {"trip_id": trip_id, "obs_idx": 0, "arc_id": 101},
        {"trip_id": trip_id, "obs_idx": 0, "arc_id": 102},

        {"trip_id": trip_id, "obs_idx": 1, "arc_id": 103},
        {"trip_id": trip_id, "obs_idx": 1, "arc_id": 104},

        {"trip_id": trip_id, "obs_idx": 2, "arc_id": 105},
        {"trip_id": trip_id, "obs_idx": 2, "arc_id": 106},
    ]

    return candidates


def predict_proba(table):
    """
    Mock classifier prediction (Person B).

    Returns sample probabilities for each candidate arc.
    """

    probabilities = [0.85, 0.15, 0.75, 0.25, 0.90, 0.10]

    return probabilities


def decode(candidates, probs, network):
    """
    Mock route decoding (Person C).

    Selects the highest-probability candidate
    for each GPS observation.

    This is NOT the actual HMM/Viterbi algorithm.
    """

    predictions = {}

    for candidate, probability in zip(candidates, probs):

        obs_idx = candidate["obs_idx"]
        arc_id = candidate["arc_id"]

        if (obs_idx not in predictions or
                probability > predictions[obs_idx]["probability"]):

            predictions[obs_idx] = {
                "arc_id": arc_id,
                "probability": probability
            }

    predicted_route = [
        predictions[index]["arc_id"]
        for index in sorted(predictions)
    ]

    return predicted_route


def run_pipeline(trip_id):
    """
    Executes the temporary PathSnap pipeline.
    """

    # Step 1: Generate candidates
    candidates = build_candidates(trip_id)

    # Step 2: Obtain classifier probabilities
    probabilities = predict_proba(candidates)

    # Step 3: Decode the predicted route
    network = None
    predicted_route = decode(
        candidates,
        probabilities,
        network
    )

    return {
        "trip_id": trip_id,
        "predicted_route": predicted_route,
        "is_mock": True,
        "message": "Temporary mock prediction. Not actual ML output."
    }
