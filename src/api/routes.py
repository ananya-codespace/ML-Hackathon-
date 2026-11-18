
import logging

from fastapi import APIRouter, HTTPException, Request

from src.api.config import TRIPS_PARTITION
from src.api.pipeline import available_trips, run_pipeline


logger = logging.getLogger(__name__)

# Create an API router
router = APIRouter()


# Endpoint 1: Get available trips (the trip folders present on this server)
@router.get("/trips", tags=["Trips"])
def get_trips():

    trips = available_trips()

    return {
        "partition": TRIPS_PARTITION,
        "total_trips": len(trips),
        "trips": trips
    }


# Endpoint 2: Predict route for a selected trip
@router.post("/predict/{trip_id}", tags=["Prediction"])
def predict_route(trip_id: str, request: Request):

    # Check whether the trip ID is valid
    if trip_id not in available_trips():
        logger.warning("Rejected trip %s from %s: not found", trip_id, request.client.host)
        raise HTTPException(
            status_code=404,
            detail=f"Trip {trip_id} not found"
        )

    logger.info("Predicting trip %s for %s", trip_id, request.client.host)

    # Run the full prediction pipeline (features -> classifier -> HMM), then score server-side
    result = run_pipeline(trip_id)

    m = result["metrics"]
    logger.info(
        "Trip %s done in %.2fs: %d GPS points -> %d route arcs | point accuracy %.3f "
        "(classifier only %.3f) | continuity %.3f",
        trip_id, result["processing_seconds"], result["n_gps_points"], len(result["predicted_route"]),
        m["point_accuracy"], m["classifier_only_point_accuracy"], m["route_continuity"]
    )

    return result
