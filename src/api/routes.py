
from fastapi import APIRouter, HTTPException

from src.api.pipeline import run_pipeline


# Create an API router
router = APIRouter()

# Temporary student-test trip IDs
# Replace these with actual IDs after checking the dataset.
AVAILABLE_TRIPS = [
    f"trip_{i:03d}" for i in range(1, 11)
]


# Endpoint 1: Get available trips
@router.get("/trips", tags=["Trips"])
def get_trips():

    return {
        "total_trips": len(AVAILABLE_TRIPS),
        "trips": AVAILABLE_TRIPS
    }


# Endpoint 2: Predict route for a selected trip
@router.post("/predict/{trip_id}", tags=["Prediction"])
def predict_route(trip_id: str):

    # Check whether the trip ID is valid
    if trip_id not in AVAILABLE_TRIPS:
        raise HTTPException(
            status_code=404,
            detail=f"Trip {trip_id} not found"
        )

    # Run the mock prediction pipeline
    result = run_pipeline(trip_id)

    return result
