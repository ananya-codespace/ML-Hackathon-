
from fastapi import APIRouter, HTTPException

from src.data.loader import list_trips
from src.api.pipeline import run_pipeline


router = APIRouter()


@router.get("/trips", tags=["Trips"])
def get_trips():

    try:
        trips = list_trips("student_test")

        return {
            "total_trips": len(trips),
            "trips": trips
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Student-test dataset directory not found."
        )


@router.post("/predict/{trip_id}", tags=["Prediction"])
def predict_route(trip_id: str):

    try:
        available_trips = list_trips("student_test")

        if trip_id not in available_trips:
            raise HTTPException(
                status_code=404,
                detail=f"Trip {trip_id} not found."
            )

        return run_pipeline(trip_id)

    except HTTPException:
        raise

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
