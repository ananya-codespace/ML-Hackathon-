
import logging

from fastapi import APIRouter, HTTPException, Request

from src.data.loader import list_trips
from src.api.config import TRIPS_PARTITION
from src.api.pipeline import run_pipeline


logger = logging.getLogger(__name__)


router = APIRouter()


@router.get("/trips", tags=["Trips"])
def get_trips():

    try:
        trips = list_trips(TRIPS_PARTITION)

        return {
            "partition": TRIPS_PARTITION,
            "total_trips": len(trips),
            "trips": trips
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=f"Trip directory data/{TRIPS_PARTITION} not found."
        )


@router.post("/predict/{trip_id}", tags=["Prediction"])
def predict_route(trip_id: str, request: Request):

    try:
        available_trips = list_trips(TRIPS_PARTITION)

        if trip_id not in available_trips:
            raise HTTPException(
                status_code=404,
                detail=f"Trip {trip_id} not found."
            )

        client = request.client.host if request.client else "unknown"
        logger.info("Predicting trip %s for %s", trip_id, client)

        result = run_pipeline(trip_id)

        m = result["metrics"]
        logger.info(
            "Trip %s done in %.2fs: point accuracy %.3f (classifier only %.3f) | "
            "continuity %.3f | route F1 %.3f",
            trip_id, result["processing_seconds"], m["point_accuracy"],
            m["classifier_only_point_accuracy"], m["route_continuity"], m["route_f1"]
        )

        return result

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
