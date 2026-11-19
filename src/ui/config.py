import os

API_BASE_URL = os.getenv(
    "PATHSNAP_API_URL",
    "http://127.0.0.1:8000"
)

TRIPS_ENDPOINT = f"{API_BASE_URL}/trips"
PREDICTION_ENDPOINT = f"{API_BASE_URL}/predict"