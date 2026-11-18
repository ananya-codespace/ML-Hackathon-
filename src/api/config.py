
import os
from pathlib import Path

# Find the root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Project directories
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"

# API configuration
API_TITLE = "PathSnap API"
API_DESCRIPTION = "GPS Map-Matching System"
API_VERSION = "1.0.0"

HOST = "0.0.0.0"
PORT = 8000

# Dataset configuration
# The trips offered to the evaluator: one folder per trip under data/<TRIPS_PARTITION>/.
# Default is the 10 evaluator trips; set PATHSNAP_TRIPS_PARTITION=student_test to rehearse.
TRIPS_PARTITION = os.environ.get("PATHSNAP_TRIPS_PARTITION", "evaluator_test")
TRIPS_DIR = DATA_DIR / TRIPS_PARTITION
STUDENT_TEST_DIR = DATA_DIR / "student_test"

# Response size limits (keeps the JSON small for big trips)
MAX_GPS_POINTS_RETURNED = 2000

# Logging configuration
LOG_LEVEL = "INFO"
