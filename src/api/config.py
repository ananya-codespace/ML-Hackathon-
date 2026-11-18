
import os
from pathlib import Path

# Project root directory
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
# Trips offered to the evaluator: one folder per trip under data/<TRIPS_PARTITION>/.
# Default is the 10 evaluator trips; PATHSNAP_TRIPS_PARTITION=student_test to rehearse.
TRIPS_PARTITION = os.environ.get("PATHSNAP_TRIPS_PARTITION", "evaluator_test")
TRIPS_DIR = DATA_DIR / TRIPS_PARTITION
STUDENT_TEST_DIR = DATA_DIR / "student_test"

# Tuned HMM/Viterbi settings (chosen on the training trips)
HMM_CONFIG_PATH = ARTIFACTS_DIR / "hmm_config.json"

# Logging configuration
LOG_LEVEL = "INFO"
