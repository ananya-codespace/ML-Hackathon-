
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
STUDENT_TEST_DIR = DATA_DIR / "student_test"

# Logging configuration
LOG_LEVEL = "INFO"
