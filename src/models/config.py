"""Shared configuration for Person B (models + Bayesian optimisation).

Everything that has to be agreed with A / C / D lives here: file locations,
column names, seed, number of folds.  Paths can be overridden with
environment variables so the same code runs on every teammate's laptop.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PATHSNAP_ROOT", Path(__file__).resolve().parents[2]))

# ---- inputs produced by Person A (feature tables) -------------------------
FEATURE_DIR = Path(os.environ.get("PATHSNAP_FEATURE_DIR", ROOT / "data" / "processed"))
TRAIN_TABLE_STEM = "train_features"            # .parquet or .csv
TEST_TABLE_STEM = "student_test_features"      # .parquet or .csv

# ---- outputs ---------------------------------------------------------------
RESULTS_DIR = Path(os.environ.get("PATHSNAP_RESULTS_DIR", ROOT / "results"))
ARTIFACTS_DIR = Path(os.environ.get("PATHSNAP_ARTIFACTS_DIR", ROOT / "artifacts"))
FIG_DIR = RESULTS_DIR / "figures"
FOLDS_FILE = ARTIFACTS_DIR / "folds.json"      # shared trip-level split (interface item 4)

# ---- table schema ----------------------------------------------------------
# One row per (GPS point, candidate arc).  Every other column is a feature.
KEY_COLS = ["trip_id", "obs_idx", "arc_id"]
LABEL_COL = "label"                            # 1 if arc_id is the reference arc for that point
META_COLS = KEY_COLS + [LABEL_COL]
# Columns that exist in the table but must NOT be fed to a model (add A's extras here).
# Person A's table uses `y` for the label and carries two non-feature columns: `trip_flagged` and `fold`.
LABEL_ALIASES = {"y": "label"}                 # renamed to `label` on load
EXCLUDE_FEATURE_COLS: list = ["trip_flagged", "fold"]
# A flags trips whose reference route lies >50 m from most GPS points (misaligned labels).
# Dropping them from TRAINING/CV only (never from student-test) keeps label noise out of fitting.
DROP_FLAGGED_TRAIN_TRIPS = True

SEED = 42
N_FOLDS = 5
EPS = 1e-15
