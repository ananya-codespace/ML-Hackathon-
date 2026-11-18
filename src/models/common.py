"""Loading feature tables, trip-level folds and leakage guards."""
import json
import re
import warnings

import numpy as np
import pandas as pd

from . import config as C


def _read_table(stem: str) -> pd.DataFrame:
    pq, csv = C.FEATURE_DIR / f"{stem}.parquet", C.FEATURE_DIR / f"{stem}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(
        f"Feature table '{stem}' not found in {C.FEATURE_DIR} (looked for .parquet and .csv). "
        "Person A must export it - see README_B.md, section 'Input contract'."
    )


def feature_columns(df: pd.DataFrame) -> list:
    cols = [c for c in df.columns if c not in C.META_COLS and c not in C.EXCLUDE_FEATURE_COLS]
    bad = [c for c in cols if re.search(r"(label|truth|ground|reference|^ref_|route_id)", c, re.I)]
    if bad:
        raise ValueError(f"Possible label leakage in feature columns: {bad}. "
                         "Rename them or add them to config.EXCLUDE_FEATURE_COLS.")
    non_num = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_num:
        raise ValueError(f"Non-numeric feature columns: {non_num}")
    return cols


def validate_table(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in C.META_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}")
    if not set(df[C.LABEL_COL].unique()) <= {0, 1}:
        raise ValueError(f"{name}: label must be 0/1")
    per_obs = df.groupby(["trip_id", "obs_idx"])[C.LABEL_COL].sum()
    n_multi = int((per_obs > 1).sum())
    if n_multi:
        warnings.warn(f"{name}: {n_multi} observations have >1 positive candidate row "
                      "(duplicate arc_id rows?). Top-1 logic assumes at most one.")
    if df.duplicated(C.KEY_COLS).any():
        warnings.warn(f"{name}: duplicate (trip_id, obs_idx, arc_id) rows found")


def load_train_test():
    train, test = _read_table(C.TRAIN_TABLE_STEM), _read_table(C.TEST_TABLE_STEM)
    validate_table(train, "train")
    validate_table(test, "student_test")
    overlap = set(train.trip_id.unique()) & set(test.trip_id.unique())
    if overlap:
        raise ValueError(f"Trips appear in both train and student-test: {sorted(overlap)[:5]}")
    fc = feature_columns(train)
    if fc != feature_columns(test):
        raise ValueError("Train and student-test feature columns differ")
    train = train.sort_values(["trip_id", "obs_idx", "arc_id"]).reset_index(drop=True)
    test = test.sort_values(["trip_id", "obs_idx", "arc_id"]).reset_index(drop=True)
    return train, test, fc


def get_folds(train: pd.DataFrame, n_splits: int = C.N_FOLDS, seed: int = C.SEED) -> np.ndarray:
    """Trip-level folds: no trip is ever split across folds.

    If A already saved artifacts/folds.json we use it unchanged (shared split);
    otherwise we create one: trips are shuffled with a fixed seed and assigned
    greedily (largest first) to the currently lightest fold so folds hold a
    similar number of candidate rows.
    """
    trips = train.trip_id.astype(str)
    if C.FOLDS_FILE.exists():
        spec = json.loads(C.FOLDS_FILE.read_text())
        m = spec["trip_to_fold"]
        unknown = set(trips.unique()) - set(m)
        if unknown:
            raise ValueError(f"folds.json does not cover trips: {sorted(unknown)[:5]}")
        return trips.map(m).to_numpy().astype(int)
    sizes = trips.value_counts()
    rng = np.random.RandomState(seed)
    order = sizes.index.to_numpy()[rng.permutation(len(sizes))]
    order = sorted(order, key=lambda t: -sizes[t])          # stable: ties keep shuffled order
    load, mapping = np.zeros(n_splits), {}
    for t in order:
        f = int(np.argmin(load))
        mapping[t] = f
        load[f] += sizes[t]
    C.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    C.FOLDS_FILE.write_text(json.dumps({"seed": seed, "n_splits": n_splits,
                                        "trip_to_fold": mapping}, indent=1))
    return trips.map(mapping).to_numpy().astype(int)


def to_xy(df: pd.DataFrame, cols: list):
    return df[cols].to_numpy(dtype=np.float32), df[C.LABEL_COL].to_numpy(dtype=np.int8)
