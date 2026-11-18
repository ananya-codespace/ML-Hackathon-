import pandas as pd
from pathlib import Path
P = Path(__file__).resolve().parents[1] / "data" / "processed"

def test_no_trip_in_two_folds():
    t = pd.read_parquet(P / "train_features.parquet", columns=["trip_id", "fold"])
    assert t.groupby("trip_id")["fold"].nunique().max() == 1

def test_no_label_columns_in_features():
    from sys import path; path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "features"))
    from build_features import FEATURES
    assert not {"y", "label_arc_id", "fold"} & set(FEATURES)

def test_partitions_disjoint():
    a = set(pd.read_parquet(P / "train_features.parquet", columns=["trip_id"]).trip_id)
    b = set(pd.read_parquet(P / "student_test_features.parquet", columns=["trip_id"]).trip_id)
    assert not a & b