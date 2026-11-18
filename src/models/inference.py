"""Interface item 2:  predict_proba(table) -> classifier probability per candidate row.

Person D loads this once at API start-up:
    from src.models.inference import FinalClassifier
    clf = FinalClassifier.load()           # artifacts/final_model.joblib
    probs = clf.predict_proba(table)       # np.ndarray, len(table)
"""
import joblib
import numpy as np
import pandas as pd

from . import config as C

MODEL_PATH = C.ARTIFACTS_DIR / "final_model.joblib"
_CACHE = {}


class FinalClassifier:
    def __init__(self, pipeline, feature_cols, threshold, meta):
        self.pipeline, self.feature_cols, self.threshold, self.meta = pipeline, feature_cols, threshold, meta

    @classmethod
    def load(cls, path=MODEL_PATH):
        b = joblib.load(path)
        return cls(b["pipeline"], b["feature_cols"], b["threshold"], b["meta"])

    def predict_proba(self, table: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_cols if c not in table.columns]
        if missing:
            raise ValueError(f"table is missing feature columns {missing}")
        if C.LABEL_COL in self.feature_cols:
            raise RuntimeError("label must never be a model input")
        X = table[self.feature_cols].to_numpy(dtype=np.float32)   # same order as training; label never used
        return self.pipeline.predict_proba(X)[:, 1]


def predict_proba(table: pd.DataFrame) -> np.ndarray:
    if "clf" not in _CACHE:
        _CACHE["clf"] = FinalClassifier.load()
    return _CACHE["clf"].predict_proba(table)
