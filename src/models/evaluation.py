"""Evaluation harness: ONE function that scores any model the same way.

run_cv()        grouped (by trip) cross-validation on the 80 training trips
evaluate_model() CV + full fit + train / student-test scoring + timings
"""
import time

import numpy as np

from . import config as C
from . import metrics as M
from .common import to_xy


def load_continuity_fn():
    """Return C's route-continuity function if it exists, else None (reported as NaN).

    Expected signature: f(pred_df, split) -> float   where pred_df has columns
    trip_id, obs_idx, arc_id (one predicted arc per GPS point) and split is
    'train' or 'student_test'.  Adjust the import below to the name C uses.
    """
    try:
        from src.hmm_viterbi import route_continuity           # noqa: F401  (C / D own this)
        return route_continuity
    except Exception:
        return None


def run_cv(build_fn, X, y, meta, fold_ids, n_folds=C.N_FOLDS):
    oof = np.full(len(y), np.nan, dtype=np.float64)
    fit_times = []
    for f in range(n_folds):
        tr, va = fold_ids != f, fold_ids == f
        model = build_fn()
        t0 = time.perf_counter()
        model.fit(X[tr], y[tr])
        fit_times.append(time.perf_counter() - t0)
        oof[va] = model.predict_proba(X[va])[:, 1]
    thr, pooled_f1 = M.best_threshold(y, oof)
    folds = []
    for f in range(n_folds):
        va = fold_ids == f
        folds.append(M.full_metrics(y[va], oof[va], meta.loc[va], thr))
    mf1 = np.array([r["macro_f1"] for r in folds])
    return dict(oof=oof, threshold=thr, pooled_macro_f1=pooled_f1, folds=folds,
                cv_macro_f1_mean=float(mf1.mean()), cv_macro_f1_std=float(mf1.std(ddof=0)),
                cv_top1_mean=float(np.mean([r["top1_accuracy"] for r in folds])),
                cv_top1_std=float(np.std([r["top1_accuracy"] for r in folds])),
                cv_log_loss_mean=float(np.mean([r["log_loss"] for r in folds])),
                cv_fit_time_mean_s=float(np.mean(fit_times)))


def evaluate_model(name, build_fn, train, test, feat_cols, fold_ids, config_dict, continuity_fn=None):
    """Full record for one model.  Everything in the returned dict is later written to results/."""
    Xtr, ytr = to_xy(train, feat_cols)
    Xte, yte = to_xy(test, feat_cols)

    cv = run_cv(build_fn, Xtr, ytr, train, fold_ids)

    model = build_fn()
    t0 = time.perf_counter(); model.fit(Xtr, ytr); train_time = time.perf_counter() - t0
    p_tr = model.predict_proba(Xtr)[:, 1]
    t0 = time.perf_counter(); p_te = model.predict_proba(Xte)[:, 1]; infer_time = time.perf_counter() - t0

    thr = cv["threshold"]                      # chosen on TRAIN out-of-fold predictions only
    m_tr = M.full_metrics(ytr, p_tr, train, thr)
    m_te = M.full_metrics(yte, p_te, test, thr)
    m_te_05 = M.prf_at(yte, p_te, 0.5)

    cont = {"train": float("nan"), "student_test": float("nan")}
    pred_te = M.top1_predictions(test, p_te)
    if continuity_fn is not None:
        try:
            cont["student_test"] = float(continuity_fn(pred_te[C.KEY_COLS], "student_test"))
        except Exception as e:                  # never let a hook crash the experiment
            print(f"  [warn] continuity_fn failed: {e}")

    rec = dict(
        model=name, seed=C.SEED, config=config_dict, n_features=len(feat_cols), features=feat_cols,
        n_train_rows=int(len(ytr)), n_test_rows=int(len(yte)), positive_rate_train=float(ytr.mean()),
        cv={k: v for k, v in cv.items() if k != "oof"},
        train=m_tr, student_test=m_te, student_test_thr05=m_te_05,
        route_continuity_student_test_top1=cont["student_test"],
        train_time_s=train_time, inference_time_s=infer_time,
        inference_us_per_row=1e6 * infer_time / len(yte),
    )
    return rec, cv["oof"], p_te, model, pred_te
