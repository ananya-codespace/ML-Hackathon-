"""Step 3 - Bayesian hyperparameter optimisation of the selected model, refit, compare, export.

    python -m src.models.run_bayes [--model hist_gb] [--n-trials 30] [--n-init 8]

Objective  : mean macro F1 over the trip-grouped CV folds of the 80 training trips (maximise).
Evaluation : the 10 student-test trips are used ONLY to report tuned-vs-untuned (never to choose parameters).
Outputs    : results/bayes_trials.csv|json, results/bayes_best_params.json, results/bayes_comparison.csv,
             artifacts/final_model.joblib, results/final_model_probs_*.csv (hand-off to C)
"""
import argparse
import json
import time

import joblib
import pandas as pd

from . import config as C
from . import zoo
from .bayes_opt import BayesOpt
from .common import get_folds, load_train_test, to_xy
from .evaluation import evaluate_model, load_continuity_fn, run_cv
from .reporting import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--n-init", type=int, default=8)
    ap.add_argument("--fresh", action="store_true", help="ignore saved trials and start over")
    args = ap.parse_args()

    name = args.model or json.loads((C.RESULTS_DIR / "final_selection.json").read_text())["selected_model"]
    space = zoo.SEARCH_SPACES[name]
    untuned = next(r for r in json.loads((C.RESULTS_DIR / "comparison_results.json").read_text()) if r["model"] == name)
    imb = untuned["config"]["imbalance"]

    train, test, feat_cols = load_train_test()
    folds = get_folds(train)
    Xtr, ytr = to_xy(train, feat_cols)
    print(f"Optimising {name}: {len(space)} hyperparameters, {args.n_trials} trials "
          f"({args.n_init} Sobol + {args.n_trials - args.n_init} GP-EI), objective = grouped-CV macro F1")

    def objective(params):
        cv = run_cv(lambda: zoo.build(name, params, imb), Xtr, ytr, train, folds)
        return dict(value=cv["cv_macro_f1_mean"],
                    extra=dict(cv_std=cv["cv_macro_f1_std"], cv_top1=cv["cv_top1_mean"],
                               cv_log_loss=cv["cv_log_loss_mean"], threshold=cv["threshold"]))

    path = C.RESULTS_DIR / "bayes_trials.json"
    if args.fresh and path.exists():
        path.unlink()
    opt = BayesOpt(space, objective, seed=C.SEED, n_init=args.n_init, trials_path=path)
    t0 = time.perf_counter(); best = opt.run(args.n_trials); search_s = time.perf_counter() - t0

    flat = [{"trial": t["trial"], "phase": t["phase"], "cv_macro_f1": t["value"], "seconds": t["seconds"],
             **t["params"], **{f"x_{k}": v for k, v in t["extra"].items()}} for t in opt.trials]
    pd.DataFrame(flat).to_csv(C.RESULTS_DIR / "bayes_trials.csv", index=False)
    save_json(dict(model=name, best_params=best["params"], best_cv_macro_f1=best["value"],
                   n_trials=len(opt.trials), n_init=args.n_init, search_seconds_this_session=search_s,
                   search_space=space, objective="mean macro F1 over trip-grouped CV folds (train trips only)",
                   seed=C.SEED), C.RESULTS_DIR / "bayes_best_params.json")

    # ---- refit with the best parameters and score exactly like the untuned model ----
    cfg = dict(params={**zoo.DEFAULTS[name], **best["params"]}, imbalance=imb)
    tuned, oof, p_te, model, _ = evaluate_model(
        name, lambda: zoo.build(name, cfg["params"], imb), train, test, feat_cols, folds, cfg, load_continuity_fn())
    rows = []
    for tag, r in [("untuned", untuned), ("tuned", tuned)]:
        rows.append({"variant": tag, "CV macro F1 (mean)": r["cv"]["cv_macro_f1_mean"], "CV macro F1 (std)": r["cv"]["cv_macro_f1_std"],
                     "Student-test Top-1": r["student_test"]["top1_accuracy"], "Student-test macro F1": r["student_test"]["macro_f1"],
                     "Student-test log loss": r["student_test"]["log_loss"], "Inference time (s)": r["inference_time_s"],
                     "Train time (s)": r["train_time_s"], "Config": json.dumps(r["config"]["params"], default=str)})
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(C.RESULTS_DIR / "bayes_comparison.csv", index=False)
    save_json(tuned, C.RESULTS_DIR / "tuned_model_record.json")
    print("\n", cmp_df.drop(columns="Config").to_string(index=False))

    # ---- choose what to ship: decided on CV only (not on student-test) ----
    use_tuned = tuned["cv"]["cv_macro_f1_mean"] >= untuned["cv"]["cv_macro_f1_mean"]
    if use_tuned:
        final_model, final_rec, final_oof, final_pte = model, tuned, oof, p_te
    else:                                                    # untuned wins on CV -> refit it for export
        _, final_oof, final_pte, final_model, _ = evaluate_model(
            name, lambda: zoo.build(name, untuned["config"]["params"], imb), train, test, feat_cols, folds, untuned["config"])
        final_rec = untuned
    C.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(pipeline=final_model, feature_cols=feat_cols, threshold=final_rec["cv"]["threshold"],
                     meta=dict(model=name, tuned=use_tuned, params=final_rec["config"]["params"], seed=C.SEED,
                               trained_on="80 train trips only", n_train_rows=final_rec["n_train_rows"])),
                C.ARTIFACTS_DIR / "final_model.joblib", compress=3)
    for df_, p, tag in [(train, final_oof, "train_oof"), (test, final_pte, "student_test")]:
        o = df_[C.KEY_COLS + [C.LABEL_COL]].copy(); o["prob"] = p
        o.to_csv(C.RESULTS_DIR / f"final_model_probs_{tag}.csv", index=False)
    save_json(dict(model=name, shipped="tuned" if use_tuned else "untuned", feature_cols=feat_cols,
                   threshold=final_rec["cv"]["threshold"]), C.ARTIFACTS_DIR / "final_model_card.json")
    print(f"\nExported artifacts/final_model.joblib ({'tuned' if use_tuned else 'untuned'} {name})")


if __name__ == "__main__":
    main()
