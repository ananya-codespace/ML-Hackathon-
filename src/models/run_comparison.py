"""Step 1 - train + compare the six models.   python -m src.models.run_comparison [--ablation]

Writes (all under results/):
  comparison_results.json   full records (config, seed, CV folds, train/test metrics, timings)
  comparison_table.csv      Table 3.1 of the report
  model_writeup_facts.md    numbers for the per-model subsections
  oof/<model>_train_oof.csv out-of-fold train probabilities   (hand-off to C, leak-free)
  mlp_history.json          epoch history (only model with a real one)
  imbalance_ablation.csv    (with --ablation) class-weighting vs none
"""
import argparse
import time

import pandas as pd

from . import config as C
from . import zoo
from .common import get_folds, load_train_test
from .evaluation import evaluate_model, load_continuity_fn
from .reporting import comparison_table, save_json, writeup_facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=zoo.MODEL_ORDER)
    ap.add_argument("--imbalance", default="none", choices=["none", "balanced"])
    ap.add_argument("--ablation", action="store_true", help="also compare none vs balanced class weights")
    args = ap.parse_args()

    train, test, feat_cols = load_train_test()
    folds = get_folds(train)
    cont_fn = load_continuity_fn()
    print(f"train rows={len(train):,} (trips={train.trip_id.nunique()}), test rows={len(test):,} "
          f"(trips={test.trip_id.nunique()}), features={feat_cols}")
    print("route continuity:", "using C's function" if cont_fn else "C's function not found -> NaN for now")

    records = []
    for name in args.models:
        print(f"\n=== {name} ===")
        params = zoo.DEFAULTS[name]
        cfg = dict(params=params, imbalance=args.imbalance if zoo.INFO[name]["supports_imbalance"] else "none")
        t0 = time.perf_counter()
        rec, oof, p_te, model, _ = evaluate_model(
            name, lambda n=name, c=cfg: zoo.build(n, c["params"], c["imbalance"]),
            train, test, feat_cols, folds, cfg, cont_fn)
        rec["wall_s"] = time.perf_counter() - t0
        records.append(rec)
        print(f"  CV macro F1 {rec['cv']['cv_macro_f1_mean']:.4f}+/-{rec['cv']['cv_macro_f1_std']:.4f} | "
              f"test Top-1 {rec['student_test']['top1_accuracy']:.4f} macroF1 {rec['student_test']['macro_f1']:.4f} "
              f"logloss {rec['student_test']['log_loss']:.4f} | train {rec['train_time_s']:.1f}s")
        oofdir = C.RESULTS_DIR / "oof"; oofdir.mkdir(parents=True, exist_ok=True)
        o = train[C.KEY_COLS + [C.LABEL_COL]].copy(); o["prob"] = oof
        o.to_csv(oofdir / f"{name}_train_oof.csv", index=False)
        if zoo.INFO[name]["has_epoch_history"]:
            clf = model.named_steps["clf"]
            save_json(dict(loss_curve=clf.loss_curve_, validation_accuracy=getattr(clf, "validation_scores_", None),
                           n_iter=clf.n_iter_, note="sklearn MLP exposes train loss and validation ACCURACY (not validation loss)"),
                      C.RESULTS_DIR / "mlp_history.json")

    save_json(records, C.RESULTS_DIR / "comparison_results.json")
    comparison_table(records).to_csv(C.RESULTS_DIR / "comparison_table.csv", index=False)
    writeup_facts(records, C.RESULTS_DIR / "model_writeup_facts.md")

    if args.ablation:
        rows = []
        for name in [m for m in args.models if zoo.INFO[m]["supports_imbalance"]]:
            for imb in ["none", "balanced"]:
                cfg = dict(params=zoo.DEFAULTS[name], imbalance=imb)
                r, *_ = evaluate_model(name, lambda n=name, c=cfg: zoo.build(n, c["params"], c["imbalance"]),
                                       train, test, feat_cols, folds, cfg, cont_fn)
                rows.append(dict(model=name, imbalance=imb, cv_macro_f1=r["cv"]["cv_macro_f1_mean"],
                                 test_top1=r["student_test"]["top1_accuracy"], test_macro_f1=r["student_test"]["macro_f1"],
                                 test_log_loss=r["student_test"]["log_loss"], threshold=r["cv"]["threshold"]))
        pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "imbalance_ablation.csv", index=False)
    print("\nSaved results to", C.RESULTS_DIR)


if __name__ == "__main__":
    main()
