import joblib
from src.models import config as C, zoo
from src.models.common import load_train_test, get_folds
from src.models.evaluation import evaluate_model
from src.models.reporting import save_json

train, test, feats = load_train_test(); folds = get_folds(train)
cfg = dict(params=zoo.DEFAULTS["hist_gb"], imbalance="none")
rec, oof, p_te, model, _ = evaluate_model("hist_gb", lambda: zoo.build("hist_gb", cfg["params"], "none"),
                                          train, test, feats, folds, cfg)
C.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
thr = rec["cv"]["threshold"]
joblib.dump(dict(pipeline=model, feature_cols=feats, threshold=thr,
                 meta=dict(model="hist_gb", tuned=False, params=cfg["params"], seed=C.SEED,
                           trained_on="78 clean train trips", n_train_rows=rec["n_train_rows"])),
            C.ARTIFACTS_DIR / "final_model.joblib", compress=3)
for d, p, tag in [(train, oof, "train_oof"), (test, p_te, "student_test")]:
    o = d[C.KEY_COLS + [C.LABEL_COL]].copy(); o["prob"] = p
    o.to_csv(C.RESULTS_DIR / f"final_model_probs_{tag}.csv", index=False)
save_json(rec, C.RESULTS_DIR / "untuned_final_record.json")
save_json(dict(model="hist_gb", shipped="untuned", feature_cols=feats, threshold=thr),
          C.ARTIFACTS_DIR / "final_model_card.json")
print("DONE. threshold", thr, "| test top1", rec["student_test"]["top1_accuracy"])
