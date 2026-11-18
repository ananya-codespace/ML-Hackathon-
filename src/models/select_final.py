"""Step 2 - select the final model.   python -m src.models.select_final

Rule (written down BEFORE looking at results; applied mechanically, never on training accuracy):
  primary metric = mean grouped-CV macro F1 (macro F1 averages the 'true arc' and 'not true arc'
  classes, so the rare positive class counts as much as the abundant negatives).
  1. Keep models whose CV macro F1 is within one fold-standard-error of the best model.
  2. Among those, prefer lower CV log loss (probability quality - the HMM uses these as emissions).
  3. Break remaining ties with lower inference time, then simpler model.
Student-test numbers are shown for sanity but are NOT part of the rule.
"""
import json
import math

import pandas as pd

from . import config as C
from .reporting import save_json
from .zoo import INFO


def main():
    recs = json.loads((C.RESULTS_DIR / "comparison_results.json").read_text())
    df = pd.DataFrame([dict(model=r["model"], cv_f1=r["cv"]["cv_macro_f1_mean"], cv_std=r["cv"]["cv_macro_f1_std"],
                            cv_logloss=r["cv"]["cv_log_loss_mean"], infer_us=r["inference_us_per_row"],
                            test_top1=r["student_test"]["top1_accuracy"], test_f1=r["student_test"]["macro_f1"])
                       for r in recs])
    best = df.loc[df.cv_f1.idxmax()]
    se = best.cv_std / math.sqrt(C.N_FOLDS)
    df["within_1se_of_best"] = df.cv_f1 >= best.cv_f1 - se
    cand = df[df.within_1se_of_best].sort_values(["cv_logloss", "infer_us"])
    chosen = cand.iloc[0].model
    df["selected"] = df.model == chosen
    df.sort_values("cv_f1", ascending=False).to_csv(C.RESULTS_DIR / "model_selection_table.csv", index=False)
    save_json(dict(selected_model=chosen, label=INFO[chosen]["label"], rule=__doc__.split("Rule")[1].split('"""')[0].strip(),
                   best_cv_f1_model=best.model, best_cv_f1=best.cv_f1, one_se=se,
                   candidates_within_1se=cand.model.tolist()), C.RESULTS_DIR / "final_selection.json")
    print(df.sort_values("cv_f1", ascending=False).to_string(index=False))
    print(f"\nSelected: {INFO[chosen]['label']}  (candidates within 1 SE: {cand.model.tolist()})")


if __name__ == "__main__":
    main()
