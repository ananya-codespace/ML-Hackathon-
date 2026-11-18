# Person B - Models and Bayesian optimisation

Everything here runs on **Person A's candidate/feature tables**. The code is finished and was exercised end to end on
a *synthetic* table (tests/make_synthetic_feature_tables.py) to prove it runs. **No real result exists yet** - nothing
from the synthetic run may go in the report. When A's real tables land, run `./run_all_B.sh`.

## 1. Input contract (what A must export)
`data/processed/train_features.(csv|parquet)` (80 train trips) and `student_test_features.(csv|parquet)` (10 trips).
One row per (GPS point, candidate arc):

| column | meaning |
|---|---|
| `trip_id`, `obs_idx`, `arc_id` | keys |
| `label` | 1 if `arc_id` is the reference arc chosen for that GPS point, else 0 |
| everything else | numeric features (distance to arc, position along arc, heading diff, speed, time gap, arc length, candidate rank, neighbour context ...) |

Guards in `common.py`: label must be 0/1, no trip in both tables, same feature columns in both, numeric features only,
and any feature whose name looks like `label/truth/reference/ref_` is rejected (leakage). NaNs are fine (median-imputed, fit on train only).
Evaluator-trip data is never read by any B script.

Needed from A, otherwise B reports slightly optimistic numbers: **a candidate row for the reference arc must exist whenever possible**
(Candidate Miss Rate is computed here from the table as the share of GPS points with no `label=1` row). GPS points with *zero*
candidates are absent from the table, so they are not counted in Top-1 - ask A for a count of such points to quote beside it.

## 2. Run order and outputs
| step | command | key outputs in `results/` |
|---|---|---|
| 1 | `python -m src.models.run_comparison --ablation` | `comparison_results.json`, `comparison_table.csv` (report Table 3.1), `model_writeup_facts.md`, `imbalance_ablation.csv`, `mlp_history.json`, `oof/*_train_oof.csv` |
| 2 | `python -m src.models.select_final` | `final_selection.json`, `model_selection_table.csv` |
| 3 | `python -m src.models.run_bayes --n-trials 30` | `bayes_trials.csv/json` (every trial), `bayes_best_params.json`, `bayes_comparison.csv`, `tuned_model_record.json`, **`artifacts/final_model.joblib`**, `final_model_probs_train_oof.csv`, `final_model_probs_student_test.csv` |
| 4 | `python -m src.models.plots` | `figures/*.png` (CV macro F1, student-test Top-1/macro F1, log loss, over/underfit check, compute cost, confusion matrices, MLP loss curve, BO convergence, tuned vs untuned) |

`run_bayes` is resumable (saved trials are reloaded); use `--fresh` to restart. It is slow on few CPU cores: budget time for the
30 trials x 5 folds. Models: Logistic Regression, Gaussian NB, distance-weighted KNN, Random Forest, HistGradientBoosting, MLP.

## 3. Hand-offs
* **C (HMM)**: `results/final_model_probs_train_oof.csv` (out-of-fold, leak-free) and `final_model_probs_student_test.csv`, columns `trip_id, obs_idx, arc_id, label, prob`. Tune decoder parameters on these only. C's route-continuity function is auto-detected in `evaluation.load_continuity_fn()` (expects `src.hmm_viterbi.route_continuity(pred_df, split)`; edit the import if the name differs). Until then the continuity column is NaN. `metrics.adjacency_continuity` is a fallback.
* **D (API)**: `from src.models.inference import FinalClassifier; clf = FinalClassifier.load(); clf.predict_proba(table)` - load once at start-up. It selects the stored feature columns in training order, so the `label` column can never reach the model. Ship `artifacts/final_model.joblib` (+ `final_model_card.json`, `folds.json`).
* **Folds**: `artifacts/folds.json` (trip -> fold, seed 42). If A already created one it is used unchanged; otherwise B creates it (trips shuffled, then greedy row-balanced assignment - no trip is ever split).

## 4. Design decisions (viva cheat-sheet)
* **Primary metric = grouped-CV macro F1.** Rows are heavily imbalanced (few true arcs); macro F1 averages the F1 of "true arc" and "not true arc", so accuracy-by-predicting-all-negative scores ~0.5. Threshold for F1 is chosen on *train out-of-fold* predictions only (`metrics.best_threshold`), then frozen. Top-1 is rank-based (arg-max per GPS point) and needs no threshold - it is the metric closest to map-matching.
* **Imbalance**: default `none` (keeps probabilities honest for log loss and HMM emissions); `--ablation` measures class-weighting for LR / RF / HGB and saves `imbalance_ablation.csv`. Decide and document from that file.
* **Top-1** counts a GPS point wrong if the reference arc is not among its candidates (so it is bounded by 1 - miss rate); `top1_given_hit` is also saved.
* **Selection rule** (`select_final.py`): best CV macro F1; among models within one fold-standard-error of it, lowest CV log loss, then fastest inference. Student-test is *not* in the rule. Never training accuracy.
* **Bayesian optimisation** (`bayes_opt.py`, own implementation, only scikit-learn/scipy): 8 scrambled-Sobol points, then Gaussian-process (Matern 5/2 + noise) + Expected Improvement; log-uniform for learning rate / regularisation / leaf size. Objective = mean grouped-CV macro F1 on the 80 train trips; student-test only used afterwards to compare tuned vs untuned. Search spaces: `zoo.SEARCH_SPACES` (HGB: learning_rate 0.02-0.3 log, max_iter 100-500, max_leaf_nodes 8-96, min_samples_leaf 5-200 log, l2 1e-4-10 log - ranges bracket common good values while keeping trees small enough for fast inference). Caveat to state: the tuned CV score is selected on the same folds, so it is slightly optimistic; the student-test comparison is the unbiased one.
* **Ship decision**: tuned model is exported only if its CV macro F1 >= untuned (decided on CV, not on student-test). Final model is fit on the 80 train trips only.
* **Loss curves**: only the MLP has epoch history (train loss; scikit-learn gives validation *accuracy*, not loss). No curves are drawn for other models.

## 5. What you still have to write yourself
`results/model_writeup_facts.md` gives the numbers for Deliverables Q5-Q9 per model (hyperparameters, objective, test Top-1 / macro F1 / log loss,
train-vs-CV-vs-test gaps). The *reasoning* - why each algorithm suits PathSnap, strengths/limits, and your reading of over/underfitting - must be
written by you from your own results (the policy forbids unexplained AI-written analysis; also every member must defend it in the viva).
Disclose AI assistance in line with the event policy.

## 6. Not done / open
* Real results: blocked on A's feature tables. * Route continuity column: blocked on C. * Deadline/extra rules: the provided video is a 51 s intro clip (GPS noise -> map matching) with no deadline or rules, so check the original Drive video / organisers.
* `xgboost`/`lightgbm` were not used (not installable offline here); HistGradientBoosting covers the boosting family.
