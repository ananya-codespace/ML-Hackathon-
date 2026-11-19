#!/usr/bin/env bash
# Person B pipeline, run from the repo root once A's tables are in data/processed/
set -e
python -m src.models.check_tables               # 0. pre-flight check of the real tables
python -m src.models.run_comparison --ablation --models logreg gaussian_nb decision_tree random_forest hist_gb   # 1. five models (KNN and MLP dropped: too slow on 1.4M rows), grouped CV, student-test, timings, OOF probs
python -m src.models.select_final                # 2. mechanical selection rule -> results/final_selection.json
python -m src.models.run_bayes --model hist_gb --n-trials 20 --n-init 6     # 3. Bayesian optimisation, refit, tuned-vs-untuned, export
python -m src.models.plots                       # 4. all figures -> results/figures/
