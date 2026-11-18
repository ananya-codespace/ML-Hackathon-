#!/usr/bin/env bash
# Person B pipeline, run from the repo root once A's tables are in data/processed/
set -e
python -m src.models.run_comparison --ablation   # 1. six models, grouped CV, student-test, timings, OOF probs
python -m src.models.select_final                # 2. mechanical selection rule -> results/final_selection.json
python -m src.models.run_bayes --n-trials 30     # 3. Bayesian optimisation, refit, tuned-vs-untuned, export
python -m src.models.plots                       # 4. all figures -> results/figures/
