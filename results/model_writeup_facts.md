# Per-model fact sheet (generated from results/comparison_results.json)

Numbers below are copied from saved result files. The heuristic flags are only prompts for you to look at the evidence - write your own interpretation for the report.


## Logistic Regression  (Linear / probabilistic)

- Hyperparameters (initial values): C=1.0, max_iter=500, imbalance=none
- Objective / loss: L2-regularised binary log loss (convex, solved by L-BFGS)
- Student-test: Top-1 = 0.8288, macro F1 = 0.8888, precision = 0.8081, recall = 0.8037, log loss = 0.1360
- Train (in-sample): Top-1 = 0.8253, macro F1 = 0.8866, log loss = 0.1433
- Grouped CV (5 folds by trip): macro F1 = 0.8849 +/- 0.0191, Top-1 = 0.8241 +/- 0.0405
- Timing: train 1.68s, inference 0.007s (0.0 us/row)
- Over/underfitting evidence: train->CV macro-F1 gap +0.0017; CV->student-test gap -0.0039
- Heuristic flags: none triggered

## Gaussian Naive Bayes  (Generative probabilistic)

- Hyperparameters (initial values): var_smoothing=1e-09, imbalance=none
- Objective / loss: Maximise class-conditional Gaussian likelihood (closed-form, no iterative loss)
- Student-test: Top-1 = 0.7920, macro F1 = 0.8277, precision = 0.6045, recall = 0.8556, log loss = 0.9985
- Train (in-sample): Top-1 = 0.7967, macro F1 = 0.8319, log loss = 0.9313
- Grouped CV (5 folds by trip): macro F1 = 0.8305 +/- 0.0173, Top-1 = 0.7954 +/- 0.0344
- Timing: train 1.28s, inference 0.011s (0.1 us/row)
- Over/underfitting evidence: train->CV macro-F1 gap +0.0014; CV->student-test gap +0.0029
- Heuristic flags: none triggered

## Decision Tree  (Single decision tree)

- Hyperparameters (initial values): max_depth=12, min_samples_leaf=50, imbalance=none
- Objective / loss: Gini impurity reduction per split (greedy); probability = leaf class frequency
- Student-test: Top-1 = 0.8439, macro F1 = 0.9032, precision = 0.7939, recall = 0.8747, log loss = 0.1105
- Train (in-sample): Top-1 = 0.8505, macro F1 = 0.9094, log loss = 0.1006
- Grouped CV (5 folds by trip): macro F1 = 0.8979 +/- 0.0221, Top-1 = 0.8302 +/- 0.0411
- Timing: train 10.98s, inference 0.011s (0.1 us/row)
- Over/underfitting evidence: train->CV macro-F1 gap +0.0115; CV->student-test gap -0.0053
- Heuristic flags: none triggered

## Random Forest  (Bagged tree ensemble)

- Hyperparameters (initial values): n_estimators=150, max_depth=None, min_samples_leaf=5, max_features=sqrt, max_samples=0.25, imbalance=none
- Objective / loss: Gini impurity reduction per split; probability = mean of tree leaf frequencies
- Student-test: Top-1 = 0.8425, macro F1 = 0.9059, precision = 0.8195, recall = 0.8537, log loss = 0.1039
- Train (in-sample): Top-1 = 0.8690, macro F1 = 0.9235, log loss = 0.0862
- Grouped CV (5 folds by trip): macro F1 = 0.9020 +/- 0.0221, Top-1 = 0.8346 +/- 0.0418
- Timing: train 17.51s, inference 0.193s (1.3 us/row)
- Over/underfitting evidence: train->CV macro-F1 gap +0.0214; CV->student-test gap -0.0039
- Heuristic flags: none triggered

## HistGradientBoosting  (Gradient boosting)

- Hyperparameters (initial values): learning_rate=0.1, max_iter=200, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0, imbalance=none
- Objective / loss: Binary log loss, minimised stage-wise with histogram-binned trees
- Student-test: Top-1 = 0.8392, macro F1 = 0.9028, precision = 0.8087, recall = 0.8547, log loss = 0.1020
- Train (in-sample): Top-1 = 0.8465, macro F1 = 0.9091, log loss = 0.1045
- Grouped CV (5 folds by trip): macro F1 = 0.9003 +/- 0.0214, Top-1 = 0.8332 +/- 0.0408
- Timing: train 3.53s, inference 0.055s (0.4 us/row)
- Over/underfitting evidence: train->CV macro-F1 gap +0.0088; CV->student-test gap -0.0024
- Heuristic flags: none triggered
